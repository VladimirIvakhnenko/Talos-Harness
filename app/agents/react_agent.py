"""
app/agents/react_agent.py — LangGraph: Retrieval (deterministic) + Expert (ReAct).

Инструменты Expert: generate_st_code, validate_st_syntax + skill tools.
"""
from __future__ import annotations

import logging
import uuid
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, END

from app.agents.llm_client import get_llm, embed_single
from app.prompts.system_prompts import BASE_EXPERT_PROMPT, ENGINEER_PROMPT
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_RETRIEVAL_HISTORY_MESSAGES = 5


class AgentState(TypedDict):
    messages: list
    session_id: str
    user_request: str
    task_id: str | None
    upload_context: str | None
    retrieval_context: dict
    final_code: str | None
    matiec_ok: bool | None
    validation_attempts: int
    max_validation_attempts: int | None
    tokens: dict
    active_skills: list[str]


async def _generate_st_code_impl(
    spec: str,
    controller: str = "elbrus",
    session_id: str | None = None,
    task_id: str | None = None,
) -> str:
    from app.skills.prompt_builder import build_engineer_prompt

    active = _skill_registry.active_slugs if _skill_registry else None
    prompt = build_engineer_prompt(ENGINEER_PROMPT, _skill_registry, active)
    llm = get_llm(
        "engineer",
        session_id=session_id,
        task_id=task_id,
        tool_name="generate_st_code",
    )
    msgs = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Controller: {controller}\n\nSpecification:\n{spec}"),
    ]
    resp = await llm.ainvoke(msgs)
    return _content_str(resp.content)


@tool
async def generate_st_code(spec: str, controller: str = "elbrus") -> str:
    """Generate IEC 61131-3 Structured Text or PLCopen XML from specification."""
    return await _generate_st_code_impl(spec, controller)


@tool
async def validate_st_syntax(code: str = "", task_id: str = "agent_val") -> str:
    """Validate ST code syntax using MatIEC compiler (iec2c)."""
    from app.tools.matiec_client import compile_st

    if not code.strip():
        return "❌ No ST code to validate. Call generate_st_code first."
    result = await compile_st(code, task_id)
    if result.ok:
        return "✅ Syntax valid. Compilation rate: 1.0"
    errors = "\n".join(result.errors[:10])
    if not errors and result.stderr:
        errors = result.stderr.strip()[:2000]
    return f"❌ Syntax errors:\n{errors}"


EXPERT_TOOLS = [generate_st_code, validate_st_syntax]


async def _run_search_memory(query: str, session_id: str, top_k: int = 5) -> tuple[str, list[float], list[str]]:
    from app.database import AsyncSessionLocal
    from app.memory.store import search_documents

    emb = await embed_single(query)
    async with AsyncSessionLocal() as db:
        results, scores, sources = await search_documents(
            db, query, emb, session_id, top_k=top_k, with_scores=True
        )
    if not results:
        return "No relevant documentation found.", [], []
    text = "\n\n---\n\n".join(
        f"[Source: {r['metadata'].get('source', '?')} | scope: {r['metadata'].get('scope', '?')}]\n{r['content']}"
        for r in results
    )
    max_chars = settings.retrieval_doc_max_chars
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[... truncated ...]", scores, sources
    return text, scores, sources


async def retrieval_node(state: AgentState) -> AgentState:
    _log_node(state, "retrieval")
    from app.database import AsyncSessionLocal
    from app.memory.store import load_chat_history

    query = state.get("user_request", "")
    sid = state["session_id"]
    upload = (state.get("upload_context") or "").strip()

    document_chunks, retrieval_scores, retrieval_sources = await _run_search_memory(
        query, sid, top_k=settings.top_k_retrieval
    )
    history_msgs: list[dict] = []
    async with AsyncSessionLocal() as db:
        history_msgs = await load_chat_history(db, sid)
    recent = history_msgs[-_RETRIEVAL_HISTORY_MESSAGES:]
    recent_history = "\n".join(f"{m['role']}: {m['content']}" for m in recent) or "(empty)"

    doc_count = 0 if document_chunks.startswith("No relevant") else document_chunks.count("[Source:")
    retrieval_context = {
        "query": query,
        "document_chunks": document_chunks,
        "recent_history": recent_history,
        "upload_notes": upload,
        "doc_count": doc_count,
        "history_count": len(recent),
        "retrieval_scores": retrieval_scores,
        "retrieval_sources": retrieval_sources,
    }
    return {**state, "retrieval_context": retrieval_context}


def _tool_rounds(state: AgentState) -> int:
    return sum(
        1
        for m in state["messages"]
        if getattr(m, "type", "") == "ai" and getattr(m, "tool_calls", None)
    )


def _last_validation_ok(state: AgentState) -> bool:
    for m in reversed(state["messages"]):
        if getattr(m, "type", "") != "tool":
            continue
        if getattr(m, "name", "") != "validate_st_syntax":
            continue
        content = _content_str(getattr(m, "content", ""))
        return "Syntax valid" in content or content.startswith("✅")
    return False


def _has_generated_st_code(state: AgentState) -> bool:
    return _last_generated_st_code(state["messages"]) is not None


def _last_generated_st_code(messages: list) -> str | None:
    for m in reversed(messages):
        if getattr(m, "type", "") != "tool" or getattr(m, "name", "") != "generate_st_code":
            continue
        content = _content_str(getattr(m, "content", "")).strip()
        if not content:
            continue
        if any(kw in content for kw in ("PROGRAM", "FUNCTION_BLOCK", "FUNCTION")):
            return content
        if "<?xml" in content or "<project" in content.lower():
            return content
    return None


def _normalize_st(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.replace("\r\n", "\n").splitlines()).strip()


def _max_validation_attempts(state: AgentState) -> int:
    override = state.get("max_validation_attempts")
    if override is not None:
        return override
    return settings.expert_max_validation_attempts


def _generate_attempts(messages: list) -> int:
    return sum(
        1
        for m in messages
        if getattr(m, "type", "") == "tool" and getattr(m, "name", "") == "generate_st_code"
    )


def _first_validation_ok(messages: list) -> bool:
    seen_validate = 0
    for m in messages:
        if getattr(m, "type", "") != "tool" or getattr(m, "name", "") != "validate_st_syntax":
            continue
        seen_validate += 1
        content = _content_str(getattr(m, "content", ""))
        ok = "Syntax valid" in content or content.startswith("✅")
        if seen_validate == 1:
            return ok
    return False


def _agent_result_from_state(result: dict, sid: str) -> dict:
    messages = result.get("messages") or []
    last_msg = messages[-1] if messages else None
    retrieval_ctx = result.get("retrieval_context") or {}
    validation_attempts = result.get("validation_attempts", 0)
    matiec_ok = result.get("matiec_ok")
    return {
        "session_id": sid,
        "task_id": result.get("task_id"),
        "response": _content_str(last_msg.content) if last_msg and hasattr(last_msg, "content") else "",
        "final_code": result.get("final_code"),
        "matiec_ok": matiec_ok,
        "validation_attempts": validation_attempts,
        "generate_attempts": _generate_attempts(messages),
        "tool_rounds": _tool_rounds(result),
        "pass_at_1": validation_attempts == 1 and matiec_ok is True,
        "retrieval_context": retrieval_ctx,
        "steps": len(messages),
    }
def _compute_recursion_limit(state: AgentState | None = None) -> int:
    max_tr = settings.expert_max_iterations
    max_va = _max_validation_attempts(state) if state else settings.expert_max_validation_attempts
    return 2 + max(max_tr, max_va) * 2 + 4


def _log_node(state: AgentState, node_name: str) -> None:
    logger.info(
        "node=%s session=%s tool_rounds=%s validation_attempts=%s",
        node_name,
        (state.get("session_id") or "")[:8],
        _tool_rounds(state),
        state.get("validation_attempts", 0),
    )


def _resolve_task_id(session_id: str, task_id: str | None) -> str:
    if task_id:
        return task_id
    sid_prefix = session_id.replace("-", "")[:8]
    return f"{sid_prefix}_{uuid.uuid4().hex[:8]}"


def _expert_system(state: AgentState) -> str:
    # Build base prompt with skill injection
    from app.skills.prompt_builder import build_expert_prompt

    active_skills = state.get("active_skills") or []
    base = build_expert_prompt(BASE_EXPERT_PROMPT, _skill_registry, active_skills)

    # Append tool list dynamically
    tool_names = ["generate_st_code", "validate_st_syntax"]
    if _skill_registry:
        try:
            for t in _skill_registry.active_tools():
                tool_names.append(t.name)
        except Exception:
            pass
    base += f"\n\nTools: {', '.join(tool_names)}."

    base += (
        "\n\n=== OUTPUT FORMAT ===\n"
        "When you need to call a tool, output EXACTLY this format:\n"
        "Action: tool_name\n"
        "Action Input: {'param': 'value'}\n"
        "Example:\n"
        "Action: generate_st_code\n"
        "Action Input: {'spec': '...', 'controller': 'elbrus'}\n\n"
        "When the task is complete, respond with just your final answer.\n"
        "Never use 'Action:' or 'Action Input:' in your final answer."
    )

    parts = [base]
    req = state.get("user_request", "")
    if req:
        parts.append(
            "=== USER REQUEST (complete this task; do not ask what the user wants) ===\n"
            + req
        )
    ctx = state.get("retrieval_context") or {}
    ctx_parts = []
    if ctx.get("upload_notes"):
        ctx_parts.append("Upload / task notes:\n" + ctx["upload_notes"])
    if ctx.get("document_chunks"):
        ctx_parts.append("Documentation:\n" + ctx["document_chunks"])
    if ctx.get("recent_history"):
        ctx_parts.append("Recent chat history:\n" + ctx["recent_history"])
    if ctx_parts:
        parts.append("=== RETRIEVAL CONTEXT ===\n" + "\n\n".join(ctx_parts))

    va = state.get("validation_attempts", 0)
    tr = _tool_rounds(state)
    max_va = _max_validation_attempts(state)
    max_tr = settings.expert_max_iterations
    parts.append(
        "=== ATTEMPTS BUDGET ===\n"
        f"Validation attempts used: {va} / {max_va}\n"
        f"Tool rounds used: {tr} / {max_tr}\n"
        "If attempts remain and last validation failed: you MUST call generate_st_code again.\n"
        "If attempts exhausted: Final Answer only, no more tool_calls."
    )
    if va >= max_va or tr >= max_tr:
        parts.append(
            "=== LIMIT REACHED ===\n"
            "Attempt budget exhausted. Return Final Answer with the best available code "
            "WITHOUT calling any tools."
        )
    elif va >= max_va - 1 or tr >= max_tr - 1:
        parts.append(
            "=== LIMIT WARNING ===\n"
            "One attempt remaining. If validation fails again, return Final Answer "
            "with the best code on the next turn without tools."
        )
    return "\n\n".join(parts)


def _parse_tool_calls(text: str) -> list[dict]:
    """Parse ReAct-style tool calls from LLM text response.

    Expects lines in the form:
      Action: tool_name
      Action Input: {"param": "value"} or {'param': 'value'}
    Returns list of dicts with name/args/id keys.
    """
    import ast
    import json
    import re

    results: list[dict] = []
    pattern = re.compile(
        r"Action:\s*(\w[\w-]*)\s*\n\s*Action Input:\s*(\{.*?\}|\[.*?\]|`[^`]+`)",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        name = match.group(1)
        raw = match.group(2).strip().strip("`")
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            try:
                args = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                args = {"input": raw}
        results.append({
            "name": name,
            "args": args if isinstance(args, dict) else {"input": str(args)},
            "id": f"call_{name}_{len(results)}",
        })
    return results


async def expert_planner_node(state: AgentState) -> AgentState:
    _log_node(state, "expert_planner")
    llm = get_llm(
        "engineer",
        session_id=state["session_id"],
        task_id=state.get("task_id"),
        react_step="thought",
    )
    msgs = [SystemMessage(content=_expert_system(state))] + state["messages"]
    resp = await llm.ainvoke(msgs)
    text = _content_str(resp.content)

    tool_calls = _parse_tool_calls(text)
    if tool_calls:
        from langchain_core.messages import AIMessage
        resp = AIMessage(content=text, tool_calls=tool_calls)

    return {**state, "messages": state["messages"] + [resp]}


async def expert_tools_node(state: AgentState) -> AgentState:
    _log_node(state, "expert_tools")
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    tid = state.get("task_id") or "agent"
    validation_attempts = state.get("validation_attempts", 0)

    # Build dynamic dispatch map: builtin tools + skill tools
    builtin_handlers: dict[str, callable] = {
        "generate_st_code": _handle_generate_st_code,
        "validate_st_syntax": _handle_validate_st_syntax,
    }
    skill_handlers: dict[str, callable] = {}
    if _skill_registry:
        try:
            for t in _skill_registry.active_tools():
                skill_handlers[t.name] = t.ainvoke
        except Exception:
            pass

    new_msgs: list = []
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")

        if name in builtin_handlers:
            content, validation_attempts = await builtin_handlers[name](
                state, name, args, validation_attempts, tid
            )
        elif name in skill_handlers:
            content = await skill_handlers[name](args)
        else:
            content = f"Unknown tool: {name}"

        new_msgs.append(ToolMessage(content=str(content), tool_call_id=tc_id, name=name))

    return {
        **state,
        "messages": state["messages"] + new_msgs,
        "validation_attempts": validation_attempts,
    }


async def _handle_generate_st_code(
    state: AgentState, name: str, args: dict, validation_attempts: int, tid: str
) -> tuple[str, int]:
    content = await _generate_st_code_impl(
        args.get("spec", ""),
        args.get("controller", "elbrus"),
        state["session_id"],
        state.get("task_id"),
    )
    return content, validation_attempts


async def _handle_validate_st_syntax(
    state: AgentState, name: str, args: dict, validation_attempts: int, tid: str
) -> tuple[str, int]:
    last_code = _last_generated_st_code(state["messages"])
    requested = _normalize_st(args.get("code") or "")
    if last_code:
        if requested != _normalize_st(last_code):
            logger.warning(
                "validate_st_syntax: LLM code mismatch (req=%d chars, generated=%d chars) — using generate_st_code output",
                len(requested),
                len(last_code),
            )
        args = {**args, "code": last_code}
    elif requested:
        args = {**args, "code": requested}
    val_tid = f"{tid}_v{validation_attempts + 1}"
    args = {**args, "task_id": args.get("task_id", val_tid)}
    content = await validate_st_syntax.ainvoke(args)
    return content, validation_attempts + 1


def should_continue(state: AgentState) -> str:
    va = state.get("validation_attempts", 0)
    tr = _tool_rounds(state)
    if va >= _max_validation_attempts(state):
        logger.info("should_continue=END reason=validation_attempts session=%s", (state.get("session_id") or "")[:8])
        return END
    if tr >= settings.expert_max_iterations:
        logger.info("should_continue=END reason=tool_rounds session=%s", (state.get("session_id") or "")[:8])
        return END
    if _last_validation_ok(state) and _has_generated_st_code(state):
        logger.info("should_continue=END reason=validation_ok session=%s", (state.get("session_id") or "")[:8])
        return END

    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def _extract_final_code_from_messages(messages: list) -> tuple[str | None, bool]:
    code = None
    is_st = False

    for m in reversed(messages):
        if getattr(m, "type", "") != "tool" or getattr(m, "name", "") != "generate_st_code":
            continue
        content = _content_str(getattr(m, "content", ""))
        is_st = "PROGRAM" in content or "FUNCTION_BLOCK" in content
        is_xml = "<?xml" in content or "<project" in content.lower()
        if is_st or is_xml:
            code = content
        break

    if not code and messages:
        last_content = _content_str(getattr(messages[-1], "content", ""))
        is_st = "PROGRAM" in last_content or "FUNCTION_BLOCK" in last_content
        is_xml = "<?xml" in last_content or "<project" in last_content.lower()
        if is_st or is_xml:
            code = last_content

    return code, is_st


async def post_process(state: AgentState) -> AgentState:
    _log_node(state, "post_process")
    from app.tools.matiec_client import compile_st

    code, is_st = _extract_final_code_from_messages(state["messages"])

    matiec_ok = None
    if code and is_st:
        result = await compile_st(code, state.get("task_id") or "agent")
        matiec_ok = result.ok

    return {**state, "final_code": code, "matiec_ok": matiec_ok}


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("retrieval", retrieval_node)
    g.add_node("expert_planner", expert_planner_node)
    g.add_node("expert_tools", expert_tools_node)
    g.add_node("post_process", post_process)

    g.set_entry_point("retrieval")
    g.add_edge("retrieval", "expert_planner")
    g.add_conditional_edges(
        "expert_planner", should_continue, {"tools": "expert_tools", END: "post_process"}
    )
    g.add_edge("expert_tools", "expert_planner")
    g.add_edge("post_process", END)
    return g.compile()


_graph = None
_skill_registry: object = None  # set by set_skill_registry() from main.py


def set_skill_registry(registry: object) -> None:
    """Set the global skill registry reference (called from main.py lifespan)."""
    global _skill_registry
    _skill_registry = registry


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph():
    global _graph
    _graph = None


def _content_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _format_expert_step(msg) -> str:
    lines = []
    text = _content_str(getattr(msg, "content", ""))
    if text.strip():
        lines.append(f"**Expert:** {text}")
    for tc in getattr(msg, "tool_calls", None) or []:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        lines.append(f"**Action:** `{name}({args})`")
    return "\n".join(lines) if lines else "**Expert:** (шаг)"


def _format_retrieval_step(ctx: dict) -> str:
    doc_n = ctx.get("doc_count", 0)
    hist_n = ctx.get("history_count", 0)
    upload = "да" if ctx.get("upload_notes") else "нет"
    return f"**Retrieval:** {doc_n} фрагментов документации, {hist_n} сообщений истории, контекст загрузки: {upload}"


def _format_tool_observation(name: str, content: str) -> str:
    body = _content_str(content).strip()
    return f"**Observation** (`{name}`):\n\n```\n{body}\n```"


def _extract_final_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    marker = "Final Answer:"
    if marker in text:
        return text.split(marker, 1)[-1].strip()
    return text


def extract_user_facing_response(steps: list[str], final_state: dict) -> str:
    code = final_state.get("final_code")
    if code:
        if "<?xml" in code or "<project" in code.lower():
            return f"```xml\n{code}\n```"
        return f"```\n{code}\n```"
    for s in reversed(steps):
        if s.startswith("**Ответ:**"):
            return s.replace("**Ответ:**\n", "", 1)
    for s in reversed(steps):
        if s.startswith("**Expert:**"):
            text = s.replace("**Expert:**", "", 1).strip()
            if text and text != "(шаг)" and "tool" not in text.lower()[:20]:
                return _extract_final_answer(text)
    return "Готово."


def persist_text_from_agent_log(agent_log: str) -> str:
    """Краткий ответ для БД из полного markdown-лога агента."""
    import re

    steps = [s.strip() for s in agent_log.split("\n\n") if s.strip()]
    final_state: dict = {}
    for s in steps:
        if s.startswith("**MatIEC:**"):
            if "OK" in s:
                final_state["matiec_ok"] = True
            elif "ошибки" in s:
                final_state["matiec_ok"] = False
        if s.startswith("**Результат (ST):**") or s.startswith("**Результат (PLCopen XML):**"):
            m = re.search(r"```(?:xml)?\n(.*?)```", s, re.DOTALL)
            if m:
                final_state["final_code"] = m.group(1).strip()
    return extract_user_facing_response(steps, final_state)


def _build_initial_state(
    user_message: str,
    session_id: str,
    task_id: str | None,
    upload_context: str | None,
    max_validation_attempts: int | None = None,
    skills: list[str] | None = None,
) -> AgentState:
    resolved_task_id = _resolve_task_id(session_id, task_id)
    return {
        "messages": [HumanMessage(content=user_message)],
        "session_id": session_id,
        "user_request": user_message,
        "task_id": resolved_task_id,
        "upload_context": upload_context,
        "retrieval_context": {},
        "final_code": None,
        "matiec_ok": None,
        "validation_attempts": 0,
        "max_validation_attempts": max_validation_attempts,
        "tokens": {},
        "active_skills": skills or [],
    }


def _append_stream_update(
    node_name: str,
    patch: dict,
    steps: list[str],
    all_messages: list,
) -> None:
    if node_name == "retrieval":
        ctx = patch.get("retrieval_context", {})
        if ctx:
            steps.append(_format_retrieval_step(ctx))
    elif node_name == "expert_planner":
        msgs = patch.get("messages", [])
        if msgs:
            all_messages.extend(msgs)
            steps.append(_format_expert_step(msgs[-1]))
    elif node_name == "expert_tools":
        for m in patch.get("messages", []):
            all_messages.append(m)
            if getattr(m, "type", "") == "tool":
                name = getattr(m, "name", "tool")
                steps.append(_format_tool_observation(name, getattr(m, "content", "")))
    elif node_name == "post_process":
        pass


def _finalize_agent_steps(
    steps: list[str],
    all_messages: list,
    final_state: dict,
) -> list[str]:
    if all_messages:
        last = all_messages[-1]
        if getattr(last, "type", "") == "ai":
            answer = _extract_final_answer(_content_str(getattr(last, "content", "")))
            if answer and not any(s.startswith("**Ответ:**") for s in steps):
                steps.append(f"**Ответ:**\n{answer}")

    matiec = final_state.get("matiec_ok")
    if matiec is True:
        steps.append("**MatIEC:** OK")
    elif matiec is False:
        steps.append("**MatIEC:** ошибки компиляции")
    else:
        steps.append("**MatIEC:** не проверялся")

    code = final_state.get("final_code")
    if code and not any(s.startswith("**Результат") for s in steps):
        if "<?xml" in code or "<project" in code.lower():
            steps.append(f"**Результат (PLCopen XML):**\n\n```xml\n{code}\n```")
        else:
            steps.append(f"**Результат (ST):**\n\n```\n{code}\n```")
    elif code:
        steps.append(f"```\n{code}\n```")

    return steps


async def stream_agent(
    user_message: str,
    session_id: str | None = None,
    task_id: str | None = None,
    context: str | None = None,
    skills: list[str] | None = None,
):
    sid = session_id or str(uuid.uuid4())
    from app.agents.cancellation import AgentRunner

    runner = AgentRunner.get(sid)
    runner.cancel.clear()
    runner.running = True

    graph = get_graph()
    steps: list[str] = []
    final_state: dict = {}
    all_messages: list = []

    # Clear stale token data from previous runs
    from app.monitoring.token_tracker import drain_cycle_tokens
    drain_cycle_tokens()

    initial_state = _build_initial_state(user_message, sid, task_id, context, skills=skills)
    all_messages = list(initial_state["messages"])

    graph_config = {"recursion_limit": _compute_recursion_limit(initial_state)}

    try:
        async for update in graph.astream(initial_state, stream_mode="updates", config=graph_config):
            for node_name, patch in update.items():
                _append_stream_update(node_name, patch, steps, all_messages)
                if node_name == "post_process":
                    final_state.update(patch)

            if steps:
                yield "\n\n".join(steps)

            if runner.cancel.is_set():
                runner.cancel.clear()
                steps.append("**⏹ Остановлено пользователем.**")
                break
    except GraphRecursionError:
        logger.warning(
            "GraphRecursionError session=%s task_id=%s",
            sid[:8],
            initial_state.get("task_id"),
        )
        code, is_st = _extract_final_code_from_messages(all_messages)
        if code:
            final_state["final_code"] = code
            from app.tools.matiec_client import compile_st

            if is_st:
                result = await compile_st(code, initial_state.get("task_id") or "agent")
                final_state["matiec_ok"] = result.ok
        steps.append(
            "**⚠ Лимит итераций:** достигнут recursion_limit. "
            "Показан последний сгенерированный код."
        )

    steps = _finalize_agent_steps(steps, all_messages, final_state)
    yield "\n\n".join(steps)
    runner.running = False


async def run_agent(
    user_message: str,
    session_id: str | None = None,
    task_id: str | None = None,
    context: str | None = None,
    max_validation_attempts: int | None = None,
    skills: list[str] | None = None,
    route_skills: bool = False,
) -> dict:
    from app.agents.cancellation import AgentRunner

    sid = session_id or str(uuid.uuid4())
    graph = get_graph()
    runner = AgentRunner.get(sid)

    # If skill routing requested, auto-select skills via cosine similarity
    if route_skills and _skill_registry:
        from app.skills.router import route_skills as do_route

        selected = await do_route(user_message, _skill_registry)
        if selected:
            skills = selected
            logger.info("Routed skills for benchmark: %s", selected)

    async def _run_one(msg: str) -> dict:
        state = _build_initial_state(msg, sid, task_id, context, max_validation_attempts, skills=skills)
        cfg = {"recursion_limit": _compute_recursion_limit(state)}
        try:
            result = await graph.ainvoke(state, config=cfg)
        except GraphRecursionError:
            logger.warning("GraphRecursionError session=%s task_id=%s", sid[:8], state.get("task_id"))
            return {
                "session_id": sid, "task_id": state.get("task_id"),
                "response": "Достигнут лимит итераций.",
                "final_code": None, "matiec_ok": None,
                "validation_attempts": 0, "generate_attempts": 0,
                "tool_rounds": 0, "pass_at_1": False,
                "retrieval_context": {}, "steps": 0,
            }
        return _agent_result_from_state(result, sid)

    runner.running = True
    try:
        last_result = await _run_one(user_message)

        while next_msg := runner.drain():
            last_result = await _run_one(next_msg)

        return last_result
    finally:
        runner.running = False


reset_graph()
