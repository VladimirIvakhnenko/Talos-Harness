"""
app/agents/react_agent.py — LangGraph: Retrieval (deterministic) + Expert (ReAct).

Инструменты Expert: generate_st_code, validate_st_syntax.
"""
from __future__ import annotations

import uuid
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from app.agents.llm_client import get_llm, embed_single
from app.prompts.system_prompts import EXPERT_PROMPT, ENGINEER_PROMPT
from app.config import get_settings

settings = get_settings()

_RETRIEVAL_DOC_MAX_CHARS = 8000
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
    tokens: dict


@tool
async def generate_st_code(spec: str, controller: str = "elbrus") -> str:
    """Generate IEC 61131-3 Structured Text or PLCopen XML from specification."""
    llm = get_llm("engineer", tool_name="generate_st_code")
    msgs = [
        SystemMessage(content=ENGINEER_PROMPT),
        HumanMessage(content=f"Controller: {controller}\n\nSpecification:\n{spec}"),
    ]
    resp = await llm.ainvoke(msgs)
    return _content_str(resp.content)


@tool
async def validate_st_syntax(code: str, task_id: str = "agent_val") -> str:
    """Validate ST code syntax using MatIEC compiler."""
    from app.tools.matiec_client import validate_st
    result = await validate_st(code, task_id)
    if result.ok:
        return "✅ Syntax valid. Compilation rate: 1.0"
    errors = "\n".join(result.errors[:10])
    return f"❌ Syntax errors:\n{errors}"


EXPERT_TOOLS = [generate_st_code, validate_st_syntax]


async def _run_search_memory(query: str, session_id: str, top_k: int = 5) -> str:
    from app.database import AsyncSessionLocal
    from app.memory.store import search_documents

    emb = await embed_single(query)
    async with AsyncSessionLocal() as db:
        results = await search_documents(db, query, emb, session_id, top_k=top_k)
    if not results:
        return "No relevant documentation found."
    text = "\n\n---\n\n".join(
        f"[Source: {r['metadata'].get('source', '?')} | scope: {r['metadata'].get('scope', '?')}]\n{r['content']}"
        for r in results
    )
    if len(text) > _RETRIEVAL_DOC_MAX_CHARS:
        return text[:_RETRIEVAL_DOC_MAX_CHARS] + "\n\n[... truncated ...]"
    return text


async def retrieval_node(state: AgentState) -> AgentState:
    from app.database import AsyncSessionLocal
    from app.memory.store import load_chat_history

    query = state.get("user_request", "")
    sid = state["session_id"]
    upload = (state.get("upload_context") or "").strip()

    document_chunks = await _run_search_memory(
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
    }
    return {**state, "retrieval_context": retrieval_context}


def _expert_system(state: AgentState) -> str:
    parts = [EXPERT_PROMPT]
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
    return "\n\n".join(parts)


async def expert_planner_node(state: AgentState) -> AgentState:
    llm = get_llm(
        "engineer",
        session_id=state["session_id"],
        task_id=state.get("task_id"),
        react_step="thought",
    )
    llm_with_tools = llm.bind_tools(EXPERT_TOOLS)
    msgs = [SystemMessage(content=_expert_system(state))] + state["messages"]
    resp = await llm_with_tools.ainvoke(msgs)
    return {**state, "messages": state["messages"] + [resp]}


async def expert_tools_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    tid = state.get("task_id") or "agent"
    validation_attempts = state.get("validation_attempts", 0)

    new_msgs: list = []
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")

        if name == "generate_st_code":
            content = await generate_st_code.ainvoke(args)
        elif name == "validate_st_syntax":
            args = {**args, "task_id": args.get("task_id", tid)}
            content = await validate_st_syntax.ainvoke(args)
            validation_attempts += 1
        else:
            content = f"Unknown tool: {name}"

        new_msgs.append(ToolMessage(content=str(content), tool_call_id=tc_id, name=name))

    return {
        **state,
        "messages": state["messages"] + new_msgs,
        "validation_attempts": validation_attempts,
    }


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


async def post_process(state: AgentState) -> AgentState:
    from app.tools.matiec_client import compile_st

    code = None
    is_st = False

    for m in reversed(state["messages"]):
        if getattr(m, "type", "") != "tool" or getattr(m, "name", "") != "generate_st_code":
            continue
        content = _content_str(getattr(m, "content", ""))
        is_st = "PROGRAM" in content or "FUNCTION_BLOCK" in content
        is_xml = "<?xml" in content or "<project" in content.lower()
        if is_st or is_xml:
            code = content
        break

    if not code and state["messages"]:
        last_content = _content_str(state["messages"][-1].content)
        is_st = "PROGRAM" in last_content or "FUNCTION_BLOCK" in last_content
        is_xml = "<?xml" in last_content or "<project" in last_content.lower()
        if is_st or is_xml:
            code = last_content

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
) -> AgentState:
    return {
        "messages": [HumanMessage(content=user_message)],
        "session_id": session_id,
        "user_request": user_message,
        "task_id": task_id,
        "upload_context": upload_context,
        "retrieval_context": {},
        "final_code": None,
        "matiec_ok": None,
        "validation_attempts": 0,
        "tokens": {},
    }


async def stream_agent(
    user_message: str,
    session_id: str | None = None,
    task_id: str | None = None,
    context: str | None = None,
):
    sid = session_id or str(uuid.uuid4())
    graph = get_graph()
    steps: list[str] = []
    final_state: dict = {}
    all_messages: list = []

    initial_state = _build_initial_state(user_message, sid, task_id, context)
    all_messages = list(initial_state["messages"])

    graph_config = {
        "recursion_limit": max(settings.expert_max_iterations * 2 + 2, 12),
    }

    async for update in graph.astream(initial_state, stream_mode="updates", config=graph_config):
        for node_name, patch in update.items():
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
                final_state.update(patch)

        if steps:
            yield "\n\n".join(steps)

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

    yield "\n\n".join(steps)


async def run_agent(
    user_message: str,
    session_id: str | None = None,
    task_id: str | None = None,
    context: str | None = None,
) -> dict:
    sid = session_id or str(uuid.uuid4())
    graph = get_graph()

    initial_state = _build_initial_state(user_message, sid, task_id, context)
    graph_config = {
        "recursion_limit": max(settings.expert_max_iterations * 2 + 2, 12),
    }

    result = await graph.ainvoke(initial_state, config=graph_config)
    last_msg = result["messages"][-1] if result.get("messages") else None

    return {
        "session_id": sid,
        "response": _content_str(last_msg.content) if last_msg and hasattr(last_msg, "content") else "",
        "final_code": result.get("final_code"),
        "matiec_ok": result.get("matiec_ok"),
        "steps": len(result.get("messages", [])),
    }


reset_graph()
