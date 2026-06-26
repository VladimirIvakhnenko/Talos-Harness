"""
app/agents/react_agent.py — LangGraph ReAct-агент.

Граф: Planner → Engineer → Retriever с общим состоянием.
Инструменты: search_memory, generate_st_code, validate_st_syntax, remember_fact.
"""
from __future__ import annotations
import uuid
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from app.agents.llm_client import get_llm, embed_single
from app.prompts.system_prompts import PLANNER_PROMPT, ENGINEER_PROMPT
from app.config import get_settings

settings = get_settings()


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:      list
    session_id:    str
    user_request:  str
    task_id:       str | None
    final_code:    str | None
    matiec_ok:     bool | None
    tokens:        dict


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
async def search_memory(query: str, session_id: str = "", top_k: int = 5) -> str:
    """Search documentation and chat history using hybrid dense+sparse retrieval."""
    from app.database import AsyncSessionLocal
    from app.memory.store import hybrid_search
    emb = await embed_single(query)
    async with AsyncSessionLocal() as db:
        results = await hybrid_search(db, query, emb, top_k=top_k)
    if not results:
        return "No relevant documentation found."
    return "\n\n---\n\n".join(
        f"[Source: {r['metadata'].get('source','?')}]\n{r['content']}"
        for r in results
    )


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


@tool
async def remember_fact(fact: str, session_id: str = "") -> str:
    """Save an important fact to long-term memory."""
    from app.database import AsyncSessionLocal
    from app.memory.store import save_fact
    emb = await embed_single(fact)
    async with AsyncSessionLocal() as db:
        await save_fact(db, fact, emb, session_id or "global")
    return "Fact saved."


TOOLS = [search_memory, generate_st_code, validate_st_syntax, remember_fact]
tool_node = ToolNode(TOOLS)


def build_initial_messages(user_message: str, context: str | None = None) -> list:
    """Сообщение пользователя — отдельный HumanMessage, целиком и без изменений."""
    messages: list = []
    if context and context.strip():
        messages.append(HumanMessage(content=f"[Контекст задачи]\n{context}"))
    messages.append(HumanMessage(content=user_message))
    return messages


def _planner_system(state: AgentState) -> str:
    req = state.get("user_request", "")
    parts = [PLANNER_PROMPT]
    if req:
        parts.append(
            "=== USER REQUEST (complete this task; do not ask what the user wants) ===\n"
            + req
        )
    return "\n\n".join(parts)


# ── Nodes ─────────────────────────────────────────────────────────────────────
async def planner_node(state: AgentState) -> AgentState:
    llm = get_llm("planner", session_id=state["session_id"],
                  task_id=state.get("task_id"), react_step="thought")
    llm_with_tools = llm.bind_tools(TOOLS)
    msgs = [SystemMessage(content=_planner_system(state))] + state["messages"]
    resp = await llm_with_tools.ainvoke(msgs)
    return {**state, "messages": state["messages"] + [resp]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


async def post_process(state: AgentState) -> AgentState:
    """Извлечь артефакт из generate_st_code или финального ответа."""
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


# ── Graph ─────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("planner",      planner_node)
    g.add_node("tools",        tool_node)
    g.add_node("post_process", post_process)

    g.set_entry_point("planner")
    g.add_conditional_edges("planner", should_continue, {"tools": "tools", END: "post_process"})
    g.add_edge("tools", "planner")
    g.add_edge("post_process", END)
    return g.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph():
    """Сброс кэша графа (после изменения tools)."""
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


def _format_planner_step(msg) -> str:
    lines = []
    text = _content_str(getattr(msg, "content", ""))
    if text.strip():
        lines.append(f"**Planner:** {text}")
    tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in tool_calls:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        lines.append(f"**Action:** `{name}({args})`")
    return "\n".join(lines) if lines else "**Planner:** (шаг)"


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


async def stream_agent(
    user_message: str,
    session_id: str | None = None,
    task_id: str | None = None,
    context: str | None = None,
):
    """Потоковый ReAct-агент: yield накопленный markdown лога шагов."""
    sid = session_id or str(uuid.uuid4())
    graph = get_graph()
    steps: list[str] = []
    final_state: dict = {}
    all_messages: list = []

    initial_state: AgentState = {
        "messages": build_initial_messages(user_message, context),
        "session_id": sid,
        "user_request": user_message,
        "task_id": task_id,
        "final_code": None,
        "matiec_ok": None,
        "tokens": {},
    }
    all_messages = list(initial_state["messages"])

    graph_config = {"recursion_limit": max(settings.agent_max_iterations * 3, 15)}

    async for update in graph.astream(initial_state, stream_mode="updates", config=graph_config):
        for node_name, patch in update.items():
            if node_name == "planner":
                msgs = patch.get("messages", [])
                if msgs:
                    all_messages.extend(msgs)
                    steps.append(_format_planner_step(msgs[-1]))
            elif node_name == "tools":
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
    """Запустить ReAct-агент и вернуть результат."""
    sid = session_id or str(uuid.uuid4())
    graph = get_graph()

    initial_state: AgentState = {
        "messages":   build_initial_messages(user_message, context),
        "session_id": sid,
        "user_request": user_message,
        "task_id":    task_id,
        "final_code": None,
        "matiec_ok":  None,
        "tokens":     {},
    }

    result = await graph.ainvoke(initial_state)
    last_msg = result["messages"][-1]

    return {
        "session_id": sid,
        "response":   _content_str(last_msg.content) if hasattr(last_msg, "content") else str(last_msg),
        "final_code": result.get("final_code"),
        "matiec_ok":  result.get("matiec_ok"),
        "steps":      len(result["messages"]),
    }


reset_graph()