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
    messages:   list
    session_id: str
    task_id:    str | None
    final_code: str | None
    matiec_ok:  bool | None
    tokens:     dict


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
    """Generate IEC 61131-3 Structured Text code from specification."""
    llm = get_llm("engineer", tool_name="generate_st_code")
    from app.prompts.system_prompts import ENGINEER_PROMPT
    msgs = [
        SystemMessage(content=ENGINEER_PROMPT),
        HumanMessage(content=f"Controller: {controller}\n\nSpecification:\n{spec}"),
    ]
    resp = await llm.ainvoke(msgs)
    return resp.content


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


# ── Nodes ─────────────────────────────────────────────────────────────────────
async def planner_node(state: AgentState) -> AgentState:
    llm = get_llm("planner", session_id=state["session_id"],
                  task_id=state.get("task_id"), react_step="thought")
    llm_with_tools = llm.bind_tools(TOOLS)
    msgs = [SystemMessage(content=PLANNER_PROMPT)] + state["messages"]
    resp = await llm_with_tools.ainvoke(msgs)
    return {**state, "messages": state["messages"] + [resp]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


async def post_process(state: AgentState) -> AgentState:
    """Extract generated code and run MatIEC validation."""
    from app.tools.matiec_client import compile_st
    last_content = state["messages"][-1].content if state["messages"] else ""

    # Извлекаем ST-код из финального ответа
    code = None
    if "PROGRAM" in last_content or "FUNCTION_BLOCK" in last_content:
        code = last_content

    matiec_ok = None
    if code:
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


async def run_agent(message: str, session_id: str | None = None,
                    task_id: str | None = None) -> dict:
    """Запустить ReAct-агент и вернуть результат."""
    sid = session_id or str(uuid.uuid4())
    graph = get_graph()

    initial_state: AgentState = {
        "messages":   [HumanMessage(content=message)],
        "session_id": sid,
        "task_id":    task_id,
        "final_code": None,
        "matiec_ok":  None,
        "tokens":     {},
    }

    result = await graph.ainvoke(initial_state)
    last_msg = result["messages"][-1]

    return {
        "session_id": sid,
        "response":   last_msg.content if hasattr(last_msg, "content") else str(last_msg),
        "final_code": result.get("final_code"),
        "matiec_ok":  result.get("matiec_ok"),
        "steps":      len(result["messages"]),
    }