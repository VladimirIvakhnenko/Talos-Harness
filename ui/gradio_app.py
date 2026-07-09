"""
ui/gradio_app.py — Gradio UI: список чатов + чат-агент + вложение файла.

Запуск: python -m ui.gradio_app
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from app.agents.llm_client import embed_single
from app.agents.react_agent import persist_text_from_agent_log, stream_agent
from app.api.deps import UPLOAD_DIR
from app.database import get_db
from app.memory.sessions import create_session, ensure_session, list_sessions, rename_session, touch_session
from app.memory.store import load_chat_history, next_turn_index, save_chat_message
from app.tools.pdf_processor import stream_process_pdf
from app.tools.signal_parser import parse_signal_table, signals_to_st_var
from app.tools.text_processor import stream_process_text_file

GRADIO_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

_FILE_HINT = "PDF, MD, TXT, CSV/XLSX или .st"


def _file_path(file) -> str | None:
    if file is None:
        return None
    path = file if isinstance(file, str) else getattr(file, "name", None)
    if path and Path(path).exists():
        return path
    return None


def _session_choices(sessions: list[dict]) -> list[tuple[str, str]]:
    return [(s.get("title") or "Без названия", s["id"]) for s in sessions]


def _fmt_tokens(tok: dict) -> str:
    cost_str = f"${tok['cost_usd']:.6f}" if tok["cost_usd"] else "free"
    return (
        f"⚡ prompt: **{tok['prompt_tokens']}** · "
        f"completion: **{tok['completion_tokens']}** · "
        f"{cost_str}"
    )


def _format_doc_event(ev: dict) -> str:
    phase = ev.get("phase")
    if phase == "read":
        return f"**Чтение** {ev.get('message', '')}"
    if phase == "convert":
        return f"**PDF** {ev.get('message', 'Конвертация…')}"
    if phase == "ocr":
        return f"**OCR** {ev.get('message', '')}"
    if phase == "chunk":
        return f"**Чанкинг** {ev.get('message', '')}"
    if phase == "embed":
        return f"**Векторизация** {ev.get('message', '')}"
    if phase == "done":
        r = ev.get("result", {})
        pages = r.get("pages", 0)
        if pages > 1:
            return f"**Готово:** {pages} стр., {r.get('chunks', 0)} фрагментов в памяти"
        return f"**Готово:** {r.get('chunks', 0)} фрагментов в памяти"
    return ev.get("message", "")


async def _ingest_non_pdf(path: str) -> str:
    suffix = Path(path).suffix.lower()
    name = Path(path).name

    if suffix in (".csv", ".xlsx", ".xls"):
        signals = parse_signal_table(path)
        var_section = signals_to_st_var(signals, "elbrus")
        return f"[Таблица сигналов «{name}»: {len(signals)} сигналов]\n{var_section}"

    if suffix == ".st":
        code = Path(path).read_text(encoding="utf-8", errors="replace")
        return f"[ST-файл «{name}»]\n{code[:8000]}"

    import uuid

    dest = UPLOAD_DIR / f"{uuid.uuid4()}_{name}"
    dest.write_bytes(Path(path).read_bytes())
    return f"[Файл «{name}» сохранён]"


def _assistant_content(ingest_log: list[str], agent_log: str) -> str:
    parts = []
    if ingest_log:
        parts.append("\n\n".join(ingest_log))
    if agent_log:
        if parts:
            parts.append("---")
        parts.append(agent_log)
    return "\n\n".join(parts) if parts else "⏳ Обработка…"


def _patch_assistant(history: list, content: str) -> list:
    out = [dict(m) for m in history]
    out[-1] = {**out[-1], "content": content}
    return out


async def init_ui():
    async with get_db() as db:
        sessions = await list_sessions(db, limit=50)
        if not sessions:
            sid = await create_session(db)
            sessions = await list_sessions(db, limit=50)
        else:
            sid = sessions[0]["id"]
        history = await load_chat_history(db, sid)
    choices = _session_choices(sessions)
    return gr.update(choices=choices, value=sid), sid, history, ""


async def switch_chat(session_id: str):
    if not session_id:
        return [], ""
    async with get_db() as db:
        history = await load_chat_history(db, session_id)
    return history, session_id, ""


async def new_chat():
    async with get_db() as db:
        sid = await create_session(db)
        sessions = await list_sessions(db, limit=50)
    return gr.update(choices=_session_choices(sessions), value=sid), sid, [], ""


async def chat_fn(message: str, file, history: list, session_id: str, token_html: str = ""):
    from app.agents.cancellation import AgentRunner

    path = _file_path(file)
    user_display = message if message else (f"📎 {Path(path).name}" if path else "")
    if not user_display and not path:
        return

    history = list(history or [])
    ingest_log: list[str] = []
    context_parts: list[str] = []
    agent_log = ""

    async with get_db() as db:
        sid = await ensure_session(db, session_id or None)

    runner = AgentRunner.get(sid)

    async def _run_agent(msg: str, ctx: str | None = None) -> str:
        nonlocal agent_log
        agent_log = ""
        async for agent_partial in stream_agent(msg, session_id=sid, context=ctx):
            agent_log = agent_partial
            yield agent_partial

    async def _persist(msg: str, display: str, log: str):
        persisted = persist_text_from_agent_log(log) if log else "Готово."
        async with get_db() as db:
            ti = await next_turn_index(db, sid)
            emb_u = await embed_single(display)
            await save_chat_message(db, sid, "user", display, emb_u, ti)
            emb_a = await embed_single(persisted[:4000] or " ")
            await save_chat_message(db, sid, "assistant", persisted, emb_a, ti)
            await touch_session(db, sid)
            if ti == 0 and msg.strip():
                await rename_session(db, sid, msg.strip()[:60])

    history.append({"role": "user", "content": user_display})
    history.append({"role": "assistant", "content": "⏳ Обработка…"})
    yield history, sid, None, gr.update(), ""

    try:
        if path:
            suffix = Path(path).suffix.lower()
            name = Path(path).name

            if suffix in (".pdf", ".md", ".txt"):
                async with get_db() as db:
                    stream = (
                        stream_process_pdf(
                            path,
                            doc_type="general",
                            source_name=name,
                            db=db,
                            session_id=sid,
                            scope="session",
                        )
                        if suffix == ".pdf"
                        else stream_process_text_file(
                            path,
                            doc_type="general",
                            source_name=name,
                            db=db,
                            session_id=sid,
                            scope="session",
                        )
                    )
                    async for ev in stream:
                        line = _format_doc_event(ev)
                        if ev.get("phase") == "done":
                            ingest_log = [line]
                            r = ev.get("result", {})
                            pages_note = (
                                f"{r.get('pages', 0)} стр., "
                                if r.get("pages", 0) > 1
                                else ""
                            )
                            context_parts.append(
                                f"[Документ «{name}» в памяти чата: "
                                f"{pages_note}{r.get('chunks', 0)} фрагментов]"
                            )
                        else:
                            ingest_log = [line]
                        history = _patch_assistant(
                            history, _assistant_content(ingest_log, "")
                        )
                        yield history, sid, None, gr.update(), ""
            else:
                ctx = await _ingest_non_pdf(path)
                ingest_log = [f"**Файл** {ctx[:500]}{'…' if len(ctx) > 500 else ''}"]
                context_parts.append(ctx)
                history = _patch_assistant(
                    history, _assistant_content(ingest_log, "")
                )
                yield history, sid, None, gr.update(), ""

        task_context = "\n\n".join(context_parts) if context_parts else None

        history = _patch_assistant(
            history, _assistant_content(ingest_log, "**Агент** запуск…")
        )
        yield history, sid, None, gr.update(), ""

        async for agent_partial in _run_agent(message, task_context):
            history = _patch_assistant(
                history, _assistant_content(ingest_log, agent_partial)
            )
            from app.monitoring.token_tracker import drain_cycle_tokens
            tok = drain_cycle_tokens()
            tok_line = _fmt_tokens(tok) if tok.get("calls") else ""
            yield history, sid, None, gr.update(), tok_line

        await _persist(message, user_display, agent_log)

        while next_msg := runner.drain():
            history.append({"role": "user", "content": next_msg})
            history.append({"role": "assistant", "content": "⏳ Обработка из очереди…"})
            yield history, sid, None, gr.update(), ""

            agent_log = ""
            async for agent_partial in _run_agent(next_msg):
                history = _patch_assistant(
                    history, _assistant_content(ingest_log, agent_partial)
                )
                from app.monitoring.token_tracker import drain_cycle_tokens
                tok = drain_cycle_tokens()
                tok_line = _fmt_tokens(tok) if tok.get("calls") else ""
                yield history, sid, None, gr.update(), tok_line

            await _persist(next_msg, next_msg, agent_log)

        async with get_db() as db:
            sessions = await list_sessions(db, limit=50)

        yield history, sid, None, gr.update(
            choices=_session_choices(sessions), value=sid
        ), ""

    except Exception as e:
        history = _patch_assistant(
            history,
            _assistant_content(ingest_log, f"**Ошибка:** {type(e).__name__}: {e}"),
        )
        yield history, sid, None, gr.update(), ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Talos Harness", theme=gr.themes.Soft(primary_hue="cyan")) as demo:
        gr.Markdown("# Talos Harness\nГенерация ST-кода для ПЛК")

        session_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                chat_list = gr.Radio(
                    label="Чаты",
                    choices=[],
                    interactive=True,
                )
                new_chat_btn = gr.Button("+ Новый чат", variant="secondary")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Чат",
                    height=480,
                    show_copy_button=True,
                    type="messages",
                    render_markdown=True,
                    sanitize_html=False,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Опишите задачу или прикрепите файл…",
                        show_label=False,
                        scale=4,
                        container=False,
                    )
                    file_input = gr.File(
                        label="Файл",
                        file_types=[".pdf", ".md", ".txt", ".csv", ".xlsx", ".xls", ".st"],
                        scale=1,
                        type="filepath",
                    )

                with gr.Row():
                    send_btn = gr.Button("Отправить", variant="primary")
                    stop_btn = gr.Button("⏹ Стоп", variant="stop")

                token_display = gr.Markdown(
                    value="",
                    visible=True,
                    height=24,
                )

                queue_display = gr.Markdown(
                    value="",
                    visible=True,
                    height=24,
                )

        gr.Markdown(f"_{_FILE_HINT}_")

        def _stop_agent(session_id: str):
            if not session_id:
                return "Нет активной сессии"
            from app.agents.cancellation import AgentRunner
            runner = AgentRunner.get(session_id)
            if not runner.running:
                return "Агент не выполняется"
            runner.request_cancel()
            return "⏹ Остановка…"

        def _check_queue(session_id: str):
            if not session_id:
                return ""
            from app.agents.cancellation import AgentRunner
            runner = AgentRunner.get(session_id)
            parts = []
            if runner.running:
                parts.append("🔄 Агент выполняется…")
            if runner.queue_size():
                parts.append(f"📋 В очереди: {runner.queue_size()}")
            return " · ".join(parts)

        demo.load(init_ui, outputs=[chat_list, session_state, chatbot, token_display])

        chat_outputs = [chatbot, session_state, file_input, chat_list, token_display]
        chat_inputs = [msg_input, file_input, chatbot, session_state, token_display]

        send_btn.click(chat_fn, chat_inputs, chat_outputs).then(
            lambda: "", outputs=msg_input
        ).then(
            _check_queue, inputs=[session_state], outputs=[queue_display]
        )
        msg_input.submit(chat_fn, chat_inputs, chat_outputs).then(
            lambda: "", outputs=msg_input
        ).then(
            _check_queue, inputs=[session_state], outputs=[queue_display]
        )

        stop_btn.click(_stop_agent, inputs=[session_state], outputs=[queue_display])
        chat_list.change(switch_chat, inputs=[chat_list], outputs=[chatbot, session_state, token_display])
        new_chat_btn.click(new_chat, outputs=[chat_list, session_state, chatbot, token_display])

    return demo


def main():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, show_error=True)


if __name__ == "__main__":
    main()
