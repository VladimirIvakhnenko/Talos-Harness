"""
ui/gradio_app.py — минималистичный Gradio UI: чат-агент + вложение файла.

Запуск: python -m ui.gradio_app
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import gradio as gr

from app.agents.react_agent import stream_agent
from app.api.deps import UPLOAD_DIR
from app.database import get_db
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
    """Новая копия history — Gradio надёжнее обновляет чат без мутации на месте."""
    out = [dict(m) for m in history]
    out[-1] = {**out[-1], "content": content}
    return out


def _prior_user_messages(history: list) -> list[str]:
    """Все предыдущие реплики пользователя целиком (без текущего хода)."""
    return [
        m.get("content") or ""
        for m in history[:-2]
        if m.get("role") == "user"
    ]


def _build_task_context(prior_user_messages: list[str], context_parts: list[str]) -> str:
    sections: list[str] = []
    if prior_user_messages:
        sections.append(
            "[Предыдущие сообщения пользователя]\n"
            + "\n---\n\n".join(prior_user_messages)
        )
    if context_parts:
        sections.append("[Контекст задачи]\n" + "\n\n".join(context_parts))
    return "\n\n".join(sections)


async def chat_fn(message: str, file, history: list, session_id: str):
    path = _file_path(file)
    user_display = message if message else (f"📎 {Path(path).name}" if path else "")
    if not user_display and not path:
        return

    history = list(history or [])
    sid = session_id or str(uuid.uuid4())
    ingest_log: list[str] = []
    context_parts: list[str] = []
    prior_msgs = _prior_user_messages(history)

    history.append({"role": "user", "content": user_display})
    history.append({"role": "assistant", "content": "⏳ Обработка…"})
    yield history, sid, None

    try:
        if path:
            suffix = Path(path).suffix.lower()
            name = Path(path).name

            if suffix in (".pdf", ".md", ".txt"):
                async with get_db() as db:
                    stream = (
                        stream_process_pdf(path, doc_type="general", source_name=name, db=db)
                        if suffix == ".pdf"
                        else stream_process_text_file(path, doc_type="general", source_name=name, db=db)
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
                                f"[Документ «{name}» в памяти: "
                                f"{pages_note}{r.get('chunks', 0)} фрагментов]"
                            )
                        else:
                            ingest_log = [line]
                        history = _patch_assistant(
                            history, _assistant_content(ingest_log, "")
                        )
                        yield history, sid, None
            else:
                ctx = await _ingest_non_pdf(path)
                ingest_log = [f"**Файл** {ctx[:500]}{'…' if len(ctx) > 500 else ''}"]
                context_parts.append(ctx)
                history = _patch_assistant(
                    history, _assistant_content(ingest_log, "")
                )
                yield history, sid, None

        task_context = _build_task_context(prior_msgs, context_parts)

        history = _patch_assistant(
            history, _assistant_content(ingest_log, "**Агент** запуск…")
        )
        yield history, sid, None

        async for agent_partial in stream_agent(
            message,
            session_id=sid,
            context=task_context or None,
        ):
            history = _patch_assistant(
                history, _assistant_content(ingest_log, agent_partial)
            )
            yield history, sid, None

    except Exception as e:
        history = _patch_assistant(
            history,
            _assistant_content(ingest_log, f"**Ошибка:** {type(e).__name__}: {e}"),
        )
        yield history, sid, None


def clear_chat():
    return [], "", None


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Talos Harness", theme=gr.themes.Soft(primary_hue="cyan")) as demo:
        gr.Markdown("# Talos Harness\nГенерация ST-кода для ПЛК")

        session_state = gr.State("")
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
            clear_btn = gr.Button("Очистить")

        outputs = [chatbot, session_state, file_input]
        inputs = [msg_input, file_input, chatbot, session_state]

        send_btn.click(chat_fn, inputs, outputs).then(lambda: "", outputs=msg_input)
        msg_input.submit(chat_fn, inputs, outputs).then(lambda: "", outputs=msg_input)
        clear_btn.click(clear_chat, outputs=[chatbot, session_state, file_input])

        gr.Markdown(f"_{_FILE_HINT}_")

    return demo


def main():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, show_error=True)


if __name__ == "__main__":
    main()
