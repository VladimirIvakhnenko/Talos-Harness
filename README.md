# Talos Harness

**Гибкий оркестратор агентов для генерации и верификации IEC 61131-3 ST-кода для ПЛК**

Talos Harness — инфраструктурный каркас (harness), соединяющий LLM-агентов, инструменты, RAG-память и валидатор в единый workflow. Генерирует Structured Text и PLCopen XML, валидирует через MatIEC, ищет контекст в документации через pgvector. Поддерживает динамическое подключение скиллов — модульных расширений, меняющих поведение агента.

---

## Архитектура

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Skill Router (автоматический выбор)              │
│  эмбеддинг запроса → cosine similarity            │
│  с description скиллов → топ-3 > 0.45             │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Retrieval (без LLM)                             │
│  dense + hybrid RAG (pgvector)                   │
│  история чата + контекст загрузки                │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Expert (ReAct)                                  │
│  generate_st_code → validate_st (MatIEC)         │
│  + выбранные скиллы (st-style-guide, …)          │
│  итерации до EXPERT_MAX_ITERATIONS               │
└──────────────────────────────────────────────────┘
    │
    ▼
  ST / PLCopen XML + MatIEC verification
```

| Компонент | Назначение |
|-----------|-----------|
| **Skill Router** | Автоматический выбор скиллов: эмбеддинг запроса → cosine similarity с `description` скиллов → топ-3 выше порога 0.45. Реализация: `app/skills/router.py` |
| **Retrieval** | Детерминированный сбор контекста: dense RAG (pgvector, cosine), tsvector keyword search, RRF fusion, parent-child чанкинг |
| **Expert** | ReAct-цикл: LLM + инструменты `generate_st_code` и `validate_st_syntax`, итеративное исправление по ошибкам MatIEC |
| **Skills** | Динамически подключаемые модули: Markdown-промпты, опционально Python-инструменты. Активируются автоматически или вручную через API |
| **MatIEC** | `iec2c` / `iec2iec` — синтаксическая валидация и компиляция ST-кода |
| **Эмбеддинги** | llama.cpp (GGUF) локально или OpenRouter API |
| **LLM** | OpenRouter / LM Studio (OpenAI-compatible). Поддерживаются любые модели |
| **Память** | PostgreSQL 17 + pgvector, персистентные сессии чатов, parent-child документы |

Документы из UI индексируются с `scope=session`; через API — `scope=global`.

---

## API Endpoints

### Agent

| Method | Path | Описание |
|--------|------|---------|
| `POST` | `/chat` | ReAct-агент: генерация ST-кода по текстовому запросу. Body: `{ message, session_id?, skills? }` |
| `POST` | `/generate_module` | Генерация ST-модуля (PROGRAM) по CSV-таблице сигналов |
| `GET` | `/module/download` | Скачать сгенерированный `.st` файл |
| `POST` | `/validate` | Валидация ST-кода через MatIEC |

### Skills

| Method | Path | Описание |
|--------|------|---------|
| `GET` | `/skills` | Список всех обнаруженных скиллов (статус, превью промпта) |
| `GET` | `/skills/active` | Только активные скиллы |
| `GET` | `/skills/{slug}` | Детальная информация: полный промпт, инструменты, зависимости |
| `POST` | `/skills/{slug}/activate` | Активировать скилл (глобально) |
| `POST` | `/skills/{slug}/deactivate` | Деактивировать скилл |
| `POST` | `/skills/upload` | Загрузить `.md` файл скилла (multipart/form-data). Сохраняется в `app/skills/uploads/` |

### Сессии и память

| Method | Path | Описание |
|--------|------|---------|
| `GET` | `/sessions` | Список сессий (последние 50) |
| `POST` | `/sessions` | Создать новую сессию. Body: `{ title }` |
| `GET` | `/sessions/{id}/messages` | История сообщений сессии |
| `PATCH` | `/sessions/{id}` | Переименовать сессию. Body: `{ title }` |
| `GET` | `/memories/search` | Полнотекстовый + векторный поиск по документации. Query: `q`, `session_id?`, `top_k?` |

### Документы

| Method | Path | Описание |
|--------|------|---------|
| `POST` | `/upload_pdf` | Загрузить PDF/MD/TXT в память (global или session scope) |
| `POST` | `/upload_signals` | Загрузить CSV таблицы сигналов |
| `DELETE` | `/clear_documents` | Очистить документы сессии или глобальные |
| `GET` | `/documents` | Список загруженных документов |

### Бенчмарк

| Method | Path | Описание |
|--------|------|---------|
| `POST` | `/benchmark/run` | Agents4PLC: прогон задач (easy/medium/hard) |
| `POST` | `/benchmark/st_coding/run` | ST coding benchmark: 10 задач IA01–IA10 |
| `GET` | `/benchmark/results` | Последние 50 результатов бенчмарков |

### Мониторинг

| Method | Path | Описание |
|--------|------|---------|
| `GET` | `/monitoring/tokens` | Статистика токенов по сессиям |
| `GET` | `/monitoring/cost` | Стоимость вызовов LLM в USD |
| `GET` | `/monitoring/cost/per_task` | Стоимость в разрезе benchmark-конфигов |
| `GET` | `/monitoring/dashboard` | Plotly-дашборд (HTML) |

### System

| Method | Path | Описание |
|--------|------|---------|
| `GET` | `/health` | Проверка статуса всех сервисов |

---

## Skills Guide

Skills — модульные расширения, которые динамически меняют поведение агента. Каждый скилл — директория с `SKILL.md` (YAML frontmatter + Markdown тело) и опциональным `tool.py` с Python-инструментами.

### Встроенные скиллы (`app/skills/builtin/`)

| Скилл | Назначение | Содержимое |
|-------|-----------|-----------|
| `st-style-guide` | Конвенции именования, типы данных, таймеры, формат ST/PLCopen XML | Извлечён из `ENGINEER_PROMPT` |
| `matiec-st-guide` | Саммари ключевых правил совместимости с MatIEC (no `BEGIN`, strict typing, safe naming, debugging hints) | Извлечён из `IEC-61131-3-ST-GUIDE.md` |

### Автоматический выбор (Skill Router)

Если в запросе `POST /chat` не указано поле `skills`, система автоматически выбирает релевантные скиллы:

1. Запрос пользователя эмбеддится через `embed_single()`
2. Вычисляется cosine similarity с эмбеддингами `description` всех доступных скиллов
3. Выбираются топ-3 скилла с similarity ≥ 0.45
4. Если ничего не прошло порог — используются все доступные скиллы (fallback)

**Ручной режим** (явно указанные скиллы отключает авто-роутинг):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "FB для насоса", "skills": ["st-style-guide"]}'
```

### Загрузка своего скилла

```bash
curl -F "file=@my-skill.md" http://localhost:8000/skills/upload
```

Формат `SKILL.md`:
```markdown
---
name: my-skill
description: Use when working with specific PLC controllers...
version: "1.0.0"
has_tools: false
---

# My Skill

Инструкции для агента...
```

Скилл с Python-инструментами (`tool.py`):
```python
from langchain_core.tools import tool

@tool
async def my_tool(param: str) -> str:
    """Tool description."""
    return f"Result: {param}"
```

### Механика

При старте сессии (или первом запросе) `Skill Router` эмбеддит запрос и выбирает релевантные скиллы. Их Markdown-тело инжектится в system prompt агента. Если скилл имеет `tool.py` — инструменты регистрируются и становятся доступны LLM для вызова. Система полностью backward-compatible: при `skills_enabled=False` или пустом списке скиллов поведение идентично версии без скиллов.

---

## Быстрый старт

```bash
cp .env.example .env
# Настройте LLM_BACKEND, модели и ключи
# Положите GGUF эмбеддинга в ./models/

docker compose up --build -d
```

| Сервис | URL |
|--------|-----|
| Gradio UI | http://localhost:7860 |
| API (Swagger) | http://localhost:8000/docs |
| MatIEC | http://localhost:8001/health |

### Локальная разработка

```bash
python -m venv .venv
pip install -r requirements.txt
docker compose up -d postgres matiec llama-embedding
uvicorn app.main:app --reload --port 8000
python -m ui.gradio_app
```

### LM Studio

```env
LLM_BACKEND=lmstudio
LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1
ENGINEER_MODEL=mistralai/ministral-3-3b
AGENT_CONTEXT_LIMIT=32768
AGENT_MAX_TOKENS=8192
```

---

## Конфигурация (`.env`)

| Переменная | По умолчанию | Описание |
|-----------|-------------|---------|
| `LLM_BACKEND` | `openrouter` | `openrouter` или `lmstudio` |
| `OPENROUTER_API_KEY` | — | Ключ OpenRouter |
| `ENGINEER_MODEL` | `qwen/qwen3.6-27b` | Модель для Expert и `generate_st_code` |
| `EMBEDDING_BACKEND` | `local` | `local` (llama.cpp) или `openrouter` |
| `EXPERT_MAX_ITERATIONS` | `5` | Лимит шагов ReAct-цикла |
| `SKILLS_ENABLED` | `true` | Включить систему скиллов |
| `TOP_K_RETRIEVAL` | `8` | Число parent-фрагментов в Retrieval |
| `RETRIEVAL_HYBRID_ENABLED` | `true` | Dense + keyword (tsvector) fusion |
| `BENCHMARK_MAX_VALIDATION_ATTEMPTS` | `2` | Лимит `validate_st_syntax` в ST coding bench |
| `OCR_BACKEND` | `openrouter` | `openrouter` или `paddle` |

---

## Структура проекта

```
app/
  agents/        — LangGraph: Retrieval + Expert, LLM-клиент
  api/           — REST-маршруты (8 модулей: agent, skills, sessions, ...)
  memory/        — pgvector RAG, гибридный поиск, сессии чатов
  monitoring/    — учёт токенов и стоимости
  prompts/       — системные промпты (BASE_EXPERT_PROMPT, ENGINEER_PROMPT, ...)
  skills/        — ядро системы скиллов
    builtin/     — встроенные скиллы (st-style-guide, matiec-st-guide)
    uploads/     — загруженные через API скиллы
  tools/         — MatIEC, PDF/OCR, эмбеддинги, чанкинг
ui/              — Gradio UI
benchmark/       — ST coding bench (10 задач IA01–IA10), Agents4PLC runner
matiec/          — MatIEC HTTP-сервис (Docker)
scripts/         — SQL миграции
models/          — GGUF-файлы эмбеддинга (не в git)
```

---

## Для кого

| Кто | Зачем |
|-----|-------|
| Инженеры-автоматизаторы | Быстрые прототипы кода для ПЛК с валидацией |
| AI/ML-инженеры | Агентные workflow, RAG, LangGraph, скиллы |
| Исследователи | Эксперименты с генерацией и верификацией PLC-кода |
| Студенты | Пример production-oriented агентной системы на Python |

---

## Лицензия

Apache 2.0