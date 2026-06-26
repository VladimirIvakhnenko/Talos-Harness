# Talos Harness

**Гибкий оркестратор агентов для индустриальной автоматизации**

Talos Harness — инфраструктурный каркас (harness), который соединяет агентов, модели, инструменты и память в единый workflow для автоматизации разработки ПО для ПЛК. Платформа подходит для генерации ST, валидации через MatIEC, RAG по документации и может быть адаптирована под другие языки и задачи.

## Концепция

Talos Harness построен вокруг двух принципов.

### 1. Harness — оркестратор, а не монолит

Вы не привязаны к конкретной модели, векторной БД или инструменту. Через `.env` можно заменить:

- LLM (OpenRouter, LM Studio, любой OpenAI-compatible API)
- эмбеддинги (llama.cpp GGUF, OpenRouter)
- валидатор (MatIEC; в перспективе — PLCverif и др.)
- стратегию чанкинга (сейчас recursive + parent-child в `app/tools/doc_indexer.py`)

### 2. Два агента — одна цель

- **Retrieval** (без LLM) — детерминированный сбор контекста: dense RAG, история чата, заметки о загрузке.
- **Expert** (ReAct) — генерация ST/PLCopen XML, валидация MatIEC, итеративное исправление.

Разделение даёт предсказуемый поиск, меньше вызовов LLM на этапе retrieval и независимую отладку компонентов.

## Архитектура

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Retrieval (без LLM)                │
│  dense RAG + история чата + upload  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Expert (ReAct)                     │
│  generate_st_code → validate_st     │
│  итерации до EXPERT_MAX_ITERATIONS  │
└─────────────────────────────────────┘
    │
    ▼
  ST / PLCopen XML + MatIEC
```

| Компонент | Назначение |
|-----------|------------|
| **Retrieval** | Поиск по pgvector (`scope=global` + `scope=session`), последние сообщения чата, контекст загрузки |
| **Expert** | ReAct-цикл: генерация ST/XML и валидация через MatIEC |
| **MatIEC** | `iec2c` / `iec2iec` — синтаксис и компиляция |
| **Эмбеддинги** | llama.cpp (GGUF) или OpenRouter API |
| **LLM** | OpenRouter или LM Studio (OpenAI-compatible API) |
| **Память** | PostgreSQL 17 + pgvector, персистентные сессии чатов |

Документы из Gradio индексируются с `scope=session`; через API — `scope=global` (опционально `session_id`).

Реализация графа: [`app/agents/react_agent.py`](app/agents/react_agent.py).

## Возможности и точки расширения

| Компонент | Что делает сейчас | Можно заменить / добавить |
|-----------|-------------------|-------------------------|
| **Retrieval** | Dense RAG + история чата | Гибридный поиск (BM25 + dense), reranking |
| **Expert** | ST/XML + MatIEC | Другой целевой язык (LD, FBD), другой валидатор |
| **Память** | pgvector, 768-dim | Qdrant, Milvus и др. (потребует адаптер в `app/memory/`) |
| **Эмбеддинги** | llama.cpp (GGUF) | OpenRouter, локальные sentence-transformers |
| **LLM** | OpenRouter / LM Studio | Любая OpenAI-compatible модель |

## Ключевые сценарии

1. **Генерация ST по документации** — загрузка PDF/MD → индексация → запрос на естественном языке → код с валидацией.
2. **PLCopen XML** — экспорт схем для обмена с CODESYS и совместимыми средами.
3. **Валидация и исправление** — MatIEC находит ошибки, Expert перегенерирует код (до лимита итераций).
4. **Модуль по таблице сигналов** — CSV/XLSX → VAR-секция и каркас PROGRAM.
5. **Бенчмарк Agents4PLC** (опционально) — прогон задач через `benchmark/runner.py`.

## Быстрый старт

```bash
cp .env.example .env
# Настройте LLM_BACKEND, модели и ключи (см. Конфигурация)
# Положите GGUF эмбеддинга в ./models/ (EMBEDDING_GGUF_FILE в .env)

docker compose up --build -d
```

После старта:

| Сервис | URL |
|--------|-----|
| Gradio UI | http://localhost:7860 |
| API (Swagger) | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| MatIEC | http://localhost:8001/health |
| Мониторинг | http://localhost:8000/monitoring/dashboard |

### LM Studio (локальный LLM)

1. Загрузите модель в LM Studio, включите Local Server (порт 1234).
2. В `.env`:

```env
LLM_BACKEND=lmstudio
LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1
ENGINEER_MODEL=mistralai/ministral-3-3b
AGENT_CONTEXT_LIMIT=32768
AGENT_MAX_TOKENS=8192
```

3. `docker compose up -d api ui`

## Конфигурация (`.env`)

| Переменная | Описание |
|------------|----------|
| `LLM_BACKEND` | `openrouter` \| `lmstudio` |
| `OPENROUTER_API_KEY` | Ключ OpenRouter (LLM и OCR) |
| `LMSTUDIO_BASE_URL` | URL LM Studio (`/v1`) |
| `ENGINEER_MODEL` | Модель Expert и `generate_st_code` |
| `EMBEDDING_BACKEND` | `local` (llama.cpp) \| `openrouter` |
| `LLAMA_EMBEDDING_URL` | Docker: `http://llama-embedding:8080` |
| `EXPERT_MAX_ITERATIONS` | Лимит шагов ReAct Expert (default 5) |
| `TOP_K_RETRIEVAL` | Число фрагментов в Retrieval |

Полный список — в [`.env.example`](.env.example).

### Миграция существующей БД

```bash
docker exec -i plc_postgres psql -U plc -d plc_agent < scripts/migrate_memory_sessions.sql
```

## Использование

### Чат-агент (Gradio)

Список чатов в sidebar, загрузка PDF/MD/TXT в память текущей сессии, потоковый лог Retrieval + Expert.

### Чат через API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Напиши FB для управления насосом с защитой от сухого хода", "session_id": "uuid-опционально"}'
```

### Сессии чатов

```bash
curl http://localhost:8000/sessions
curl -X POST http://localhost:8000/sessions -H "Content-Type: application/json" -d '{"title":"Новый чат"}'
curl http://localhost:8000/sessions/{id}/messages
```

### Загрузка документации (global)

```bash
curl -F "file=@manual.pdf" "http://localhost:8000/upload_pdf?doc_type=elbrus_manual"
curl "http://localhost:8000/memories/search?q=TON&session_id=&top_k=5"
```

### Генерация модуля по сигналам

```bash
curl -F "file=@signals.csv" "http://localhost:8000/upload_signals?controller=elbrus"
curl -X POST http://localhost:8000/generate_module \
  -H "Content-Type: application/json" \
  -d '{"controller":"elbrus","signals_path":"/app/uploads/signals_XXX.csv"}'
```

### Бенчмарк Agents4PLC (опционально)

```bash
git clone https://github.com/Luoji-zju/Agents4PLC_release benchmark/Agents4PLC_release

curl -X POST http://localhost:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"subset":"medium","n_tasks":10,"configs":["baseline","full_agent"]}'
```

## MatIEC

Сборка образа (форк [sm1820/matiec](https://github.com/sm1820/matiec)):

```bash
docker compose build matiec
docker compose up -d matiec
curl http://localhost:8001/health
```

Проверка валидации:

```bash
curl -X POST http://localhost:8001/validate \
  -H "Content-Type: application/json" \
  -d '{"code":"PROGRAM Test\nVAR x: BOOL; END_VAR\nx := TRUE;\nEND_PROGRAM","task_id":"test"}'
```

Ожидается `ok: true`. При ошибке `lib/ieclib.txt` пересоберите `matiec` — для `iec2iec` в `matiec_server.py` передаётся `-I /usr/local/lib/matiec/`.

## Локальная разработка

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d postgres matiec llama-embedding

uvicorn app.main:app --reload --port 8000
python -m ui.gradio_app
```

После изменения `.env` перезапустите процессы — `get_settings()` кэшируется при старте.

## Структура проекта

```
app/
  agents/      — LangGraph Retrieval + Expert, LLM-клиент
  api/         — REST-маршруты (agent, documents, sessions, memory, benchmark)
  memory/      — pgvector RAG, сессии чатов
  monitoring/  — учёт токенов
  prompts/     — системные промпты
  tools/       — MatIEC, PDF/OCR, индексация документов
ui/            — Gradio (список чатов + агент)
benchmark/     — Agents4PLC runner
scripts/       — init_db.sql, migrate_memory_sessions.sql
matiec/        — MatIEC HTTP-сервис
models/        — GGUF-файлы эмбеддинга (не в git)
```

## Для кого это

| Кто | Зачем |
|-----|-------|
| Инженеры-автоматизаторы | Быстрые шаблоны и прототипы кода для ПЛК |
| AI/ML-инженеры | Агентные workflow, RAG, оркестрация LangGraph |
| Исследователи | Эксперименты с генерацией и верификацией PLC-кода |
| Студенты | Пример production-oriented агентной системы на Python |

## Дорожная карта

- Поддержка LD и FBD, углублённая интеграция с CODESYS
- Гибридный поиск (dense + sparse), Cross-Encoder reranking
- Адаптеры памяти под внешние vector DB
- Расширяемый набор инструментов Expert (PLCverif, симуляция)

## Почему Harness

**Harness** — упряжь и каркас: система, которая связывает компоненты в единое целое и направляет их на задачу. Talos Harness оркестрирует агентов, модели и инструменты; домен PLC/ST — текущая реализация, а не жёсткое ограничение платформы.

## Ключевые отличия

| Решение | Ограничение | Роль Talos Harness |
|---------|-------------|-------------------|
| Универсальные copilot / LLM | Слабая специализация на IEC 61131-3 | ST/XML, MatIEC, RAG по мануалам |
| Классические IDE ПЛК | Шаблоны без семантического поиска | Контекст из документации и истории чата |
| Только валидаторы (PLCverif и др.) | Проверка без генерации | Генерация + валидация в одном workflow |

## Документация

- [Техническое задание](docs/project.md)
- [Справочник агентов](docs/AGENTS.md) — может отставать от кода
- [API Reference](http://localhost:8000/docs) (после запуска)

## Лицензия

Планируется Apache 2.0 — свободное использование и модификация с указанием авторства.

## Контрибьютинг

Приветствуются PR: новые tools для Expert, улучшения Retrieval, интеграции с IDE, поддержка других языков ПЛК. Обсуждение — через Issues репозитория.
