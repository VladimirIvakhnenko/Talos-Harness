# Talos Harness

Многоагентная система генерации и верификации ST-кода для промышленных ПЛК (IEC 61131-3) с поддержкой контроллеров Эльбрус, Байкал и CODESYS.

## Архитектура

- **Planner** (Nex-N2-Pro) — декомпозиция задач, ReAct-цикл
- **Engineer** (Qwen3.5-9B) — генерация ST-кода
- **Retriever** (Qwen3.5-4B) — RAG по документации
- **MatIEC** — синтаксическая верификация и компиляция
- **PostgreSQL + pgvector** — гибридный поиск (dense + sparse + RRF)

## Быстрый старт

```bash
cp .env.example .env
# Укажите OPENROUTER_API_KEY в .env

docker compose up --build -d
```

## Сервисы

| Сервис | URL | Описание |
|--------|-----|----------|
| API (Swagger) | http://localhost:8000/docs | REST API |
| Gradio UI | http://localhost:7860 | Веб-интерфейс |
| MatIEC | http://localhost:8001/health | Компилятор ST |
| Dashboard | http://localhost:8000/monitoring/dashboard | Мониторинг токенов |

## Сценарии использования

### 1. Чат-агент (Gradio или API)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Напиши FB для управления насосом с защитой от сухого хода"}'
```

### 2. Загрузка документации

```bash
curl -F "file=@manual.pdf" "http://localhost:8000/upload_pdf?doc_type=elbrus_manual"
```

### 3. Генерация модуля по сигналам

```bash
curl -F "file=@signals.csv" "http://localhost:8000/upload_signals?controller=elbrus"
curl -X POST http://localhost:8000/generate_module \
  -H "Content-Type: application/json" \
  -d '{"controller":"elbrus","signals_path":"/app/uploads/signals_XXX.csv"}'
```

### 4. Бенчмарк Agents4PLC

Клонируйте датасет:

```bash
git clone https://github.com/Luoji-zju/Agents4PLC_release benchmark/Agents4PLC_release
```

Запуск:

```bash
curl -X POST http://localhost:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"subset":"medium","n_tasks":10,"configs":["baseline","full_agent"]}'
```

## Проверка MatIEC

Сборка и запуск компилятора ST (форк [sm1820/matiec](https://github.com/sm1820/matiec)):

```bash
docker compose build --no-cache matiec
docker compose up -d matiec
docker compose logs matiec

curl http://localhost:8001/health
# Ожидается: {"status":"ok","iec2c":true,"iec2iec":true,"lib_ok":true}

curl -X POST http://localhost:8001/validate \
  -H "Content-Type: application/json" \
  -d '{"code":"PROGRAM Test\nVAR x: BOOL; END_VAR\nx := TRUE;\nEND_PROGRAM","task_id":"test"}'
```

После того как `matiec` healthy, поднимите остальные сервисы:

```bash
docker compose up -d
curl http://localhost:8000/health
```

## Локальная разработка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# PostgreSQL и MatIEC через Docker
docker compose up -d postgres matiec

# API
uvicorn app.main:app --reload --port 8000

# Gradio UI
python -m ui.gradio_app
```

## Структура проекта

```
app/           — FastAPI приложение и бизнес-логика
  agents/      — LangGraph ReAct-агент, LLM-клиент
  api/         — HTTP-маршруты и схемы
  memory/      — pgvector RAG
  monitoring/  — учёт токенов
  prompts/     — системные промпты
  tools/       — MatIEC, PDF, парсер сигналов
ui/            — Gradio веб-интерфейс
benchmark/     — Agents4PLC runner
scripts/       — init_db.sql
matiec/        — MatIEC HTTP-сервис
```

## Документация

- [Техническое задание](docs/project.md)
- [Справочник агентов](docs/AGENTS.md)
