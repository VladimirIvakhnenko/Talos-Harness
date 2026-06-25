# AGENTS.md — Talos Harness Agent Reference

> Справочный документ по агентам системы.  
> Версия 2.0 · Talos Harness · Junior ML Contest 2026

---

## Обзор архитектуры

```
User Request
     │
     ▼
┌────────────────────────────────────────────────────────┐
│               FastAPI  :8000                           │
│         POST /chat  ·  POST /generate_module           │
└─────────────────────┬──────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│          LangGraph StateGraph (ReAct)                  │
│                                                        │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
│  │  PLANNER    │──▶│  ToolNode   │──▶│  PLANNER    │  │
│  │ Nex-N2-Pro  │   │  (4 tools)  │   │  (loop)     │  │
│  └──────┬──────┘   └─────────────┘   └──────┬──────┘  │
│         │                                    │         │
│         └──────────── END ──────────────────▶│         │
│                                    ┌──────────▼──────┐ │
│                                    │  post_process   │ │
│                                    │  (MatIEC call)  │ │
│                                    └─────────────────┘ │
└────────────────────────────────────────────────────────┘
         │                    │                │
         ▼                    ▼                ▼
   PostgreSQL           OpenRouter          MatIEC
   pgvector             API :443           :8001
   (memory)          (Nex/Qwen/emb)      (iec2c)
```

---

## Агент 1: Планировщик (Planner)

### Назначение
Анализирует запрос пользователя, декомпозирует его на подзадачи, маршрутизирует вызовы инструментов и контролирует финальный ReAct-цикл.

### Модель
| Параметр | Значение |
|----------|----------|
| Model ID | `nex-agi/nex-n2-pro` |
| Провайдер | OpenRouter |
| Параметры | 397B total / 17B active (MoE) |
| VRAM | ~10 GB (Q4) |
| Цена | **Бесплатно** на OpenRouter |
| Ключевые бенчмарки | Terminal-Bench 2.1: **75.3** · BrowseComp: **83.7** · Toolathlon: **51.9** |

### Системный промпт (сокращённо)
```
You are a PLC engineering task planner.
Decompose the user request into sub-tasks and route them.

ReAct format:
Thought: <reasoning>
Action: <tool_name>(<args>)
Observation: <result>
Final Answer: <answer>

Available tools: search_memory, generate_st_code, validate_st_syntax, remember_fact.
Always validate generated code before returning it.
```

### Инструменты
| Инструмент | Описание |
|------------|----------|
| `search_memory(query, session_id, top_k)` | Hybrid dense+sparse поиск по pgvector |
| `generate_st_code(spec, controller)` | Делегирует Engineer-модели |
| `validate_st_syntax(code, task_id)` | Вызывает MatIEC через HTTP |
| `remember_fact(fact, session_id)` | Сохраняет факт в pgvector (type=fact) |

### Поведение
- Запускается первым при каждом запросе (`/chat`)
- Реализует ReAct-цикл: `Thought → Action → Observation` (до `agent_max_iterations=10`)
- При наличии `tool_calls` в ответе → переходит в `ToolNode`
- При отсутствии → переходит в `post_process` (MatIEC финальная проверка)

---

## Агент 2: Инженер (Engineer)

### Назначение
Генерирует ST-код по спецификации. Вызывается через инструмент `generate_st_code` из Планировщика. Получает контекст из RAG (документация, стандарты) через `search_memory`.

### Модель
| Параметр | Значение |
|----------|----------|
| Model ID | `qwen/qwen3.5-9b` |
| Провайдер | OpenRouter |
| Параметры | 9B |
| VRAM | ~6 GB (Q4) |
| Контекст | **262K токенов** |
| Ключевые бенчмарки | GPQA Diamond: **80.6%** · τ²-Bench: **86.8%** |

### Системный промпт (Engineer Rules)
```
MANDATORY RULES:
1. PROGRAM / FUNCTION_BLOCK / FUNCTION with explicit VAR sections
2. Types: BOOL, INT, DINT, REAL, TIME, STRING only
3. SCREAMING_SNAKE_CASE globals, camelCase locals, PascalCase FBs
4. Every variable: inline (* comment *)
5. Header: (* MODULE / CONTROLLER / DATE / VERSION *)
6. CASE preferred over nested IF
7. Always ELSE. No GOTO.
8. Timers: TON, TOF, TP — named instances
9. REAL division: check denominator != 0
10. End: (* VERIFY: test vectors *)

ELBRUS ADDRESSING:
  DI → %IX0.0+   BOOL
  DO → %QX0.0+   BOOL
  AI → %IW0,%IW2,... INT; scale: REAL := INT_RAW * (range/32767.0)
  AO → %QW0,%QW2,... INT; scale: INT := ROUND(REAL * 32767.0/range)
```

### Специфика для контроллеров
| Контроллер | Особенности |
|------------|-------------|
| **Эльбрус (Elbrus-2C3)** | AI-адреса через 2 (%IW0, %IW2...), max scan 10ms |
| **Байкал** | Аналогично Эльбрус, CODESYS-совместимый |
| **CODESYS** | Стандартные адреса, полный IEC 61131-3 |

---

## Агент 3: Ретривер (Retriever)

### Назначение
Семантический поиск по документации (PDF), стандартам IEC 61131-3, истории чатов и сохранённым скиллам. Реализован как набор функций в `app/memory/store.py`, вызывается через инструмент `search_memory`.

### Модель (для суммаризации длинных диалогов)
| Параметр | Значение |
|----------|----------|
| Model ID | `qwen/qwen3.5-4b` |
| Провайдер | OpenRouter |
| VRAM | ~3 GB (Q4) |
| Роль | Суммаризация старых сообщений (скользящее окно) |

### Пайплайн retrieval
```
Query text
   │
   ├─► Dense search (pgvector HNSW, cosine, top-20 child chunks)
   │
   ├─► Sparse search (tsvector GIN, BM25-like, top-20 child chunks)
   │
   ├─► RRF fusion: Score(doc) = Σ 1/(k + rank_i),  k=60
   │                                        ↓ top-20
   ├─► Lift to parent chunks (дедупликация по parent_id)
   │
   └─► Return top-5 parents → LLM context
```

### Типы памяти
| Тип | Описание | metadata.type |
|-----|----------|---------------|
| Документация | Чанки PDF (IEC 61131-3, Эльбрус) | `doc` |
| История чата | User+Assistant обмены | `chat` |
| Факты | Ключевые решения сессии | `fact` |
| Скиллы | Успешные генерации для переиспользования | `skill` |

### Трёхуровневая история чата
| Уровень | Источник | Лимит |
|---------|----------|-------|
| L1 Краткосрочная | Последние N сообщений (in-memory) | 10 сообщений |
| L2 Семантическая | Dense поиск по всей истории сессии | top-3 |
| L3 Факты | Dense поиск по type=fact | top-2 |

---

## Инструменты (Tools)

### `search_memory`
```python
async def search_memory(query: str, session_id: str = "", top_k: int = 5) -> str
```
- Вызывает `hybrid_search()` из `app/memory/store.py`
- Возвращает топ-5 parent-чанков с источником и содержимым
- Используется: Планировщик на каждом шаге Thought

### `generate_st_code`
```python
async def generate_st_code(spec: str, controller: str = "elbrus") -> str
```
- Вызывает Engineer LLM (`qwen/qwen3.5-9b`) через OpenRouter
- Прикладывает `ENGINEER_PROMPT` как SystemMessage
- Возвращает сырой ST-код

### `validate_st_syntax`
```python
async def validate_st_syntax(code: str, task_id: str = "agent_val") -> str
```
- HTTP POST `matiec:8001/validate`
- Возвращает `✅ Syntax valid` или `❌ Syntax errors: ...`
- Планировщик вызывает после каждой генерации

### `remember_fact`
```python
async def remember_fact(fact: str, session_id: str = "") -> str
```
- Векторизует факт через `text-embedding-3-small`
- Сохраняет в pgvector с `metadata.type = "fact"`
- Факты извлекаются в следующих turn-ах через L3

---

## LangGraph граф

```python
StateGraph(AgentState)
  .add_node("planner",      planner_node)   # LLM + bind_tools
  .add_node("tools",        ToolNode)        # execute tool calls
  .add_node("post_process", post_process)    # MatIEC final compile

  .set_entry_point("planner")
  .add_conditional_edges("planner", should_continue,
      {"tools": "tools", END: "post_process"})
  .add_edge("tools", "planner")              # ReAct loop
  .add_edge("post_process", END)
```

### AgentState
```python
class AgentState(TypedDict):
    messages:   list          # история LangChain сообщений
    session_id: str           # UUID сессии
    task_id:    str | None    # ID задачи Agents4PLC
    final_code: str | None    # извлечённый ST-код
    matiec_ok:  bool | None   # результат MatIEC compile
    tokens:     dict          # суммарные токены
```

---

## Мониторинг токенов

Каждый LLM-вызов записывается через `TokenUsageCallback`:

```
agent_name  tool_name          model_id              tokens   cost_usd  latency_ms
planner     search_memory      nex-agi/nex-n2-pro    1240     0.000000  823
planner     generate_st_code   nex-agi/nex-n2-pro    342      0.000000  412
engineer    (direct)           qwen/qwen3.5-9b        2100     0.000420  1240
planner     validate_st_syntax nex-agi/nex-n2-pro    180      0.000000  95
```

Доступно через:
- `GET /monitoring/tokens` — по сессиям
- `GET /monitoring/cost` — USD разбивка
- `GET /monitoring/dashboard` — Plotly HTML

---

## MatIEC интеграция

Агент вызывает MatIEC через HTTP (не subprocess):

```
Planner → validate_st_syntax tool
              │
              ▼
    POST matiec:8001/validate
    { "code": "...", "task_id": "M1" }
              │
              ▼
    MatIEC container:
      iec2iec program.st  (syntax check)
      → { ok, errors, warnings }
              │
              ▼
    "✅ Syntax valid"  →  Planner returns Final Answer
    "❌ errors: ..."   →  Planner iterates (fixes code)
```

Финальная компиляция запускается в `post_process` через `compile_st()` (`iec2c`).

---

## Конфигурация моделей (.env)

```env
# Все модели через OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...
PLANNER_MODEL=nex-agi/nex-n2-pro    # бесплатная, ReAct-специализация
ENGINEER_MODEL=qwen/qwen3.5-9b      # 262K контекст, coding
RETRIEVER_MODEL=qwen/qwen3.5-4b     # лёгкая, суммаризация
EMBEDDING_MODEL=text-embedding-3-small
```

### Альтернативные модели
| Роль | Альтернатива | Примечание |
|------|-------------|------------|
| Engineer | `poolside/laguna-m1` | Бесплатная, 256K контекст |
| Engineer | `iquestlab/iquest-coder-v1-14b` | SWE-Bench: 76.2% |
| Planner | `qwen/qwen3.5-9b` | Если Nex-N2-Pro недоступен |

---

## Benchmarks (Agents4PLC)

| Конфигурация | Compilation Rate | Pass@1 |
|-------------|-----------------|--------|
| Baseline (LLM only) | ~40–50% | ~15–25% |
| Full Agent (RAG+ReAct+MatIEC) | ~75–85% | ~45–55% |
| AutoPLC (sota) | >90% | ~78% FV |

Запуск: `POST /benchmark/run { "subset": "medium", "n_tasks": 10 }`

---

## Быстрый старт

```bash
# 1. Настройка
cp .env.example .env
# Добавить OPENROUTER_API_KEY

# 2. Запуск
docker compose up -d

# 3. Swagger UI
open http://localhost:8000/docs

# 4. Тест агента
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Напиши FB для управления насосом с защитой от сухого хода"}'

# 5. Загрузить документацию Эльбрус
curl -F "file=@elbrus_manual.pdf" \
  "http://localhost:8000/upload_pdf?doc_type=elbrus_manual"

# 6. Сгенерировать модуль по сигналам
curl -F "file=@signals.csv" \
  "http://localhost:8000/upload_signals?controller=elbrus"

curl -X POST http://localhost:8000/generate_module \
  -H "Content-Type: application/json" \
  -d '{"controller":"elbrus","signals_path":"/app/uploads/signals_XXX.csv"}'

# 7. Запустить бенчмарк
curl -X POST http://localhost:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"subset":"medium","n_tasks":10,"configs":["baseline","full_agent"]}'
```

---

## Ссылки

| Ресурс | URL |
|--------|-----|
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Token Dashboard | http://localhost:8000/monitoring/dashboard |
| Health | http://localhost:8000/health |
| Agents4PLC | https://arxiv.org/abs/2410.14209 |
| MatIEC | https://github.com/nucleron/matiec |
| pgvector | https://github.com/pgvector/pgvector |
| LangGraph | https://langchain-ai.github.io/langgraph/ |
| OpenRouter | https://openrouter.ai |