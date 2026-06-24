**ТЕХНИЧЕСКОЕ ЗАДАНИЕ**

на разработку

**Многоагентной системы генерации и верификации кода ПЛК**

_Multi-Agent PLC Code Generator with Formal Verification_

_via Agents4PLC Benchmark_

| **Версия**  | 1.0 (финальная)                              |
| ----------- | -------------------------------------------- |
| **Дата**    | Июнь 2026                                    |
| **Конкурс** | Junior ML Contest 2026 - AI Talent Hub, ИТМО |

# **1\. Назначение системы и цели проекта**

Система предназначена для автоматизации разработки программ для промышленных контроллеров (ПЛК) на языке Structured Text (ST) в соответствии со стандартом IEC 61131-3, с акцентом на поддержку отечественных контроллеров серии Эльбрус и Байкал.

## **1.1. Ключевые сценарии использования**

- Генерация шаблона модуля для контроллера Эльбрус на основе документации и таблицы сигналов
- Написание скриптов на ST при наличии стандарта IEC 61131-3 и системных промптов
- Проверка качества генерации на бенчмарке Agents4PLC v2.0 (96 задач)
- Мониторинг затраченных токенов и стоимости вызовов

## **1.2. Целевые пользователи**

- Инженеры-автоматизаторы, разрабатывающие программы для отечественных ПЛК
- Разработчики АСУТП, переходящие с западных платформ (Siemens TIA Portal, Schneider EcoStruxure) на отечественные
- Исследователи и участники конкурсов AI/ML в области промышленной автоматизации

# **2\. Архитектура системы**

## **2.1. Многоагентный ReAct-цикл**

Система построена на базе LangGraph как граф состояний с тремя специализированными агентами, реализующими цикл Reasoning → Acting → Observation.

| **Агент**                 | **Роль**                                                            | **Инструменты**                                                | **Модель** |
| ------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------- | ---------- |
| **Планировщик (Planner)** | Анализ запроса, декомпозиция задач, маршрутизация между агентами    | decompose_task create_plan search_memory                       | Nex-N2-Pro |
| **Инженер (Engineer)**    | Генерация ST-кода, настройка конфигураций ПЛК, валидация синтаксиса | generate_st_code validate_syntax retrieve_example configure_io | Qwen3.5-9B |
| **Ретривер (Retrieval)**  | Поиск по документации, стандартам и истории диалогов                | vector_search hybrid_search                                    | Qwen3.5-4B |

## **2.2. Двухуровневая память (PostgreSQL + pgvector)**

- Документальная память - чанки PDF-документов (стандарты, ТЗ, документация Эльбрус)
- История чатов - сохранение диалогов для контекстной непрерывности между сессиями

**Структура таблицы memories:**

CREATE TABLE memories (

id SERIAL PRIMARY KEY,

content TEXT NOT NULL,

embedding vector(1536) NOT NULL,

metadata JSONB NOT NULL, -- { type, source, session_id, tokens_used }

created_at TIMESTAMP DEFAULT NOW()

);

CREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops);

## **2.3. Технический стек**

| **Компонент**          | **Технология**                   | **Назначение**                                  |
| ---------------------- | -------------------------------- | ----------------------------------------------- |
| **Агентный фреймворк** | LangGraph + LangChain            | Построение графа состояний и ReAct-цикла        |
| **Веб-сервер**         | FastAPI + Uvicorn                | REST API + WebSocket для стриминга              |
| **Векторная БД**       | PostgreSQL 17 + pgvector         | Хранение эмбеддингов и семантический поиск      |
| **LLM (облако)**       | OpenRouter API                   | Доступ к Nex-N2-Pro, Qwen3.5, Laguna M.1        |
| **LLM (локально)**     | Ollama                           | Локальные модели для air-gap окружений          |
| **Обработка PDF**      | pdf2image + Pillow               | Конвертация PDF документации в изображения      |
| **Чанкинг**            | langchain-text-splitters         | Рекурсивное разбиение (chunk=1000, overlap=200) |
| **Эмбеддинги**         | text-embedding-3-small / BGE-M3  | Векторизация текстов документации               |
| **Мониторинг токенов** | LangChain callbacks + PostgreSQL | Учёт затрат на каждый вызов LLM                 |
| **Верификация**        | PLCverif + nuXmv                 | Формальная проверка ST-кода                     |
| **Контейнеризация**    | Docker + Docker Compose          | Единое окружение для всех компонентов           |

# **3\. Мониторинг токенов и стоимости**

Все вызовы LLM логируются через LangChain callback-механизм. Данные сохраняются в PostgreSQL для анализа затрат и оптимизации промптов.

## **3.1. Схема таблицы token_usage**

CREATE TABLE token_usage (

id SERIAL PRIMARY KEY,

session_id UUID NOT NULL,

agent_name VARCHAR(50), -- planner | engineer | retriever

tool_name VARCHAR(100), -- generate_st_code | search_memory | ...

model_id VARCHAR(100), -- nex-agi/nex-n2-pro | qwen/qwen3.5-9b

prompt_tokens INTEGER,

completion_tokens INTEGER,

total_tokens INTEGER,

cost_usd NUMERIC(10,6), -- рассчитывается по прайсу OpenRouter

latency_ms INTEGER,

task_id VARCHAR(50), -- ID задачи из Agents4PLC или пользовательской

created_at TIMESTAMP DEFAULT NOW()

);

## **3.2. Реализация LangChain callback**

class TokenUsageCallback(BaseCallbackHandler):

def on_llm_end(self, response, \*\*kwargs):

usage = response.llm_output.get('token_usage', {})

db.execute(SQL_INSERT, \[session_id, agent_name, model_id, ...\])

\# SQL_INSERT = 'INSERT INTO token_usage

\# (session_id, agent_name, model_id, prompt_tokens,

\# completion_tokens, total_tokens, cost_usd, latency_ms)

\# VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'

## **3.3. API-эндпоинты мониторинга**

| **Эндпоинт**                    | **Метод** | **Описание**                                                   |
| ------------------------------- | --------- | -------------------------------------------------------------- |
| /monitoring/tokens              | GET       | Суммарные затраты токенов по сессиям / агентам / моделям       |
| /monitoring/tokens/{session_id} | GET       | Детализация по конкретной сессии: разбивка по агентам          |
| /monitoring/cost                | GET       | Затраты в USD за период: ?from=2026-06-01&to=2026-06-30        |
| /monitoring/cost/per_task       | GET       | Средняя стоимость задачи на Agents4PLC по сложности            |
| /monitoring/dashboard           | GET       | HTML-дашборд с графиками (Plotly) - токены / время / стоимость |

## **3.4. Отображаемые метрики дашборда**

- Суммарные токены за сессию с разбивкой по агентам (Planner / Engineer / Retriever)
- Стоимость (\$) за задачу в сравнении с baseline (bare LLM без агентов)
- Latency (мс) на каждом шаге ReAct-цикла
- Топ-5 самых дорогих инструментов
- Токены на шаг: Thought / Action / Observation

# **4\. Верификация на бенчмарке Agents4PLC**

Agents4PLC - первый верифицируемый бенчмарк для генерации ST-кода из требований на естественном языке. Версия v2.0 содержит 96 задач с формальными спецификациями и эталонным кодом, проверенным инженерами-ПЛК.

## **4.1. Метрики оценки**

| **Метрика**                       | **Описание**                                            | **Способ вычисления**                                 |
| --------------------------------- | ------------------------------------------------------- | ----------------------------------------------------- |
| **Compilation Rate**              | Доля программ, прошедших синтаксическую компиляцию      | Запуск PLCverif / ST-парсера на каждом выходе         |
| **Formal Verification Pass Rate** | Доля программ, прошедших формальную верификацию         | nuXmv / PLCverif со спецификациями из бенчмарка       |
| **Execution Correctness**         | Доля программ с правильными выходными сигналами         | Симуляция на тестовых векторах из Agents4PLC          |
| **Pass@1**                        | P(хотя бы одна из 1 попыток проходит все проверки)      | Стандартная метрика code generation                   |
| **Recall@5 (RAG)**                | Доля релевантных документов в топ-5 результатов поиска  | Ручная разметка релевантности на 20 тестовых запросах |
| **MRR (RAG)**                     | Средняя обратная позиция первого релевантного документа | Mean Reciprocal Rank по аннотированному набору        |

## **4.2. Экспериментальный дизайн**

Три конфигурации для объективного сравнения:

| **Конфигурация**                    | **RAG** | **Chain-of-Thought** | **Self-Reflection** | **Верификация** |
| ----------------------------------- | ------- | -------------------- | ------------------- | --------------- |
| **Baseline (LLM only)**             | ❌      | ❌                   | ❌                  | ❌              |
| **RAG-only**                        | ✅      | ❌                   | ❌                  | ❌              |
| **Full Agent (RAG+CoT+Reflection)** | ✅      | ✅                   | ✅                  | ✅ PLCverif     |

## **4.3. Ожидаемые результаты**

| **Метрика**                       | **Baseline (LLM only)** | **Full Agent (RAG+CoT+Reflection)** |
| --------------------------------- | ----------------------- | ----------------------------------- |
| **Compilation Rate**              | ~40-50%                 | ~75-85%                             |
| **Formal Verification Pass Rate** | ~10-20%                 | ~50-60%                             |
| **Pass@1**                        | ~15-25%                 | ~45-55%                             |

_Цифры ориентировочные; финальные значения заполняются после запуска на подмножестве (23 medium + 10 hard задач). Для сравнения: Agents4PLC достигает 68.8%, AutoPLC - 78% Formal Verification Pass Rate на тех же 23 задачах._

## **4.4. Пайплайн оценки**

- Клонировать репозиторий: git clone <https://github.com/Luoji-zju/Agents4PLC_release.git>
- Прочитать задачу из benchmark_v2/medium.jsonl (поля: task_id, description, formal_spec, reference_code)
- Запустить Full Agent → получить сгенерированный ST-код
- Скормить код PLCverif → Compilation Rate
- Запустить nuXmv с formal_spec → Formal Verification Pass Rate
- При успехе верификации - симуляция на тестовых векторах → Execution Correctness
- Записать метрики + токены в PostgreSQL

# **5\. Генерация шаблона модуля для контроллера Эльбрус**

Сценарий: пользователь загружает документацию на контроллер Эльбрус (PDF) и таблицу сигналов (CSV/XLSX), система генерирует готовый ST-модуль с корректными объявлениями переменных, привязками к физическим адресам и базовой логикой.

## **5.1. Входные данные**

| **Артефакт**             | **Формат**             | **Содержимое**                                                                   |
| ------------------------ | ---------------------- | -------------------------------------------------------------------------------- |
| **Документация Эльбрус** | PDF (мультистраничный) | Описание модулей ввода/вывода, адресное пространство, поддерживаемые типы данных |
| **Таблица сигналов**     | CSV / XLSX             | Имя сигнала, тип (DI/DO/AI/AO), адрес, диапазон, описание                        |
| **Системный промпт**     | JSON / текст           | Соглашения именования, стиль кода, обязательные комментарии                      |

## **5.2. Пример таблицы сигналов (входной формат CSV)**

SignalName,Type,Address,Range,Engineering_Unit,Description

PUMP_START,DI,%IX0.0,,Boolean,Команда запуска насоса

PUMP_RUN,DO,%QX0.0,,Boolean,Насос в работе

TANK_LEVEL,AI,%IW0,,0-100,Уровень резервуара (%)

VALVE_POS,AO,%QW0,,0-100,Положение клапана (%)

TEMP_SENSOR,AI,%IW2,,-50..150,Температура (°C)

## **5.3. Шаблон генерируемого ST-модуля**

(\* ================================================================ \*)

(\* MODULE: &lt;ModuleName&gt; \*)

(\* CONTROLLER: Эльбрус (Elbrus-2C3) \*)

(\* GENERATED BY: Multi-Agent PLC Code Generator v1.0 \*)

(\* DATE: &lt;GenerationDate&gt; \*)

(\* IEC 61131-3 Structured Text \*)

(\* ================================================================ \*)

PROGRAM &lt;ModuleName&gt;

VAR

(\* --- Дискретные входы (DI) --- \*)

PUMP_START : BOOL; (\* %IX0.0 - Команда запуска насоса \*)

(\* --- Дискретные выходы (DO) --- \*)

PUMP_RUN : BOOL; (\* %QX0.0 - Насос в работе \*)

(\* --- Аналоговые входы (AI) --- \*)

TANK_LEVEL_RAW : INT; (\* %IW0 - Уровень резервуара RAW \*)

TANK_LEVEL : REAL; (\* Уровень резервуара, % \*)

TEMP_SENSOR_RAW : INT; (\* %IW2 - Температура RAW \*)

TEMP_SENSOR : REAL; (\* Температура, °C \*)

(\* --- Аналоговые выходы (AO) --- \*)

VALVE_POS : REAL; (\* %QW0 - Положение клапана, % \*)

(\* --- Внутренние переменные --- \*)

SCALE_FACTOR : REAL := 100.0 / 32767.0;

END_VAR

(\* ======================== ТЕЛО ПРОГРАММЫ ======================== \*)

(\* Масштабирование аналоговых входов \*)

TANK_LEVEL := REAL(TANK_LEVEL_RAW) \* SCALE_FACTOR;

TEMP_SENSOR := REAL(TEMP_SENSOR_RAW) \* (200.0 / 32767.0) - 50.0;

(\* Логика управления насосом \*)

IF PUMP_START AND (TANK_LEVEL < 95.0) THEN

PUMP_RUN := TRUE;

ELSE

PUMP_RUN := FALSE;

END_IF;

(\* Управление клапаном по уровню \*)

IF TANK_LEVEL > 80.0 THEN

VALVE_POS := 0.0;

ELSIF TANK_LEVEL < 20.0 THEN

VALVE_POS := 100.0;

END_IF;

END_PROGRAM

## **5.4. Пайплайн генерации модуля**

- POST /upload_pdf - загрузка документации Эльбрус → извлечение текста через Qwen3.5-9B → чанкинг → индексация в pgvector
- POST /upload_signals - загрузка CSV/XLSX с таблицей сигналов → парсинг → структурирование
- POST /generate_module - запуск Planner агента: декомпозиция задачи на (а) разбор сигналов, (б) поиск документации, (в) генерация кода
- Engineer агент: RAG-запрос к pgvector за описанием адресного пространства Эльбрус → генерация VAR-секции
- Engineer агент: генерация тела программы с базовой логикой по сигналам
- Валидация: PLCverif проверяет синтаксис → возврат ошибок в ReAct-цикл при необходимости
- GET /module/download - скачивание .st файла

# **6\. Написание скриптов на IEC 61131-3 ST**

Режим работы: пользователь загружает PDF стандарта IEC 61131-3, задаёт системный промпт с соглашениями проекта, и описывает задачу на естественном языке. Система генерирует корректный ST-код с соблюдением требований стандарта.

## **6.1. Системный промпт для Engineer-агента (IEC 61131-3 режим)**

SYSTEM_PROMPT_ENGINEER = """

Ты - эксперт по программированию ПЛК на языке Structured Text

в соответствии со стандартом IEC 61131-3, третье издание.

ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ К ГЕНЕРИРУЕМОМУ КОДУ:

1\. Структура: PROGRAM / FUNCTION_BLOCK / FUNCTION с явными VAR-секциями

2\. Типы данных: только стандартные (BOOL, INT, DINT, REAL, TIME, STRING)

3\. Комментарии: (\* ... \*) для блоков, // для строк

4\. Именование: SCREAMING_SNAKE_CASE для глобальных, camelCase для локальных

5\. Каждая переменная должна иметь комментарий с описанием

6\. Запрещено: GOTO, EXIT внутри циклов кроме явного требования

7\. Таймеры: использовать стандартные FB TON, TOF, TP

8\. Арифметика с REAL: всегда проверять деление на ноль

ФОРМАТ ОТВЕТА:

\- Только код ST, никаких объяснений вне блоков (\* ... \*)

\- В начале - блок с метаданными: автор, дата, версия

\- В конце - секция VERIFY с описанием тестовых векторов

"""

## **6.2. Пример сгенерированного скрипта - управление светофором**

(\* ================================================================ \*)

(\* FUNCTION_BLOCK: TrafficLightController \*)

(\* IEC 61131-3 ST | Version: 1.0 \*)

(\* VERIFY: Green=25s, Yellow=5s, Red=30s cycle \*)

(\* ================================================================ \*)

FUNCTION_BLOCK TrafficLightController

VAR_INPUT

Enable : BOOL; (\* Разрешение работы светофора \*)

EmergencyOff: BOOL; (\* Аварийное отключение \*)

END_VAR

VAR_OUTPUT

RedLight : BOOL; (\* Красный сигнал \*)

YellowLight : BOOL; (\* Жёлтый сигнал \*)

GreenLight : BOOL; (\* Зелёный сигнал \*)

END_VAR

VAR

state : INT := 0; (\* 0=Red, 1=Green, 2=Yellow \*)

phaseTimer : TON; (\* Таймер фазы \*)

GREEN_TIME : TIME := T#25S;

YELLOW_TIME : TIME := T#5S;

RED_TIME : TIME := T#30S;

END_VAR

IF EmergencyOff THEN

RedLight := FALSE; YellowLight := FALSE; GreenLight := FALSE;

state := 0;

RETURN;

END_IF;

CASE state OF

0: (\* Красный \*)

RedLight := TRUE; YellowLight := FALSE; GreenLight := FALSE;

phaseTimer(IN := Enable, PT := RED_TIME);

IF phaseTimer.Q THEN state := 1; phaseTimer(IN := FALSE); END_IF;

1: (\* Зелёный \*)

RedLight := FALSE; YellowLight := FALSE; GreenLight := TRUE;

phaseTimer(IN := Enable, PT := GREEN_TIME);

IF phaseTimer.Q THEN state := 2; phaseTimer(IN := FALSE); END_IF;

2: (\* Жёлтый \*)

RedLight := FALSE; YellowLight := TRUE; GreenLight := FALSE;

phaseTimer(IN := Enable, PT := YELLOW_TIME);

IF phaseTimer.Q THEN state := 0; phaseTimer(IN := FALSE); END_IF;

END_CASE;

END_FUNCTION_BLOCK

## **6.3. Системные промпты для различных сценариев**

| **Сценарий**                  | **Ключевые директивы в системном промпте**                                                                     |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Базовая логика управления** | Требовать CASE/IF вместо вложенных IF; обязательный ELSE; проверка граничных состояний                         |
| **ПИД-регулятор**             | Использовать стандартный FB PIDFF или реализовать с антинасыщением (anti-windup); REAL-арифметика; проверка /0 |
| **Обработка аварий**          | Обязательный приоритет INTERLOCK перед управляющей логикой; все аварии в ALARM_LOG структуре                   |
| **Таймеры и счётчики**        | Только стандартные TON/TOF/TP и CTU/CTD/CTUD; сброс по условию; защита от переполнения                         |
| **Управление моторами**       | Цикл Enable → Ready → Running → Fault с явными переходами; защита от реверса без задержки                      |

# **7\. API-эндпоинты**

| **Эндпоинт**          | **Метод** | **Описание**                                                                                                          |
| --------------------- | --------- | --------------------------------------------------------------------------------------------------------------------- |
| /upload_pdf           | POST      | Загрузка PDF → OCR через Qwen3.5-9B → чанкинг → pgvector. Поддерживает документацию Эльбрус, стандарт IEC 61131-3, ТЗ |
| /upload_signals       | POST      | Загрузка CSV/XLSX с таблицей сигналов → парсинг → сохранение в БД                                                     |
| /chat                 | POST      | Запрос пользователя → ReAct-агент → ответ + сохранение истории + логирование токенов                                  |
| /chat/stream          | WS        | WebSocket: стриминг ответа агента с промежуточными Thought/Action/Observation блоками                                 |
| /generate_module      | POST      | Генерация ST-модуля для Эльбрус по загруженной документации и таблице сигналов                                        |
| /module/download      | GET       | Скачивание последнего сгенерированного .st файла                                                                      |
| /benchmark/run        | POST      | Запуск оценки на Agents4PLC. Параметры: subset (easy/medium/hard), n_tasks, configs\[\]                               |
| /benchmark/results    | GET       | Результаты последнего прогона: метрики по конфигурациям + разбивка по задачам                                         |
| /memories/search      | GET       | Ручной поиск по pgvector: ?q=...&type=doc\|chat&top_k=5                                                               |
| /monitoring/tokens    | GET       | Суммарные метрики токенов: ?session_id=...&from=...&to=...                                                            |
| /monitoring/cost      | GET       | Затраты в USD за период с разбивкой по агентам и моделям                                                              |
| /monitoring/dashboard | GET       | HTML-дашборд (Plotly) с графиками токенов, стоимости и latency                                                        |
| /health               | GET       | Проверка статуса: БД, pgvector, LLM-провайдер, PLCverif                                                               |

# **8\. План реализации (MVP за 3 дня)**

| **День**   | **Задачи**                                                                                                                                                                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **День 1** | Настройка: Docker + PostgreSQL 17 + pgvector + Ollama (Qwen3.5-9B). Структура проекта FastAPI + LangGraph. Клонирование Agents4PLC. Схема БД: memories, token_usage. Тест подключения к OpenRouter.                                                         |
| **День 2** | ReAct-агент в LangGraph: инструменты generate_st_code, search_memory, validate_with_plcverif. Обработка PDF: pdf2image → Qwen3.5-9B → чанкинг → pgvector. Загрузка таблицы сигналов CSV. LangChain callback для мониторинга токенов → запись в token_usage. |
| **День 3** | Тест на 10-20 задачах Agents4PLC: сбор Compilation Rate и Pass@1. Генерация ST-модуля для Эльбрус по тестовой таблице сигналов. Эндпоинт /monitoring/dashboard (Plotly-графики). README + демо-видео. Подача заявки.                                        |

# **9\. Критерии оценки на конкурсе**

| **Критерий JMLC 2026**     | **Что оценивается**                                 | **Как закрывает проект**                                                                                      |
| -------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Разработка и инженерия** | Git, Docker, CI, MLOps, качество кода, пайплайны ML | Docker Compose, FastAPI, LangGraph граф, чистый Python 3.11+, README с quickstart                             |
| **Data Science**           | EDA, предобработка, метрики, валидация              | Три конфигурации с метриками (Compilation Rate, Pass@1, Recall@5, MRR); сравнительный эксперимент             |
| **Применение ИИ**          | AI-агенты, AI-инструменты в разработке ПО           | Многоагентная ReAct-система, RAG, формальная верификация - прямое попадание в критерий                        |
| **Продуктовое мышление**   | Проблема, ЦА, конкуренты, MVP, импакт               | Рынок ПЛК РФ ~15 млрд руб. (CAGR 10%), анализ конкурентов (Siemens/Schneider - недоступны), поддержка Эльбрус |
| **Мотивация**              | Цели в AI/ML, согласованность с программой ИТМО     | Мотивационное письмо: промышленный AI + импортозамещение + исследовательский трек                             |

# **10\. Ссылки и ресурсы**

## **Бенчмарк и верификация**

- Agents4PLC paper: <https://arxiv.org/abs/2410.14209>
- Agents4PLC GitHub: <https://github.com/Luoji-zju/Agents4PLC_release>
- AutoPLC paper: <https://arxiv.org/html/2412.02410v2>
- PLCverif (CERN, open-source): <https://github.com/cern-sis/PLCverif>

## **Модели**

- Nex-N2-Pro (Planner): <https://openrouter.ai/nex-agi/nex-n2-pro>
- Qwen3.5-9B (Engineer): <https://openrouter.ai/qwen/qwen3.5-9b>
- Qwen3.5-4B (Retriever): <https://openrouter.ai/qwen/qwen3.5-4b>
- Laguna M.1 (альтернатива Engineer): <https://openrouter.ai/poolside/laguna-m1>

## **Инфраструктура**

- pgvector: <https://github.com/pgvector/pgvector>
- LangGraph: <https://langchain-ai.github.io/langgraph/>
- OpenRouter API: <https://openrouter.ai/docs>

## **Конкурс**

- Junior ML Contest 2026: <https://ai.itmo.ru/junior_ml_contest>
- Шаблон презентации: <https://docs.google.com/presentation/d/1cH42eNBEUKBulLkOK8Nd6UDPbKQ70JWkZ7yRAJQ2SM8/edit>