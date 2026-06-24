-- ============================================================
-- PLC Agent — инициализация базы данных
-- PostgreSQL 17 + pgvector
-- ============================================================

-- Расширение pgvector (обязательно первым)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─── Таблица: memories (векторная память) ───────────────────
CREATE TABLE IF NOT EXISTS memories (
    id          SERIAL PRIMARY KEY,
    content     TEXT        NOT NULL,
    embedding   vector(1536) NOT NULL,          -- text-embedding-3-small / BGE-M3
    metadata    JSONB       NOT NULL DEFAULT '{}',
    -- metadata schema:
    -- { "type": "doc"|"chat",
    --   "source": "<filename>",
    --   "session_id": "<uuid>",
    --   "chunk_index": 0,
    --   "page": 1 }
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW-индекс для быстрого косинусного поиска
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw
    ON memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Индексы для фильтрации по метаданным
CREATE INDEX IF NOT EXISTS memories_type_idx
    ON memories ((metadata->>'type'));
CREATE INDEX IF NOT EXISTS memories_session_idx
    ON memories ((metadata->>'session_id'));
CREATE INDEX IF NOT EXISTS memories_created_idx
    ON memories (created_at DESC);

-- ─── Таблица: sessions (сессии пользователей) ───────────────
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata    JSONB       NOT NULL DEFAULT '{}'
    -- metadata: { "user_agent": "...", "task_type": "generate|benchmark|..." }
);

-- ─── Таблица: token_usage (мониторинг токенов и стоимости) ──
CREATE TABLE IF NOT EXISTS token_usage (
    id                  SERIAL      PRIMARY KEY,
    session_id          UUID        REFERENCES sessions(id) ON DELETE SET NULL,
    agent_name          VARCHAR(50),            -- planner | engineer | retriever
    tool_name           VARCHAR(100),           -- generate_st_code | search_memory | ...
    model_id            VARCHAR(100),           -- nex-agi/nex-n2-pro | qwen/qwen3.5-9b
    prompt_tokens       INTEGER     NOT NULL DEFAULT 0,
    completion_tokens   INTEGER     NOT NULL DEFAULT 0,
    total_tokens        INTEGER     NOT NULL DEFAULT 0,
    cost_usd            NUMERIC(12,8) NOT NULL DEFAULT 0,
    latency_ms          INTEGER,
    react_step          VARCHAR(20),            -- thought | action | observation
    task_id             VARCHAR(50),            -- ID задачи Agents4PLC (M1, H3 ...)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS token_usage_session_idx
    ON token_usage (session_id);
CREATE INDEX IF NOT EXISTS token_usage_agent_idx
    ON token_usage (agent_name);
CREATE INDEX IF NOT EXISTS token_usage_task_idx
    ON token_usage (task_id)
    WHERE task_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS token_usage_created_idx
    ON token_usage (created_at DESC);

-- ─── Таблица: benchmark_results (результаты Agents4PLC) ─────
CREATE TABLE IF NOT EXISTS benchmark_results (
    id                          SERIAL      PRIMARY KEY,
    run_id                      UUID        NOT NULL DEFAULT uuid_generate_v4(),
    task_id                     VARCHAR(20) NOT NULL,   -- M1, M2, H1 ...
    difficulty                  VARCHAR(10),             -- easy | medium | hard
    config                      VARCHAR(50),             -- baseline | rag_only | full_agent
    generated_code              TEXT,
    compilation_ok              BOOLEAN,
    formal_verification_ok      BOOLEAN,
    execution_correct           BOOLEAN,
    total_tokens                INTEGER,
    cost_usd                    NUMERIC(12,8),
    latency_ms                  INTEGER,
    error_message               TEXT,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS bench_run_idx
    ON benchmark_results (run_id);
CREATE INDEX IF NOT EXISTS bench_task_idx
    ON benchmark_results (task_id);
CREATE INDEX IF NOT EXISTS bench_config_idx
    ON benchmark_results (config);

-- ─── Таблица: uploaded_docs (загруженные документы) ─────────
CREATE TABLE IF NOT EXISTS uploaded_docs (
    id          SERIAL      PRIMARY KEY,
    filename    VARCHAR(255) NOT NULL,
    doc_type    VARCHAR(50),                    -- iec_standard | elbrus_manual | tz | signals
    page_count  INTEGER,
    chunk_count INTEGER,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending | processing | done | error
    error_msg   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── View: токены по сессии (для дашборда) ──────────────────
CREATE OR REPLACE VIEW v_session_token_summary AS
SELECT
    session_id,
    SUM(total_tokens)       AS total_tokens,
    SUM(prompt_tokens)      AS prompt_tokens,
    SUM(completion_tokens)  AS completion_tokens,
    SUM(cost_usd)           AS total_cost_usd,
    COUNT(*)                AS llm_calls,
    AVG(latency_ms)::INT    AS avg_latency_ms,
    MAX(created_at)         AS last_call_at
FROM token_usage
WHERE session_id IS NOT NULL
GROUP BY session_id;

-- ─── View: метрики бенчмарка по конфигурациям ───────────────
CREATE OR REPLACE VIEW v_benchmark_summary AS
SELECT
    run_id,
    config,
    difficulty,
    COUNT(*)                                                    AS total_tasks,
    ROUND(100.0 * SUM(CASE WHEN compilation_ok THEN 1 ELSE 0 END) / COUNT(*), 1)
                                                                AS compilation_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN formal_verification_ok THEN 1 ELSE 0 END) / COUNT(*), 1)
                                                                AS fv_pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN execution_correct THEN 1 ELSE 0 END) / COUNT(*), 1)
                                                                AS execution_correct_pct,
    SUM(total_tokens)                                           AS total_tokens,
    ROUND(SUM(cost_usd)::NUMERIC, 4)                            AS total_cost_usd,
    ROUND(AVG(cost_usd)::NUMERIC, 6)                            AS avg_cost_per_task_usd,
    MAX(created_at)                                             AS run_at
FROM benchmark_results
GROUP BY run_id, config, difficulty
ORDER BY run_at DESC, config;

-- ─── Приветственный комментарий ─────────────────────────────
DO $$
BEGIN
    RAISE NOTICE '✅ PLC Agent DB initialized: memories, sessions, token_usage, benchmark_results, uploaded_docs';
    RAISE NOTICE '✅ Views: v_session_token_summary, v_benchmark_summary';
END $$;