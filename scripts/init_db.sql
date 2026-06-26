CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── memories ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memories (
    id              SERIAL PRIMARY KEY,
    content         TEXT NOT NULL,
    embedding       vector(768) NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}',
    parent_id       INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    chunk_level     VARCHAR(10) NOT NULL DEFAULT 'single',
    doc_position    INTEGER,
    embedding_model VARCHAR(50) DEFAULT 'embeddinggemma-300m',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- tsvector для sparse-поиска
ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS tsvec tsvector
    GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED;

CREATE INDEX IF NOT EXISTS memories_hnsw ON memories
    USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX IF NOT EXISTS memories_gin   ON memories USING gin(tsvec);
CREATE INDEX IF NOT EXISTS memories_type  ON memories ((metadata->>'type'));
CREATE INDEX IF NOT EXISTS memories_sess  ON memories ((metadata->>'session_id'));
CREATE INDEX IF NOT EXISTS memories_lvl   ON memories (chunk_level);
CREATE INDEX IF NOT EXISTS memories_par   ON memories (parent_id);
CREATE INDEX IF NOT EXISTS memories_doc_source
    ON memories ((metadata->>'type'), (metadata->>'scope'), (metadata->>'session_id'), (metadata->>'source'))
    WHERE metadata->>'type' = 'doc';
CREATE INDEX IF NOT EXISTS memories_chat_session
    ON memories ((metadata->>'session_id'), created_at)
    WHERE metadata->>'type' = 'chat';

-- ── sessions ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title      TEXT NOT NULL DEFAULT 'Новый чат',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata   JSONB NOT NULL DEFAULT '{}'
);

-- ── token_usage ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS token_usage (
    id                SERIAL PRIMARY KEY,
    session_id        UUID REFERENCES sessions(id) ON DELETE SET NULL,
    agent_name        VARCHAR(50),
    tool_name         VARCHAR(100),
    model_id          VARCHAR(100),
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    cost_usd          NUMERIC(12,8) NOT NULL DEFAULT 0,
    latency_ms        INTEGER,
    react_step        VARCHAR(20),
    task_id           VARCHAR(50),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS tu_session ON token_usage (session_id);
CREATE INDEX IF NOT EXISTS tu_agent   ON token_usage (agent_name);
CREATE INDEX IF NOT EXISTS tu_task    ON token_usage (task_id) WHERE task_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS tu_ts      ON token_usage (created_at DESC);

-- ── benchmark_results ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS benchmark_results (
    id                     SERIAL PRIMARY KEY,
    run_id                 UUID NOT NULL DEFAULT uuid_generate_v4(),
    task_id                VARCHAR(20) NOT NULL,
    difficulty             VARCHAR(10),
    config                 VARCHAR(50),
    generated_code         TEXT,
    compilation_ok         BOOLEAN,
    formal_verification_ok BOOLEAN,
    execution_correct      BOOLEAN,
    matiec_errors          JSONB DEFAULT '[]',
    total_tokens           INTEGER,
    cost_usd               NUMERIC(12,8),
    latency_ms             INTEGER,
    error_message          TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS br_run    ON benchmark_results (run_id);
CREATE INDEX IF NOT EXISTS br_task   ON benchmark_results (task_id);
CREATE INDEX IF NOT EXISTS br_config ON benchmark_results (config);

-- ── uploaded_docs ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS uploaded_docs (
    id          SERIAL PRIMARY KEY,
    filename    VARCHAR(255) NOT NULL,
    doc_type    VARCHAR(50),
    page_count  INTEGER,
    chunk_count INTEGER,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    error_msg   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Views ─────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_token_summary AS
SELECT
    session_id,
    SUM(total_tokens)      AS total_tokens,
    SUM(cost_usd)          AS total_cost_usd,
    COUNT(*)               AS llm_calls,
    AVG(latency_ms)::INT   AS avg_latency_ms,
    MAX(created_at)        AS last_call_at
FROM token_usage WHERE session_id IS NOT NULL
GROUP BY session_id;

CREATE OR REPLACE VIEW v_benchmark_summary AS
SELECT
    run_id, config, difficulty,
    COUNT(*) AS total_tasks,
    ROUND(100.0 * SUM(CASE WHEN compilation_ok         THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1) AS compilation_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN formal_verification_ok THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1) AS fv_pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN execution_correct      THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1) AS exec_correct_pct,
    SUM(total_tokens)                          AS total_tokens,
    ROUND(SUM(cost_usd)::NUMERIC, 4)           AS total_cost_usd,
    MAX(created_at)                            AS run_at
FROM benchmark_results
GROUP BY run_id, config, difficulty
ORDER BY run_at DESC;

DO $$ BEGIN RAISE NOTICE 'PLC Agent DB initialized'; END $$;