-- Расширение benchmark_results для ST coding suite
ALTER TABLE benchmark_results
    ADD COLUMN IF NOT EXISTS session_id         UUID,
    ADD COLUMN IF NOT EXISTS validation_attempts INTEGER,
    ADD COLUMN IF NOT EXISTS generate_attempts   INTEGER,
    ADD COLUMN IF NOT EXISTS pass_at_1           BOOLEAN,
    ADD COLUMN IF NOT EXISTS benchmark_suite     VARCHAR(20) DEFAULT 'agents4plc',
    ADD COLUMN IF NOT EXISTS extra_metrics       JSONB DEFAULT '{}';

CREATE INDEX IF NOT EXISTS br_suite ON benchmark_results (benchmark_suite);

DROP VIEW IF EXISTS v_benchmark_summary;
CREATE VIEW v_benchmark_summary AS
SELECT
    run_id,
    config,
    difficulty,
    benchmark_suite,
    COUNT(*) AS total_tasks,
    ROUND(100.0 * SUM(CASE WHEN compilation_ok THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1)
        AS compilation_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN pass_at_1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1)
        AS pass_at_1_pct,
    ROUND(100.0 * SUM(CASE WHEN formal_verification_ok THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1)
        AS fv_pass_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN execution_correct THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1)
        AS exec_correct_pct,
    ROUND(AVG(validation_attempts)::NUMERIC, 2) AS avg_validation_attempts,
    SUM(total_tokens) AS total_tokens,
    ROUND(SUM(cost_usd)::NUMERIC, 4) AS total_cost_usd,
    ROUND(AVG(latency_ms)::NUMERIC, 0) AS avg_latency_ms,
    MAX(created_at) AS run_at
FROM benchmark_results
GROUP BY run_id, config, difficulty, benchmark_suite
ORDER BY run_at DESC;
