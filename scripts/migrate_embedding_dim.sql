-- Migrate memories.embedding to 768 dims (google/embeddinggemma-300m).
-- WARNING: truncates existing vectors — re-index documents after migration.

TRUNCATE memories RESTART IDENTITY CASCADE;

DROP INDEX IF EXISTS memories_hnsw;
DROP INDEX IF EXISTS memories_ivf;

ALTER TABLE memories
    ALTER COLUMN embedding TYPE vector(768);

CREATE INDEX IF NOT EXISTS memories_hnsw ON memories
    USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
