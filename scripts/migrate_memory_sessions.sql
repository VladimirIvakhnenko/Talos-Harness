-- Миграция: сессии чатов, удаление facts, индексы для дедупликации документов

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT 'Новый чат';

-- Удаляем facts (больше не используются)
DELETE FROM memories WHERE metadata->>'type' = 'fact';

-- Индекс для дедупликации документов по source + scope + session
CREATE INDEX IF NOT EXISTS memories_doc_source
    ON memories ((metadata->>'type'), (metadata->>'scope'), (metadata->>'session_id'), (metadata->>'source'))
    WHERE metadata->>'type' = 'doc';

-- Индекс для загрузки истории чата
CREATE INDEX IF NOT EXISTS memories_chat_session
    ON memories ((metadata->>'session_id'), created_at)
    WHERE metadata->>'type' = 'chat';

DO $$ BEGIN RAISE NOTICE 'Memory sessions migration applied'; END $$;
