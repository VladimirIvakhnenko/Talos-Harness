"""app/agents/cancellation.py — Per-session cancel flag + message queue."""
from __future__ import annotations

import asyncio
from collections import deque


class AgentRunner:
    """Per-session cancellation + message queue. One instance per session_id."""

    _instances: dict[str, AgentRunner] = {}

    def __init__(self) -> None:
        self.cancel = asyncio.Event()
        self.queue: deque[str] = deque()
        self.running = False

    @classmethod
    def get(cls, session_id: str) -> AgentRunner:
        if session_id not in cls._instances:
            cls._instances[session_id] = cls()
        return cls._instances[session_id]

    def request_cancel(self) -> None:
        self.cancel.set()

    def enqueue(self, message: str) -> None:
        self.queue.append(message)

    def drain(self) -> str | None:
        return self.queue.popleft() if self.queue else None

    def queue_size(self) -> int:
        return len(self.queue)
