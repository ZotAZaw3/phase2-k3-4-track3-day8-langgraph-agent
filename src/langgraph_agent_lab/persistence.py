"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # backends are optional extras, imported lazily below
    from langgraph.checkpoint.base import BaseCheckpointSaver

DEFAULT_SQLITE_PATH = "outputs/checkpoints.sqlite"


def build_checkpointer(
    kind: str = "memory", database_url: str | None = None
) -> BaseCheckpointSaver | None:
    """Return a LangGraph checkpointer.

    - "none"     → no persistence (state lives only inside a single invoke)
    - "memory"   → MemorySaver, per-process, good enough for tests
    - "sqlite"   → SqliteSaver, survives process restart (crash-resume evidence)
    - "postgres" → PostgresSaver, for multi-process deployments
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError("Install: pip install langgraph-checkpoint-sqlite") from exc

        path = Path((database_url or DEFAULT_SQLITE_PATH).removeprefix("sqlite:///"))
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: LangGraph may touch the connection from worker threads.
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(conn=conn)
    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import (  # type: ignore[import-not-found]
                PostgresSaver,
            )
        except ImportError as exc:
            raise RuntimeError("Install: pip install langgraph-checkpoint-postgres") from exc
        if not database_url:
            raise ValueError("postgres checkpointer requires database_url")
        saver = PostgresSaver.from_conn_string(database_url).__enter__()
        saver.setup()
        return saver
    raise ValueError(f"Unknown checkpointer kind: {kind}")
