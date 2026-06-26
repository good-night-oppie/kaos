"""ContextVar binding for the currently-running kaos agent_id."""

from __future__ import annotations

import contextlib
import contextvars
from typing import Iterator


_current_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kaos_current_agent_id", default=None
)


def current_agent_id() -> str | None:
    return _current_agent_id.get()


@contextlib.contextmanager
def bind_agent(agent_id: str) -> Iterator[str]:
    token = _current_agent_id.set(agent_id)
    try:
        yield agent_id
    finally:
        _current_agent_id.reset(token)


def require_current_agent_id() -> str:
    aid = _current_agent_id.get()
    if aid is None:
        raise RuntimeError("No current agent bound")
    return aid
