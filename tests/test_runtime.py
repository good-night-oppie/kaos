"""Tests for kaos.runtime — ContextVar agent-id binding."""

from __future__ import annotations

import asyncio

import pytest

from kaos.runtime import bind_agent, current_agent_id, require_current_agent_id


def test_unbound_returns_none() -> None:
    assert current_agent_id() is None


def test_bind_sets_and_resets() -> None:
    assert current_agent_id() is None
    with bind_agent("agent-a"):
        assert current_agent_id() == "agent-a"
    assert current_agent_id() is None


def test_nested_bind_restores_outer() -> None:
    with bind_agent("outer"):
        assert current_agent_id() == "outer"
        with bind_agent("inner"):
            assert current_agent_id() == "inner"
        assert current_agent_id() == "outer"
    assert current_agent_id() is None


def test_bind_resets_after_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with bind_agent("transient"):
            assert current_agent_id() == "transient"
            raise RuntimeError("boom")
    assert current_agent_id() is None


def test_require_raises_when_unbound() -> None:
    with pytest.raises(RuntimeError, match="No current agent bound"):
        require_current_agent_id()


def test_require_returns_bound_value() -> None:
    with bind_agent("agent-x"):
        assert require_current_agent_id() == "agent-x"


def test_asyncio_task_inherits_context() -> None:
    """asyncio.Task copies Context — bound agent_id propagates."""

    async def inner() -> str | None:
        return current_agent_id()

    async def driver() -> str | None:
        with bind_agent("agent-task"):
            return await asyncio.create_task(inner())

    result = asyncio.run(driver())
    assert result == "agent-task"


def test_parallel_tasks_see_their_own_agent_id() -> None:
    """Two concurrent agents must see distinct agent_ids (no leakage)."""
    results: dict[str, str | None] = {}

    async def agent_run(name: str) -> None:
        with bind_agent(name):
            await asyncio.sleep(0)  # yield to scheduler
            results[name] = current_agent_id()

    async def driver() -> None:
        await asyncio.gather(agent_run("agent-1"), agent_run("agent-2"))

    asyncio.run(driver())
    assert results == {"agent-1": "agent-1", "agent-2": "agent-2"}


# ── CCR integration ─────────────────────────────────────────────


class _SeenAgentTool:
    """Custom tool that records ``current_agent_id()`` at execution time."""

    def __init__(self) -> None:
        self.seen: list[str | None] = []

    def __call__(self, **kwargs) -> str:
        self.seen.append(current_agent_id())
        return "ok"


@pytest.mark.asyncio
async def test_tool_handler_sees_current_agent_id_via_contextvar(tmp_path):
    """Custom tool handler reads agent_id via ContextVar — no explicit kwarg."""
    from unittest.mock import AsyncMock, MagicMock

    from kaos.core import Kaos
    from kaos.ccr.runner import ClaudeCodeRunner, ModelResponse, ToolCall
    from kaos.ccr.tools import ToolDefinition
    from kaos.router.gepa import GEPARouter

    afs = Kaos(db_path=str(tmp_path / "ctx.db"))
    try:
        router = MagicMock(spec=GEPARouter)
        router.route = AsyncMock()
        router.route.side_effect = [
            ModelResponse(
                content="calling tool",
                tool_calls=[ToolCall(id="tc1", name="see_agent", input={})],
                stop_reason="tool_use",
            ),
            ModelResponse(content="done", tool_calls=[], stop_reason="end_turn"),
        ]

        ccr = ClaudeCodeRunner(afs, router)
        sentinel = _SeenAgentTool()
        ccr.register_tool(ToolDefinition(
            name="see_agent",
            description="record current_agent_id",
            parameters={"type": "object", "properties": {}},
            handler=sentinel,
        ))

        agent_id = afs.spawn("ctx-agent")
        await ccr.run_agent(agent_id, "use the tool")
        assert sentinel.seen == [agent_id]
    finally:
        afs.close()
