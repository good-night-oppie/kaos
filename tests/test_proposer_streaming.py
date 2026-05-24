"""PR-1 (v0.9 #11) — claude_code provider streaming + idle/wall timeouts.

Uses a real Python subprocess as the "fake claude" so we exercise the
actual asyncio.create_subprocess_exec + incremental stdout reading path,
not a mocked one. The fake CLI's behaviour (steady chunks / stall /
wall-exceed / non-zero exit) is controlled via argv.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from kaos.router.providers import ClaudeCodeProvider, ProposerStalled


def _fake_claude_cmd(mode: str, **kw) -> list[str]:
    """Return a python -c argv that simulates the claude CLI."""
    # Single-line script (no triple-quoted blocks → no \n parsing surprises).
    parts: list[str]
    if mode == "fast":
        parts = ["import sys", "sys.stdout.write('hello world')",
                 "sys.stdout.flush()"]
    elif mode == "steady":
        n = int(kw["chunks"])
        s = float(kw["sleep"])
        parts = ["import sys, time",
                 f"[ (sys.stdout.write(f'chunk{{i}} '), sys.stdout.flush(), "
                 f"time.sleep({s})) for i in range({n}) ]"]
    elif mode == "stall":
        s = float(kw["sleep"])
        parts = ["import sys, time",
                 "sys.stdout.write('partial '); sys.stdout.flush()",
                 f"time.sleep({s})",
                 "sys.stdout.write('rest'); sys.stdout.flush()"]
    elif mode == "rc":
        rc = int(kw["rc"])
        parts = ["import sys",
                 "sys.stderr.write('boom'); sys.stderr.flush()",
                 f"sys.exit({rc})"]
    else:
        raise ValueError(mode)
    return [sys.executable, "-c", "; ".join(parts)]


def _provider(*, timeout: float, idle_timeout: float) -> ClaudeCodeProvider:
    p = ClaudeCodeProvider(timeout=timeout, idle_timeout=idle_timeout)
    return p


def _run(coro):
    return asyncio.run(coro)


class TestStreamingHappyPath:
    def test_fast_response_returns_text(self):
        p = _provider(timeout=10, idle_timeout=5)
        cmd = _fake_claude_cmd("fast")
        out = _run(p._run_streaming_once(cmd, b"", {}))
        assert out == "hello world"

    def test_steady_chunks_within_idle_succeeds(self):
        # 4 chunks, 0.3s apart → never stalls beyond idle=2s.
        p = _provider(timeout=15, idle_timeout=2)
        cmd = _fake_claude_cmd("steady", chunks=4, sleep=0.3)
        out = _run(p._run_streaming_once(cmd, b"", {}))
        assert "chunk0" in out and "chunk3" in out


class TestStallDetection:
    def test_stall_longer_than_idle_raises_ProposerStalled(self):
        # Sleeps 3s mid-stream, idle=1s → must raise ProposerStalled
        # (NOT TimeoutError; wall is much larger than the total run).
        p = _provider(timeout=30, idle_timeout=1)
        cmd = _fake_claude_cmd("stall", sleep=3)
        with pytest.raises(ProposerStalled) as ei:
            _run(p._run_streaming_once(cmd, b"", {}))
        assert "no output for" in str(ei.value)

    def test_stall_message_reports_bytes_received(self):
        p = _provider(timeout=30, idle_timeout=1)
        cmd = _fake_claude_cmd("stall", sleep=3)
        with pytest.raises(ProposerStalled) as ei:
            _run(p._run_streaming_once(cmd, b"", {}))
        # At least the "partial " (8 bytes) preface was received before stall.
        assert "received" in str(ei.value)


class TestWallTimeout:
    def test_wall_exceeded_raises_TimeoutError(self):
        # idle = wall — the fake stalls 5s; wall hits first (~2s).
        p = _provider(timeout=2, idle_timeout=10)
        cmd = _fake_claude_cmd("stall", sleep=5)
        with pytest.raises(TimeoutError):
            _run(p._run_streaming_once(cmd, b"", {}))


class TestProcessFailure:
    def test_non_zero_exit_raises_RuntimeError_with_stderr(self):
        p = _provider(timeout=10, idle_timeout=5)
        cmd = _fake_claude_cmd("rc", rc=2)
        with pytest.raises(RuntimeError) as ei:
            _run(p._run_streaming_once(cmd, b"", {}))
        assert "boom" in str(ei.value)
        assert "rc=2" in str(ei.value)


class TestSearchLoopSurvival:
    """Smoke: ProposerStalled is a distinct exception type the search loop
    can catch separately from TimeoutError and generic Exception. This is
    the actual P0 #11 contract: stalls must not kill the loop."""

    def test_exception_hierarchy(self):
        assert issubclass(ProposerStalled, Exception)
        assert not issubclass(ProposerStalled, TimeoutError)
        assert not issubclass(TimeoutError, ProposerStalled)
