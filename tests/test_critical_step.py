"""Track B2 (v0.8.3) — critical-step localizer.

Verifies the localizer reconstructs a trajectory, points at the earliest
decisive step (not just the visible error), falls back to the LLM only
under the confidence floor, caches by trajectory shape, and persists.
"""

from __future__ import annotations

import pytest

from kaos import Kaos
from kaos.dream.phases.localize import (
    localize,
    latest_for_agent,
    _load_trace,
    _heuristic_localize,
)


@pytest.fixture
def afs(tmp_path, monkeypatch):
    monkeypatch.setenv("KAOS_DREAM_AUTO", "0")
    fs = Kaos(db_path=str(tmp_path / "loc.db"))
    yield fs
    fs.close()


def _tool(afs, agent_id, call_id, name, status="success", err=None):
    afs.conn.execute(
        "INSERT INTO tool_calls (call_id, agent_id, tool_name, input, "
        "status, error_message, started_at) VALUES "
        "(?, ?, ?, '{}', ?, ?, strftime('%Y-%m-%dT%H:%M:%f','now'))",
        (call_id, agent_id, name, status, err),
    )
    afs.conn.commit()


class TestTraceLoading:
    def test_empty_agent_no_trace(self, afs):
        aid = afs.spawn("a")
        assert _load_trace(afs.conn, aid) == []

    def test_merges_tools_and_log(self, afs):
        aid = afs.spawn("a")
        from kaos.shared_log import SharedLog
        log = SharedLog(afs.conn)
        log.intent(aid, "do the risky thing")
        _tool(afs, aid, "c1", "read-config")
        steps = _load_trace(afs.conn, aid)
        kinds = {s.kind for s in steps}
        assert kinds == {"tool", "log"}
        assert len(steps) == 2


class TestHeuristicLocalizer:
    def test_no_error_returns_none(self, afs):
        aid = afs.spawn("a")
        _tool(afs, aid, "c1", "read-thing")
        assert _heuristic_localize(_load_trace(afs.conn, aid)) is None

    def test_error_with_no_prior_decision_is_itself_critical(self, afs):
        aid = afs.spawn("a")
        _tool(afs, aid, "c1", "fetch", status="error",
              err="Connection refused")
        idx, rationale, conf = _heuristic_localize(_load_trace(afs.conn, aid))
        assert idx == 0
        assert "itself the critical step" in rationale

    def test_earliest_decision_before_error_is_critical(self, afs):
        aid = afs.spawn("a")
        from kaos.shared_log import SharedLog
        log = SharedLog(afs.conn)
        log.intent(aid, "deploy v2 to prod")     # decisive (log:intent)
        _tool(afs, aid, "c1", "read-manifest")   # read-only
        _tool(afs, aid, "c2", "run-deploy", status="error",
              err="rollout failed")              # visible error, late
        steps = _load_trace(afs.conn, aid)
        idx, rationale, conf = _heuristic_localize(steps)
        assert steps[idx].kind == "log"          # the intent, not the error
        assert steps[idx].is_decision
        assert "Earliest decisive step" in rationale
        assert conf > 0.55

    def test_confidence_grows_with_gap(self, afs):
        a1 = afs.spawn("near")
        from kaos.shared_log import SharedLog
        log = SharedLog(afs.conn)
        log.intent(a1, "x")
        _tool(afs, a1, "n1", "boom", status="error", err="e")
        near = _heuristic_localize(_load_trace(afs.conn, a1))[2]

        a2 = afs.spawn("far")
        log.intent(a2, "y")
        for i in range(4):
            _tool(afs, a2, f"f{i}", f"step{i}")
        _tool(afs, a2, "fE", "boom", status="error", err="e")
        far = _heuristic_localize(_load_trace(afs.conn, a2))[2]
        assert far > near


class TestLLMFallback:
    def test_llm_only_below_floor(self, afs):
        aid = afs.spawn("a")
        # High-confidence heuristic shape (big gap) → LLM must NOT be called
        from kaos.shared_log import SharedLog
        log = SharedLog(afs.conn)
        log.intent(aid, "x")
        for i in range(5):
            _tool(afs, aid, f"s{i}", f"step{i}")
        _tool(afs, aid, "sE", "boom", status="error", err="e")
        calls = []
        cs = localize(afs.conn, aid, llm_call_fn=lambda p: calls.append(p) or "{}",
                      persist=False)
        assert cs.method == "heuristic"
        assert calls == []

    def test_llm_consulted_and_cached(self, afs):
        aid = afs.spawn("a")
        # Low-confidence shape: error with no prior decision → conf 0.7;
        # set floor above it so the LLM is consulted.
        _tool(afs, aid, "c1", "boom", status="error", err="e")
        calls = []

        def fake(prompt):
            calls.append(prompt)
            return '{"index":0,"rationale":"llm says step 0","confidence":0.9}'

        cs1 = localize(afs.conn, aid, llm_call_fn=fake,
                       heuristic_floor=0.95, persist=False)
        assert cs1.method == "llm"
        assert cs1.rationale == "llm says step 0"

        cs2 = localize(afs.conn, aid, llm_call_fn=fake,
                       heuristic_floor=0.95, persist=False)
        assert cs2.method == "llm-cached"
        assert len(calls) == 1  # cached, no second call

    def test_llm_garbage_falls_back_to_heuristic(self, afs):
        aid = afs.spawn("a")
        _tool(afs, aid, "c1", "boom", status="error", err="e")
        cs = localize(afs.conn, aid, llm_call_fn=lambda p: "not json",
                      heuristic_floor=0.95, persist=False)
        assert cs.method == "heuristic"


class TestPersistence:
    def test_persisted_and_retrievable(self, afs):
        aid = afs.spawn("a")
        from kaos.shared_log import SharedLog
        SharedLog(afs.conn).intent(aid, "deploy")
        _tool(afs, aid, "c1", "run-deploy", status="error", err="failed")
        cs = localize(afs.conn, aid)
        assert cs is not None
        got = latest_for_agent(afs.conn, aid)
        assert got is not None
        assert got.rationale == cs.rationale
        assert got.method == cs.method

    def test_no_failure_returns_none(self, afs):
        aid = afs.spawn("a")
        _tool(afs, aid, "c1", "read-only-thing")
        assert localize(afs.conn, aid) is None
        assert latest_for_agent(afs.conn, aid) is None
