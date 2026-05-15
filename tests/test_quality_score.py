"""Track A (v0.8.3) — continuous quality score on skill outcomes.

Verifies the [0,1] quality signal is recorded, validated, fed into the
plasticity ranker, and fully backward-compatible with binary outcomes.
"""

from __future__ import annotations

import pytest

from kaos import Kaos
from kaos.skills import SkillStore, _quality_signal_map
from kaos.dream.signals import wilson_lower_bound, weighted_score


@pytest.fixture
def afs(tmp_path, monkeypatch):
    monkeypatch.setenv("KAOS_DREAM_AUTO", "0")
    fs = Kaos(db_path=str(tmp_path / "q.db"))
    yield fs
    fs.close()


@pytest.fixture
def sk(afs):
    return SkillStore(afs.conn)


@pytest.fixture
def agent(afs):
    return afs.spawn("a")


class TestRecordOutcomeBinary:
    def test_binary_only_leaves_quality_null(self, sk, afs, agent):
        sid = sk.save(name="s", description="d", template="t",
                      source_agent_id=agent, tags=[])
        sk.record_outcome(sid, success=True, agent_id=agent)
        row = afs.conn.execute(
            "SELECT success, quality FROM skill_uses WHERE skill_id=?", (sid,)
        ).fetchone()
        assert row["success"] == 1
        assert row["quality"] is None

    def test_aggregate_columns_still_move(self, sk, afs, agent):
        sid = sk.save(name="s", description="d", template="t",
                      source_agent_id=agent, tags=[])
        sk.record_outcome(sid, success=True, agent_id=agent)
        sk.record_outcome(sid, success=False, agent_id=agent)
        s = sk.get(sid)
        assert s.use_count == 2 and s.success_count == 1


class TestRecordOutcomeQuality:
    def test_quality_stored(self, sk, afs, agent):
        sid = sk.save(name="s", description="d", template="t",
                      source_agent_id=agent, tags=[])
        sk.record_outcome(sid, success=True, quality=0.7, agent_id=agent)
        row = afs.conn.execute(
            "SELECT quality FROM skill_uses WHERE skill_id=?", (sid,)
        ).fetchone()
        assert row["quality"] == pytest.approx(0.7)

    @pytest.mark.parametrize("bad", [1.1, -0.1, 2.0, -5.0])
    def test_out_of_range_raises(self, sk, agent, bad):
        sid = sk.save(name="s", description="d", template="t",
                      source_agent_id=agent, tags=[])
        with pytest.raises(ValueError, match=r"quality must be in"):
            sk.record_outcome(sid, success=True, quality=bad, agent_id=agent)

    @pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
    def test_boundary_values_accepted(self, sk, agent, ok):
        sid = sk.save(name="s", description="d", template="t",
                      source_agent_id=agent, tags=[])
        sk.record_outcome(sid, success=True, quality=ok, agent_id=agent)  # no raise


class TestQualitySignalMap:
    def test_absent_when_no_graded_rows(self, sk, afs, agent):
        sid = sk.save(name="s", description="d", template="t",
                      source_agent_id=agent, tags=[])
        sk.record_outcome(sid, success=True, agent_id=agent)  # binary only
        assert _quality_signal_map(afs.conn, [sid]) == {}

    def test_present_with_graded_rows(self, sk, afs, agent):
        sid = sk.save(name="s", description="d", template="t",
                      source_agent_id=agent, tags=[])
        sk.record_outcome(sid, success=True, quality=0.8, agent_id=agent)
        sk.record_outcome(sid, success=True, quality=0.6, agent_id=agent)
        m = _quality_signal_map(afs.conn, [sid])
        eff, uses = m[sid]
        assert uses == 2
        assert eff == pytest.approx(1.4)

    def test_mixed_binary_and_graded(self, sk, afs, agent):
        sid = sk.save(name="s", description="d", template="t",
                      source_agent_id=agent, tags=[])
        sk.record_outcome(sid, success=True, agent_id=agent)            # binary 1
        sk.record_outcome(sid, success=False, agent_id=agent)           # binary 0
        sk.record_outcome(sid, success=True, quality=0.5, agent_id=agent)  # graded
        m = _quality_signal_map(afs.conn, [sid])
        eff, uses = m[sid]
        # eff = 1 (binary success) + 0 (binary fail) + 0.5 (graded) = 1.5
        assert uses == 3
        assert eff == pytest.approx(1.5)


class TestWilsonContinuous:
    def test_fractional_successes_valid(self):
        # 7.5 effective successes over 10 uses — must be a real number in [0,1]
        lb = wilson_lower_bound(7.5, 10)
        assert 0.0 <= lb <= 1.0

    def test_quality_beats_partial_binary(self):
        # A skill graded 0.9 avg over 5 should outrank one that's 3/5 binary
        hi = weighted_score(bm25_score=1.0, uses=5, successes=4.5,
                            last_used_at=None)
        lo = weighted_score(bm25_score=1.0, uses=5, successes=3.0,
                            last_used_at=None)
        assert hi > lo


class TestRankingIntegration:
    def test_quality_reranks_search(self, sk, afs, agent):
        # Two equally-relevant skills; one graded high, one graded low.
        a = sk.save(name="alpha-handler",
                    description="handle alpha widget task",
                    template="t", source_agent_id=agent, tags=[])
        b = sk.save(name="alpha-helper",
                    description="handle alpha widget task",
                    template="t", source_agent_id=agent, tags=[])
        for _ in range(5):
            sk.record_outcome(a, success=True, quality=0.95, agent_id=agent)
            sk.record_outcome(b, success=True, quality=0.10, agent_id=agent)
        results = sk.search("alpha widget task", limit=2, rank="weighted")
        assert results[0].skill_id == a  # higher quality ranked first

    def test_binary_path_unchanged_when_no_quality(self, sk, afs, agent):
        a = sk.save(name="beta-one", description="beta task",
                    template="t", source_agent_id=agent, tags=[])
        b = sk.save(name="beta-two", description="beta task",
                    template="t", source_agent_id=agent, tags=[])
        for _ in range(5):
            sk.record_outcome(a, success=True, agent_id=agent)
            sk.record_outcome(b, success=False, agent_id=agent)
        results = sk.search("beta task", limit=2, rank="weighted")
        assert results[0].skill_id == a  # binary success still wins
