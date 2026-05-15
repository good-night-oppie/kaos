"""Track B integration (v0.8.3) — ISC ↔ quality ↔ taxonomy ↔ critical_step.

Proves the three tracks compose: a partially-passed ISA records a
quality-graded skill outcome (A), failed criteria group by reasoning
taxonomy (B1), and a failing criterion links to the trajectory step
where it went wrong (B2).
"""

from __future__ import annotations

import pytest

from kaos import Kaos
from kaos.ideal_state import IdealStateStore
from kaos.skills import SkillStore, _quality_signal_map
from kaos.shared_log import SharedLog


@pytest.fixture
def afs(tmp_path, monkeypatch):
    monkeypatch.setenv("KAOS_DREAM_AUTO", "0")
    fs = Kaos(db_path=str(tmp_path / "int.db"))
    yield fs
    fs.close()


def _tool(afs, aid, cid, name, status="success", err=None):
    afs.conn.execute(
        "INSERT INTO tool_calls (call_id, agent_id, tool_name, input, "
        "status, error_message, started_at) VALUES "
        "(?, ?, ?, '{}', ?, ?, strftime('%Y-%m-%dT%H:%M:%f','now'))",
        (cid, aid, name, status, err),
    )
    afs.conn.commit()


class TestQualityBridge:
    def test_partial_isa_records_quality_outcome(self, afs):
        aid = afs.spawn("a")
        sk = SkillStore(afs.conn)
        sid = sk.save(name="refund-endpoint", description="d",
                      template="t", source_agent_id=aid, tags=[])
        isa = IdealStateStore(afs.conn)
        isa_id = isa.create(aid, "Ship refund", "works", [
            {"criterion": "c1"}, {"criterion": "c2"},
            {"criterion": "c3"}, {"criterion": "c4"}, {"criterion": "c5"}])
        cs = isa.get(isa_id).criteria
        for c in cs[:4]:
            isa.mark(c.isc_id, "passed")
        isa.mark(cs[4].isc_id, "failed", failure_taxonomy="planning")

        q = isa.record_skill_outcome(isa_id, sid, agent_id=aid)
        assert q == pytest.approx(0.8)               # 4/5 passed
        # The skill use carries that graded quality (Track A path)
        eff, uses = _quality_signal_map(afs.conn, [sid])[sid]
        assert uses == 1
        assert eff == pytest.approx(0.8)

    def test_unverified_isa_records_nothing(self, afs):
        aid = afs.spawn("a")
        sk = SkillStore(afs.conn)
        sid = sk.save(name="s", description="d", template="t",
                      source_agent_id=aid, tags=[])
        isa = IdealStateStore(afs.conn)
        isa_id = isa.create(aid, "t", "s", [{"criterion": "c1"}])
        q = isa.record_skill_outcome(isa_id, sid, agent_id=aid)
        assert q is None
        n = afs.conn.execute(
            "SELECT COUNT(*) FROM skill_uses WHERE skill_id=?", (sid,)
        ).fetchone()[0]
        assert n == 0  # never invent signal


class TestTaxonomyGrouping:
    def test_failed_criteria_grouped_by_taxonomy(self, afs):
        a1 = afs.spawn("a1")
        a2 = afs.spawn("a2")
        isa = IdealStateStore(afs.conn)
        for aid in (a1, a2):
            iid = isa.create(aid, "t", "s",
                             [{"criterion": "x"}, {"criterion": "y"}])
            cs = isa.get(iid).criteria
            isa.mark(cs[0].isc_id, "failed", failure_taxonomy="planning")
            isa.mark(cs[1].isc_id, "failed", failure_taxonomy="memory")
        counts = isa.failed_criteria_by_taxonomy()
        assert counts["planning"] == 2
        assert counts["memory"] == 2

    def test_grouping_filters_by_agent(self, afs):
        a1 = afs.spawn("a1")
        a2 = afs.spawn("a2")
        isa = IdealStateStore(afs.conn)
        i1 = isa.create(a1, "t", "s", [{"criterion": "x"}])
        isa.mark(isa.get(i1).criteria[0].isc_id, "failed",
                 failure_taxonomy="action")
        i2 = isa.create(a2, "t", "s", [{"criterion": "y"}])
        isa.mark(isa.get(i2).criteria[0].isc_id, "failed",
                 failure_taxonomy="system")
        assert isa.failed_criteria_by_taxonomy(a1) == {"action": 1}


class TestCriticalStepLink:
    def test_failing_isc_links_to_critical_step(self, afs):
        aid = afs.spawn("a")
        SharedLog(afs.conn).intent(aid, "deploy unverified build")
        _tool(afs, aid, "c1", "run-deploy", status="error",
              err="rollout failed")
        isa = IdealStateStore(afs.conn)
        isa_id = isa.create(aid, "Deploy", "green", [{"criterion": "healthy"}])
        isc = isa.get(isa_id).criteria[0].isc_id
        isa.mark(isc, "failed", failure_taxonomy="planning")

        cs = isa.link_critical_step(isc, aid)
        assert cs is not None
        assert cs.isc_id == isc
        # persisted row carries the isc linkage
        row = afs.conn.execute(
            "SELECT isc_id FROM critical_steps WHERE agent_id=? "
            "ORDER BY cs_id DESC LIMIT 1", (aid,)
        ).fetchone()
        assert row[0] == isc
