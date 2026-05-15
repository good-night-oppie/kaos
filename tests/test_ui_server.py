"""Track D (v0.8.3) — war-room server endpoints.

The static HTML isn't unit-tested, but the new aggregation endpoints
are: floor shape, dossier aggregation, empty-project safety,
deterministic colour hash, intent-kanban lifecycle grouping, and a
regression guard on an existing endpoint.
"""

from __future__ import annotations

import pytest

from kaos import Kaos
from kaos.ui.server import agent_hue, create_app

try:
    from starlette.testclient import TestClient
    _HAVE_TESTCLIENT = True
except Exception:  # pragma: no cover
    _HAVE_TESTCLIENT = False

pytestmark = pytest.mark.skipif(
    not _HAVE_TESTCLIENT, reason="starlette TestClient unavailable"
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("KAOS_DREAM_AUTO", "0")
    p = tmp_path / "ui.db"
    fs = Kaos(db_path=str(p))
    a1 = fs.spawn("alpha")
    a2 = fs.spawn("beta")
    # skill + graded outcome for the dossier
    from kaos.skills import SkillStore
    sk = SkillStore(fs.conn)
    sid = sk.save(name="s1", description="d", template="t",
                  source_agent_id=a1, tags=[])
    sk.record_outcome(sid, success=True, quality=0.9, agent_id=a1)
    # memory + shared-log activity for the dossier / kanban
    from kaos.memory import MemoryStore
    MemoryStore(fs.conn).write(agent_id=a1, content="note one",
                               type="insight", key="k1")
    from kaos.shared_log import SharedLog
    log = SharedLog(fs.conn)
    iid = log.intent(a1, "do the thing")          # proposed
    log.vote(a2, iid, approve=True, reason="ok")  # → voting
    i2 = log.intent(a2, "second thing")
    log.vote(a1, i2, approve=True)
    log.decide(i2, a1)                            # → decided
    i3 = log.intent(a1, "third thing")
    log.commit(a1, i3, summary="done")            # → terminal
    fs.close()
    return str(p)


@pytest.fixture
def client(db):
    return TestClient(create_app()), db


class TestColourHash:
    def test_deterministic(self):
        assert agent_hue("agent-x") == agent_hue("agent-x")

    def test_in_range(self):
        for name in ("a", "agent-123", "", "δ-unicode"):
            assert 0 <= agent_hue(name) < 360

    def test_distinct_ids_usually_differ(self):
        hues = {agent_hue(f"agent-{i}") for i in range(20)}
        assert len(hues) > 10  # not a constant


class TestFloor:
    def test_shape(self, client):
        c, db = client
        r = c.get(f"/api/agents/floor?db={db}")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        a = data[0]
        for key in ("agent_id", "name", "status", "hue", "monogram",
                    "tool_calls", "tool_errors"):
            assert key in a
        assert 0 <= a["hue"] < 360
        assert len(a["monogram"]) <= 2

    def test_empty_project_returns_empty_not_error(self, tmp_path):
        empty = tmp_path / "empty.db"
        Kaos(db_path=str(empty)).close()
        c = TestClient(create_app())
        r = c.get(f"/api/agents/floor?db={empty}")
        assert r.status_code == 200
        assert r.json() == []


class TestDossier:
    def test_aggregates_right_agent(self, client):
        c, db = client
        floor = c.get(f"/api/agents/floor?db={db}").json()
        alpha = next(a for a in floor if a["name"] == "alpha")
        r = c.get(f"/api/agents/{alpha['agent_id']}/dossier?db={db}")
        assert r.status_code == 200
        d = r.json()
        assert d["agent"]["name"] == "alpha"
        assert any(s["name"] == "s1" for s in d["skills_used"])
        assert any(m["key"] == "k1" for m in d["memories"])
        assert d["hue"] == agent_hue(alpha["agent_id"])

    def test_unknown_agent_safe(self, client):
        c, db = client
        r = c.get(f"/api/agents/nope/dossier?db={db}")
        assert r.status_code == 200
        d = r.json()
        assert d["agent"] is None
        assert d["skills_used"] == []


class TestIntentKanban:
    def test_lifecycle_grouping(self, client):
        c, db = client
        r = c.get(f"/api/intents/kanban?db={db}")
        assert r.status_code == 200
        cols = r.json()
        assert set(cols) == {"proposed", "voting", "decided", "terminal"}
        # one intent each: voting (had a vote, no decision), decided,
        # terminal (committed)
        assert len(cols["voting"]) == 1
        assert len(cols["decided"]) == 1
        assert len(cols["terminal"]) == 1
        # every card carries a hue + vote count
        for col in cols.values():
            for card in col:
                assert "hue" in card and "votes" in card


class TestRegressionGuard:
    def test_existing_agents_endpoint_unchanged(self, client):
        c, db = client
        r = c.get(f"/api/agents?db={db}")
        assert r.status_code == 200
        assert len(r.json()) == 2  # the two spawned agents

    def test_floor_route_not_captured_as_agent_id(self, client):
        # "/api/agents/floor" must hit the floor endpoint, NOT
        # api_agent_detail with id="floor".
        c, db = client
        r = c.get(f"/api/agents/floor?db={db}")
        assert isinstance(r.json(), list)  # floor returns a list
