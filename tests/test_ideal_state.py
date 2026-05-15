"""Track B3 (v0.8.3) — Ideal State Artifacts (ISA/ISC).

Covers create / mark / finalize / list, the quality bridge, taxonomy
linkage, and migration safety.
"""

from __future__ import annotations

import pytest

from kaos import Kaos
from kaos.ideal_state import IdealStateStore


@pytest.fixture
def afs(tmp_path, monkeypatch):
    monkeypatch.setenv("KAOS_DREAM_AUTO", "0")
    fs = Kaos(db_path=str(tmp_path / "isa.db"))
    yield fs
    fs.close()


@pytest.fixture
def store(afs):
    return IdealStateStore(afs.conn)


@pytest.fixture
def agent(afs):
    return afs.spawn("a")


class TestCreate:
    def test_create_returns_id_and_criteria(self, store, agent):
        isa_id = store.create(agent, "Ship X", "X works", [
            {"criterion": "returns 201"},
            {"criterion": "idempotent", "verification": "replay key"},
        ])
        isa = store.get(isa_id)
        assert isa.title == "Ship X"
        assert len(isa.criteria) == 2
        assert isa.criteria[1].verification == "replay key"
        assert isa.overall_status == "pending"

    def test_empty_criteria_raises(self, store, agent):
        with pytest.raises(ValueError, match="at least one criterion"):
            store.create(agent, "t", "s", [])


class TestMark:
    def test_mark_pass_fail_skip(self, store, agent):
        isa_id = store.create(agent, "t", "s", [
            {"criterion": "a"}, {"criterion": "b"}, {"criterion": "c"}])
        isa = store.get(isa_id)
        a, b, c = [x.isc_id for x in isa.criteria]
        store.mark(a, "passed")
        store.mark(b, "failed", failure_taxonomy="planning", note="bad plan")
        store.mark(c, "skipped")
        isa = store.get(isa_id)
        by_id = {x.isc_id: x for x in isa.criteria}
        assert by_id[a].status == "passed"
        assert by_id[b].status == "failed"
        assert by_id[b].failure_taxonomy == "planning"
        assert by_id[b].failure_note == "bad plan"
        assert by_id[c].status == "skipped"

    def test_invalid_status_raises(self, store, agent):
        isa_id = store.create(agent, "t", "s", [{"criterion": "a"}])
        isc = store.get(isa_id).criteria[0].isc_id
        with pytest.raises(ValueError, match="status must be one of"):
            store.mark(isc, "maybe")

    def test_taxonomy_only_kept_on_failure(self, store, agent):
        isa_id = store.create(agent, "t", "s", [{"criterion": "a"}])
        isc = store.get(isa_id).criteria[0].isc_id
        store.mark(isc, "passed", failure_taxonomy="planning")
        assert store.get(isa_id).criteria[0].failure_taxonomy is None


class TestFinalize:
    def test_all_passed(self, store, agent):
        isa_id = store.create(agent, "t", "s",
                              [{"criterion": "a"}, {"criterion": "b"}])
        for c in store.get(isa_id).criteria:
            store.mark(c.isc_id, "passed")
        assert store.finalize(isa_id) == "passed"
        assert store.get(isa_id).overall_status == "passed"
        assert store.get(isa_id).completed_at is not None

    def test_any_failed(self, store, agent):
        isa_id = store.create(agent, "t", "s",
                              [{"criterion": "a"}, {"criterion": "b"}])
        cs = store.get(isa_id).criteria
        store.mark(cs[0].isc_id, "passed")
        store.mark(cs[1].isc_id, "failed")
        assert store.finalize(isa_id) == "failed"

    def test_pending_when_unverified(self, store, agent):
        isa_id = store.create(agent, "t", "s",
                              [{"criterion": "a"}, {"criterion": "b"}])
        store.mark(store.get(isa_id).criteria[0].isc_id, "passed")
        assert store.finalize(isa_id) == "pending"

    def test_passed_with_skips(self, store, agent):
        isa_id = store.create(agent, "t", "s",
                              [{"criterion": "a"}, {"criterion": "b"}])
        cs = store.get(isa_id).criteria
        store.mark(cs[0].isc_id, "passed")
        store.mark(cs[1].isc_id, "skipped")
        assert store.finalize(isa_id) == "passed"

    def test_abandon(self, store, agent):
        isa_id = store.create(agent, "t", "s", [{"criterion": "a"}])
        store.abandon(isa_id)
        assert store.get(isa_id).overall_status == "abandoned"


class TestQualityBridge:
    def test_quality_none_before_verification(self, store, agent):
        isa_id = store.create(agent, "t", "s", [{"criterion": "a"}])
        assert store.get(isa_id).quality is None

    def test_quality_fraction(self, store, agent):
        isa_id = store.create(agent, "t", "s", [
            {"criterion": "a"}, {"criterion": "b"},
            {"criterion": "c"}, {"criterion": "d"}])
        cs = store.get(isa_id).criteria
        store.mark(cs[0].isc_id, "passed")
        store.mark(cs[1].isc_id, "passed")
        store.mark(cs[2].isc_id, "passed")
        store.mark(cs[3].isc_id, "failed")
        # 3 passed of 4 scored = 0.75
        assert store.get(isa_id).quality == pytest.approx(0.75)

    def test_skipped_excluded_from_quality(self, store, agent):
        isa_id = store.create(agent, "t", "s", [
            {"criterion": "a"}, {"criterion": "b"}, {"criterion": "c"}])
        cs = store.get(isa_id).criteria
        store.mark(cs[0].isc_id, "passed")
        store.mark(cs[1].isc_id, "failed")
        store.mark(cs[2].isc_id, "skipped")
        # 1 passed of 2 scored (skip excluded) = 0.5
        assert store.get(isa_id).quality == pytest.approx(0.5)


class TestListing:
    def test_list_open_excludes_finalized(self, store, agent):
        a = store.create(agent, "open one", "s", [{"criterion": "x"}])
        b = store.create(agent, "done one", "s", [{"criterion": "y"}])
        store.mark(store.get(b).criteria[0].isc_id, "passed")
        store.finalize(b)
        open_ids = [i.isa_id for i in store.list_open()]
        assert a in open_ids
        assert b not in open_ids

    def test_list_filters_by_agent(self, store, afs):
        a1 = afs.spawn("a1")
        a2 = afs.spawn("a2")
        store.create(a1, "t1", "s", [{"criterion": "x"}])
        store.create(a2, "t2", "s", [{"criterion": "y"}])
        assert len(store.list_all(a1)) == 1
        assert len(store.list_all()) == 2

    def test_get_unknown_returns_none(self, store):
        assert store.get(99999) is None
