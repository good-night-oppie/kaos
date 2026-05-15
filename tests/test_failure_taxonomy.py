"""Track B1 (v0.8.3) — reasoning-class failure taxonomy.

Verifies every built-in heuristic gets a taxonomy class/subclass, the LLM
diagnoser emits + caches taxonomy, the classify_taxonomy() helper works,
and the failure_fingerprints write path persists the new columns.
"""

from __future__ import annotations

import pytest

from kaos import Kaos
from kaos.dream.diagnosis import (
    TAXONOMY_CLASSES,
    LLMDiagnoser,
    classify_taxonomy,
    diagnose,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def afs(tmp_path, monkeypatch):
    monkeypatch.setenv("KAOS_DREAM_AUTO", "0")
    fs = Kaos(db_path=str(tmp_path / "t.db"))
    yield fs
    fs.close()


class TestHeuristicTaxonomy:
    @pytest.mark.parametrize("error,expect_class,expect_sub", [
        ("Connection refused on localhost:8000", "system", "connectivity"),
        ("HTTP 429 Too Many Requests", "system", "throttling"),
        ("401 Unauthorized: invalid api key", "system", "auth"),
        ("No space left on device", "system", "resource_exhaustion"),
        ("Could not resolve host api.example.com", "system", "dns"),
        ("KeyError: 'user_id'", "action", "exception"),
        ("missing required argument: 'path'", "action", "malformed_call"),
    ])
    def test_heuristic_assigns_taxonomy(self, error, expect_class,
                                        expect_sub):
        d = diagnose("some-tool", error)
        assert d.taxonomy_class == expect_class
        assert d.taxonomy_subclass == expect_sub
        assert d.taxonomy_class in TAXONOMY_CLASSES

    def test_timeout_network_is_system(self):
        d = diagnose("http", "request timed out before response")
        assert d.taxonomy_class == "system"
        assert d.taxonomy_subclass == "timeout"

    def test_timeout_hang_is_action(self):
        # Must contain a timeout keyword (gate) AND a hang keyword (branch).
        d = diagnose("worker", "operation timed out: infinite loop / deadlock")
        assert d.taxonomy_class == "action"
        assert d.taxonomy_subclass == "hang"

    def test_unknown_fallthrough_taxonomy(self):
        d = diagnose("x", "totally novel non-matching failure zzz-quux")
        assert d.taxonomy_class == "unknown"

    def test_to_dict_includes_taxonomy(self):
        d = diagnose("http", "Connection refused on localhost:9")
        dd = d.to_dict()
        assert dd["taxonomy_class"] == "system"
        assert dd["taxonomy_subclass"] == "connectivity"


class TestClassifyTaxonomyHelper:
    def test_returns_pair(self):
        cls, sub = classify_taxonomy("t", "KeyError: 'x'")
        assert cls == "action"
        assert sub == "exception"

    def test_unknown_pair(self):
        cls, sub = classify_taxonomy("t", "weird zzz-quux")
        assert cls == "unknown"


class TestLLMDiagnoserTaxonomy:
    def test_llm_emits_and_caches_taxonomy(self, afs):
        calls = []

        def fake(prompt):
            calls.append(prompt)
            return ('{"category":"code","taxonomy_class":"planning",'
                    '"taxonomy_subclass":"plan_loop","root_cause":"r",'
                    '"suggested_action":"s","confidence":0.7}')

        d = LLMDiagnoser(call_fn=fake, conn=afs.conn)
        first = d.try_diagnose("agent-tool", "novel reasoning failure abc",
                               {})
        assert first.taxonomy_class == "planning"
        assert first.taxonomy_subclass == "plan_loop"
        assert first.method == "llm"

        # Cache hit must carry taxonomy too, no second call
        second = d.try_diagnose("agent-tool", "novel reasoning failure abc",
                                {})
        assert second.method == "llm-cached"
        assert second.taxonomy_class == "planning"
        assert second.taxonomy_subclass == "plan_loop"
        assert len(calls) == 1

    def test_invalid_taxonomy_normalised_to_unknown(self, afs):
        d = LLMDiagnoser(
            call_fn=lambda p: '{"category":"code","taxonomy_class":"banana",'
                              '"root_cause":"r","confidence":0.5}',
            conn=afs.conn,
        )
        res = d.try_diagnose("t", "another novel failure xyz", {})
        assert res.taxonomy_class == "unknown"

    def test_diagnose_does_not_override_llm_taxonomy(self, afs):
        # The LLM set planning; the post-stamp must NOT clobber it with a
        # heuristic-map value (the llm diagnoser isn't in _HEURISTIC_TAXONOMY
        # anyway, but assert the contract explicitly).
        d = LLMDiagnoser(
            call_fn=lambda p: '{"category":"code","taxonomy_class":"reflection",'
                              '"taxonomy_subclass":"bad_self_check",'
                              '"root_cause":"r","confidence":0.8}',
            conn=afs.conn,
        )
        res = diagnose("custom", "unmatched failure qqq", llm_fallback=d)
        assert res.taxonomy_class == "reflection"
        assert res.taxonomy_subclass == "bad_self_check"


class TestFingerprintPersistence:
    def test_taxonomy_written_to_failure_fingerprints(self, afs):
        # Drive a real failure through an agent so auto.py diagnoses it.
        aid = afs.spawn("agent-x")
        afs.conn.execute(
            "INSERT INTO tool_calls (call_id, agent_id, tool_name, input, "
            "status, error_message) VALUES "
            "('c1', ?, 'http', '{}', 'error', "
            "'Connection refused on localhost:8000')",
            (aid,),
        )
        afs.conn.commit()
        from kaos.dream.auto import record_failure_fingerprint
        fp_id = record_failure_fingerprint(afs.conn, aid)
        assert fp_id is not None
        row = afs.conn.execute(
            "SELECT category, taxonomy_class, taxonomy_subclass "
            "FROM failure_fingerprints WHERE fp_id = ?", (fp_id,),
        ).fetchone()
        assert row[0] == "infra"            # execution-class unchanged
        assert row[1] == "system"           # reasoning-class taxonomy
        assert row[2] == "connectivity"
