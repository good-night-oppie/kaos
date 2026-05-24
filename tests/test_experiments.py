"""PR-3 (v0.9) — ExperimentStore + schema v9 (additive experiments table).

Covers: schema migration v8->v9 is non-destructive, ExperimentStore
write/read/list/get/compare round-trips, verdict-prefix filtering, and
auto-fill of git_sha vs the empty-string suppression hook.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from kaos.experiments import ExperimentStore
from kaos.schema import SCHEMA_VERSION, init_schema


# ─────────────────────────────────────────────────────────────────────
# schema migration
# ─────────────────────────────────────────────────────────────────────


class TestSchemaV9:
    def test_fresh_db_has_experiments_table(self, tmp_path: Path):
        db = tmp_path / "k.db"
        conn = sqlite3.connect(db)
        init_schema(conn)
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(experiments)").fetchall()}
        assert {"exp_id", "name", "family", "git_sha", "lock_sha256",
                "started_at", "finished_at", "duration_ms", "verdict",
                "judge_kappa", "arms_json", "gates_json", "metadata",
                "results_path"}.issubset(cols)
        v = conn.execute(
            "SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert v == SCHEMA_VERSION == 9
        conn.close()

    def test_v8_db_migrates_to_v9_non_destructively(self, tmp_path: Path):
        db = tmp_path / "k.db"
        conn = sqlite3.connect(db)
        init_schema(conn)
        # Insert a row in some pre-existing table so we can prove the
        # migration preserves it.
        conn.execute(
            "INSERT INTO agents (agent_id, name) VALUES (?, ?)",
            ("a1", "preexisting"),
        )
        conn.commit()
        conn.close()

        # Simulate an "old" DB by rewinding schema_version to 8 and
        # dropping the experiments table. We replace the version row
        # (rather than delete it) so init_schema takes the incremental
        # _apply_migrations path, not the fresh-DB bulk path.
        conn = sqlite3.connect(db)
        conn.execute("DROP TABLE experiments")
        conn.execute("DELETE FROM schema_version")
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (8)"
        )
        conn.commit()
        conn.close()

        # Re-init: must run only v9 migration and keep the agents row.
        conn = sqlite3.connect(db)
        init_schema(conn)
        row = conn.execute(
            "SELECT name FROM agents WHERE agent_id='a1'").fetchone()
        assert row[0] == "preexisting"
        # experiments table is back
        conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# ExperimentStore I/O
# ─────────────────────────────────────────────────────────────────────


class TestExperimentStore:
    def test_log_and_get_round_trip(self, tmp_path: Path):
        db = tmp_path / "k.db"
        with ExperimentStore(db) as store:
            exp_id = store.log_run(
                name="synthesis-consolidation",
                family="probe",
                verdict="REJECT: kill gate(s) failed: G1",
                judge_kappa=1.0,
                lock_sha256="09310794" * 8,
                git_sha="",  # suppress auto-fill
                arms={"FULL": {"acc_hard": 0.364}},
                gates=[{"gate": "G1", "passed": False, "kill": True}],
                metadata={"workload_n": 1620},
                results_path="demo_synthesis_consolidation_bench/results.json",
            )
            exp = store.get(exp_id)

        assert exp is not None
        assert exp.name == "synthesis-consolidation"
        assert exp.family == "probe"
        assert exp.verdict.startswith("REJECT")
        assert exp.judge_kappa == 1.0
        assert exp.arms == {"FULL": {"acc_hard": 0.364}}
        assert exp.gates[0]["gate"] == "G1"
        assert exp.metadata["workload_n"] == 1620
        assert exp.git_sha is None  # suppressed

    def test_git_sha_autofill_when_omitted(self, tmp_path: Path):
        db = tmp_path / "k.db"
        with ExperimentStore(db) as store:
            exp_id = store.log_run(name="t", family="probe")
            exp = store.get(exp_id)
        # In a real git checkout this is a hex sha; outside one it's
        # None. Either way the field exists and is not the suppression
        # sentinel "".
        assert exp.git_sha is None or len(exp.git_sha) >= 7

    def test_list_filters_by_name_and_verdict_prefix(self, tmp_path: Path):
        db = tmp_path / "k.db"
        with ExperimentStore(db) as store:
            store.log_run(name="probe-A", family="probe",
                          verdict="ACCEPT", git_sha="")
            store.log_run(name="probe-A", family="probe",
                          verdict="REJECT: kill G1", git_sha="")
            store.log_run(name="probe-B", family="probe",
                          verdict="ACCEPT", git_sha="")

            assert len(store.list(name="probe-A")) == 2
            assert len(store.list(name="probe-B")) == 1
            assert len(store.list(verdict_prefix="REJECT")) == 1
            assert len(store.list(verdict_prefix="ACCEPT")) == 2

    def test_compare_reports_changed_fields_only(self, tmp_path: Path):
        db = tmp_path / "k.db"
        with ExperimentStore(db) as store:
            a = store.log_run(name="x", family="probe",
                              verdict="ACCEPT", git_sha="sha-1",
                              judge_kappa=1.0,
                              arms={"FULL": {"acc": 0.8}})
            b = store.log_run(name="x", family="probe",
                              verdict="REJECT: kill G1", git_sha="sha-2",
                              judge_kappa=1.0,
                              arms={"FULL": {"acc": 0.4}})
            diff = store.compare(a, b)

        assert "verdict" in diff["changes"]
        assert diff["changes"]["verdict"][0] == "ACCEPT"
        assert diff["changes"]["verdict"][1].startswith("REJECT")
        assert "git_sha" in diff["changes"]
        assert "arms" in diff["changes"]
        # name / family unchanged
        assert "name" not in diff["changes"]
        assert "family" not in diff["changes"]

    def test_compare_missing_exp_raises(self, tmp_path: Path):
        db = tmp_path / "k.db"
        with ExperimentStore(db) as store:
            a = store.log_run(name="x", git_sha="")
            with pytest.raises(ValueError):
                store.compare(a, 99999)

    def test_list_limit_and_ordering(self, tmp_path: Path):
        db = tmp_path / "k.db"
        with ExperimentStore(db) as store:
            ids = []
            for i in range(5):
                ids.append(store.log_run(name=f"r{i}", git_sha=""))
            rows = store.list(limit=3)
        assert len(rows) == 3
        # newest first
        assert rows[0].exp_id == ids[-1]


# ─────────────────────────────────────────────────────────────────────
# integration: ExperimentStore + harness verdict
# ─────────────────────────────────────────────────────────────────────


class TestExperimentStoreWithHarness:
    def test_logging_a_harness_verdict_round_trips(self, tmp_path: Path):
        from kaos.eval.harness import (
            ArmResults, GateOutcome, QueryResult, bootstrap_diff_ci,
            compute_verdict,
        )

        b0 = ArmResults(arm="B0", per_query=[
            QueryResult(f"q{i}", "hard", i < 40) for i in range(100)
        ])
        full = ArmResults(arm="FULL", per_query=[
            QueryResult(f"q{i}", "hard", i < 70) for i in range(100)
        ])
        md, lo, hi = bootstrap_diff_ci(
            full.labels({"hard"}), b0.labels({"hard"}), iters=300)
        gates = [GateOutcome(
            "G1", "beats", (full.acc({"hard"}) - b0.acc({"hard"})) >= 0.10
            and lo > 0, True,
            f"diff={full.acc({'hard'})-b0.acc({'hard'}):+.3f} lo={lo:+.3f}",
        )]
        verdict = compute_verdict(gates, judge_kappa=1.0)

        db = tmp_path / "k.db"
        with ExperimentStore(db) as store:
            exp_id = store.log_run(
                name="smoke-probe", family="probe",
                verdict=verdict, judge_kappa=1.0, git_sha="",
                arms={"B0": {"acc": b0.acc({"hard"})},
                      "FULL": {"acc": full.acc({"hard"})}},
                gates=[{"gate": g.gate, "passed": g.passed,
                        "kill": g.kill, "detail": g.detail}
                       for g in gates],
            )
            exp = store.get(exp_id)

        assert exp.verdict == "ACCEPT"
        assert exp.arms["FULL"]["acc"] == pytest.approx(0.70)
        assert exp.gates[0]["gate"] == "G1"
