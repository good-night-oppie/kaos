"""Schema migration tests — focus on the v0.8.3 consolidated v8 migration.

Covers:
- fresh DB lands at SCHEMA_VERSION with every v8 object present
- an existing v7 DB upgrades in place to v8 without data loss
- the migration is idempotent (re-running init_schema is a no-op)
- new columns are nullable / defaulted so old rows are untouched
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kaos import Kaos
from kaos.schema import SCHEMA_VERSION, init_schema


def _cols(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _tables(conn):
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


@pytest.fixture
def fresh(tmp_path):
    k = Kaos(db_path=str(tmp_path / "fresh.db"))
    yield k
    k.close()


class TestFreshDatabase:
    def test_lands_at_current_version(self, fresh):
        v = fresh.conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0]
        assert v == SCHEMA_VERSION == 8

    def test_track_a_column_present(self, fresh):
        assert "quality" in _cols(fresh.conn, "skill_uses")

    def test_track_b1_columns_present(self, fresh):
        c = _cols(fresh.conn, "failure_fingerprints")
        assert "taxonomy_class" in c
        assert "taxonomy_subclass" in c

    def test_track_b2_table_present(self, fresh):
        assert "critical_steps" in _tables(fresh.conn)
        c = _cols(fresh.conn, "critical_steps")
        assert {"agent_id", "log_position", "isc_id", "method"} <= c

    def test_track_b3_tables_present(self, fresh):
        t = _tables(fresh.conn)
        assert "ideal_states" in t
        assert "ideal_state_criteria" in t

    def test_track_c_columns_present(self, fresh):
        c = _cols(fresh.conn, "shared_log")
        assert "vote_confidence" in c
        assert "decide_mode" in c


class TestUpgradeFromV7:
    """Simulate a real v7 database and confirm it upgrades to v8 in place."""

    # v8 added these columns to existing tables; a real v7 DB has none of them.
    _V8_DROPPED = {
        "skill_uses": ["quality"],
        "failure_fingerprints": ["taxonomy_class", "taxonomy_subclass"],
        "shared_log": ["vote_confidence", "decide_mode"],
        "llm_diagnosis_cache": ["taxonomy_class", "taxonomy_subclass"],
    }

    def _strip_v8_columns(self, conn, table: str, drop: list[str]) -> None:
        """Rebuild `table` keeping every column except `drop`, faithfully
        simulating a pre-v8 schema regardless of what earlier migrations
        added."""
        all_cols = [r[1] for r in conn.execute(
            f"PRAGMA table_info({table})").fetchall()]
        keep = [c for c in all_cols if c not in drop]
        keep_csv = ", ".join(keep)
        conn.executescript(
            f"""
            CREATE TABLE _mig_tmp AS SELECT {keep_csv} FROM {table};
            DROP TABLE {table};
            ALTER TABLE _mig_tmp RENAME TO {table};
            """
        )

    def _make_v7_db(self, path: Path) -> None:
        # Build a current-schema DB, then rewind to a faithful v7 state:
        # drop v8 tables, strip v8 columns, reset the version pointer.
        k = Kaos(db_path=str(path))
        aid = k.spawn("legacy-agent")
        from kaos.skills import SkillStore
        sk = SkillStore(k.conn)
        sid = sk.save(name="legacy-skill", description="d", template="t",
                      source_agent_id=aid, tags=[])
        sk.record_outcome(sid, success=True, agent_id=aid)
        k.close()

        conn = sqlite3.connect(str(path))
        conn.executescript(
            """
            DROP TABLE IF EXISTS ideal_state_criteria;
            DROP TABLE IF EXISTS ideal_states;
            DROP TABLE IF EXISTS critical_steps;
            """
        )
        for table, drop in self._V8_DROPPED.items():
            self._strip_v8_columns(conn, table, drop)
        conn.executescript(
            """
            DELETE FROM schema_version;
            INSERT INTO schema_version (version) VALUES (7);
            """
        )
        conn.commit()
        conn.close()

    def test_v7_db_upgrades_in_place(self, tmp_path):
        path = tmp_path / "legacy.db"
        self._make_v7_db(path)

        # Re-opening through init_schema must migrate 7 -> 8
        conn = sqlite3.connect(str(path))
        init_schema(conn)
        v = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert v == 8
        assert "quality" in _cols(conn, "skill_uses")
        assert "critical_steps" in _tables(conn)
        assert "ideal_states" in _tables(conn)
        # Legacy data survived
        n = conn.execute("SELECT COUNT(*) FROM skill_uses").fetchone()[0]
        assert n == 1
        conn.close()


class TestIdempotence:
    def test_reinit_is_noop(self, tmp_path):
        path = str(tmp_path / "idem.db")
        k = Kaos(db_path=path)
        k.close()
        conn = sqlite3.connect(path)
        init_schema(conn)  # second call
        init_schema(conn)  # third call
        v = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert v == SCHEMA_VERSION
        # exactly one version row per applied version (no dup stamping)
        rows = conn.execute(
            "SELECT version, COUNT(*) FROM schema_version GROUP BY version "
            "HAVING COUNT(*) > 1"
        ).fetchall()
        assert rows == []
        conn.close()


class TestOldRowsUntouched:
    def test_existing_skill_use_has_null_quality(self, tmp_path):
        k = Kaos(db_path=str(tmp_path / "q.db"))
        aid = k.spawn("a")
        from kaos.skills import SkillStore
        sk = SkillStore(k.conn)
        sid = sk.save(name="s", description="d", template="t",
                      source_agent_id=aid, tags=[])
        sk.record_outcome(sid, success=True, agent_id=aid)
        row = k.conn.execute(
            "SELECT quality FROM skill_uses WHERE skill_id = ?", (sid,)
        ).fetchone()
        assert row[0] is None  # binary record → quality stays NULL
        k.close()
