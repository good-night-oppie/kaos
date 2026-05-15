"""Ideal State Artifacts (Track B3, v0.8.3).

Borrowed from PAI's ISA/ISC pattern. An Ideal State Artifact (ISA) is the
"what done looks like" doc for a non-trivial agent task — the general
equivalent of a software PRD. It decomposes into discrete Ideal State
Criteria (ISCs) that double as verification items.

Why this matters for KAOS: per-criterion pass/fail is a far finer
plasticity signal than one binary task outcome. An ISA with 4/5 ISCs
passed maps to a quality of 0.8 on the associated skill use (Track A),
and a failed ISC can carry a reasoning-class taxonomy (Track B1) and
link to the trajectory step where it went wrong (Track B2). The three
tracks compose here.

Schema lives in the v8 migration: ``ideal_states`` + ``ideal_state_criteria``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

ISC_STATUSES = ("pending", "passed", "failed", "skipped")
ISA_STATUSES = ("pending", "passed", "failed", "abandoned")


@dataclass
class Criterion:
    isc_id: int
    isa_id: int
    criterion: str
    verification: str | None
    status: str
    failure_taxonomy: str | None
    failure_note: str | None
    verified_at: str | None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Criterion":
        return cls(
            isc_id=r["isc_id"], isa_id=r["isa_id"],
            criterion=r["criterion"], verification=r["verification"],
            status=r["status"], failure_taxonomy=r["failure_taxonomy"],
            failure_note=r["failure_note"], verified_at=r["verified_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "isc_id": self.isc_id, "isa_id": self.isa_id,
            "criterion": self.criterion, "verification": self.verification,
            "status": self.status, "failure_taxonomy": self.failure_taxonomy,
            "failure_note": self.failure_note, "verified_at": self.verified_at,
        }


@dataclass
class IdealState:
    isa_id: int
    agent_id: str
    title: str
    summary: str
    created_at: str
    completed_at: str | None
    overall_status: str | None
    criteria: list[Criterion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "isa_id": self.isa_id, "agent_id": self.agent_id,
            "title": self.title, "summary": self.summary,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "overall_status": self.overall_status,
            "criteria": [c.to_dict() for c in self.criteria],
        }

    @property
    def quality(self) -> float | None:
        """Fraction of non-skipped criteria that passed, in [0,1]. None if
        nothing has been verified yet. This is the value to feed
        SkillStore.record_outcome(quality=...) — the Track A bridge."""
        scored = [c for c in self.criteria if c.status in ("passed", "failed")]
        if not scored:
            return None
        return sum(1 for c in scored if c.status == "passed") / len(scored)


class IdealStateStore:
    """Create, verify, and finalize Ideal State Artifacts.

    Usage::

        isa = IdealStateStore(kaos.conn)
        aid = isa.create("agent-1", "Ship the refund endpoint",
            "A working POST /refunds with idempotency",
            criteria=[
                {"criterion": "endpoint returns 201 on valid input"},
                {"criterion": "duplicate request is idempotent",
                 "verification": "replay the same Idempotency-Key"},
            ])
        # ... agent works ...
        isa.mark(isc_id=1, status="passed")
        isa.mark(isc_id=2, status="failed",
                 failure_taxonomy="planning", note="no idempotency store")
        overall = isa.finalize(aid)        # -> 'failed'
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    # ── Create ────────────────────────────────────────────────────

    def create(
        self,
        agent_id: str,
        title: str,
        summary: str,
        criteria: list[dict[str, Any]],
    ) -> int:
        """Create an ISA with its criteria. Each criterion dict needs a
        ``criterion`` key and may carry an optional ``verification``.
        Returns the new isa_id."""
        if not criteria:
            raise ValueError("an ISA needs at least one criterion")
        cur = self._conn.execute(
            "INSERT INTO ideal_states (agent_id, title, summary, "
            "overall_status) VALUES (?, ?, ?, 'pending')",
            (agent_id, title, summary),
        )
        isa_id = cur.lastrowid
        self._conn.executemany(
            "INSERT INTO ideal_state_criteria (isa_id, criterion, "
            "verification) VALUES (?, ?, ?)",
            [
                (isa_id, c["criterion"], c.get("verification"))
                for c in criteria
            ],
        )
        self._conn.commit()
        return isa_id  # type: ignore[return-value]

    # ── Verify ────────────────────────────────────────────────────

    def mark(
        self,
        isc_id: int,
        status: str,
        *,
        failure_taxonomy: str | None = None,
        note: str | None = None,
    ) -> None:
        """Mark one criterion passed / failed / skipped.

        ``failure_taxonomy`` should be a Track B1 taxonomy_class when
        ``status='failed'`` so consolidation can group failed ISCs by
        reasoning-class.
        """
        if status not in ISC_STATUSES:
            raise ValueError(
                f"status must be one of {ISC_STATUSES!r}, got {status!r}"
            )
        self._conn.execute(
            "UPDATE ideal_state_criteria "
            "SET status = ?, failure_taxonomy = ?, failure_note = ?, "
            "verified_at = strftime('%Y-%m-%dT%H:%M:%f','now') "
            "WHERE isc_id = ?",
            (status,
             failure_taxonomy if status == "failed" else None,
             note, isc_id),
        )
        self._conn.commit()

    # ── Finalize ──────────────────────────────────────────────────

    def finalize(self, isa_id: int) -> str:
        """Compute and persist the overall status:

        - any criterion 'failed'           -> 'failed'
        - all non-skipped criteria 'passed' -> 'passed'
        - otherwise (still pending ones)    -> 'pending' (unchanged)

        Returns the resulting overall_status.
        """
        rows = self._conn.execute(
            "SELECT status FROM ideal_state_criteria WHERE isa_id = ?",
            (isa_id,),
        ).fetchall()
        statuses = [r["status"] for r in rows]
        if not statuses:
            overall = "pending"
        elif any(s == "failed" for s in statuses):
            overall = "failed"
        elif all(s in ("passed", "skipped") for s in statuses):
            overall = "passed"
        else:
            overall = "pending"

        if overall in ("passed", "failed"):
            self._conn.execute(
                "UPDATE ideal_states SET overall_status = ?, "
                "completed_at = strftime('%Y-%m-%dT%H:%M:%f','now') "
                "WHERE isa_id = ?",
                (overall, isa_id),
            )
        else:
            self._conn.execute(
                "UPDATE ideal_states SET overall_status = ? "
                "WHERE isa_id = ?",
                (overall, isa_id),
            )
        self._conn.commit()
        return overall

    def abandon(self, isa_id: int) -> None:
        """Mark an ISA abandoned (work stopped without finishing)."""
        self._conn.execute(
            "UPDATE ideal_states SET overall_status = 'abandoned', "
            "completed_at = strftime('%Y-%m-%dT%H:%M:%f','now') "
            "WHERE isa_id = ?",
            (isa_id,),
        )
        self._conn.commit()

    # ── Read ──────────────────────────────────────────────────────

    def get(self, isa_id: int) -> IdealState | None:
        row = self._conn.execute(
            "SELECT * FROM ideal_states WHERE isa_id = ?", (isa_id,)
        ).fetchone()
        if row is None:
            return None
        isa = IdealState(
            isa_id=row["isa_id"], agent_id=row["agent_id"],
            title=row["title"], summary=row["summary"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            overall_status=row["overall_status"],
        )
        crows = self._conn.execute(
            "SELECT * FROM ideal_state_criteria WHERE isa_id = ? "
            "ORDER BY isc_id", (isa_id,),
        ).fetchall()
        isa.criteria = [Criterion.from_row(r) for r in crows]
        return isa

    def list_open(
        self, agent_id: str | None = None, limit: int = 50
    ) -> list[IdealState]:
        """ISAs not yet passed/failed/abandoned (overall_status='pending'
        or NULL), most recent first."""
        params: list[Any] = []
        where = "WHERE (overall_status IS NULL OR overall_status = 'pending')"
        if agent_id:
            where += " AND agent_id = ?"
            params.append(agent_id)
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT isa_id FROM ideal_states {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self.get(r["isa_id"]) for r in rows]  # type: ignore[misc]

    # ── Track composition bridges ─────────────────────────────────

    def record_skill_outcome(
        self,
        isa_id: int,
        skill_id: int,
        *,
        agent_id: str | None = None,
    ) -> float | None:
        """Track A bridge: finalize the ISA and record its quality as the
        outcome of ``skill_id``. An ISA with 4/5 criteria passed records
        quality=0.8 on the skill use instead of a binary coin flip.

        Returns the quality written, or None if nothing was verified
        (in which case no outcome is recorded — we never invent signal).
        """
        self.finalize(isa_id)
        isa = self.get(isa_id)
        if isa is None:
            return None
        q = isa.quality
        if q is None:
            return None
        from kaos.skills import SkillStore
        SkillStore(self._conn).record_outcome(
            skill_id, success=(q >= 0.5), quality=q, agent_id=agent_id,
        )
        return q

    def failed_criteria_by_taxonomy(
        self, agent_id: str | None = None
    ) -> dict[str, int]:
        """Track B1 bridge: count failed ISCs grouped by their
        reasoning-class taxonomy, so consolidation/narrative can surface
        systemic patterns ("3 agents failed `planning` criteria")."""
        params: list[Any] = []
        join = ""
        if agent_id:
            join = ("JOIN ideal_states s ON s.isa_id = c.isa_id "
                    "AND s.agent_id = ?")
            params.append(agent_id)
        rows = self._conn.execute(
            f"SELECT COALESCE(c.failure_taxonomy,'unspecified') AS tx, "
            f"COUNT(*) AS n FROM ideal_state_criteria c {join} "
            f"WHERE c.status = 'failed' GROUP BY tx ORDER BY n DESC",
            params,
        ).fetchall()
        return {r["tx"]: r["n"] for r in rows}

    def link_critical_step(
        self, isc_id: int, agent_id: str,
        *, llm_call_fn=None,
    ):
        """Track B2 bridge: localize the owning agent's trajectory and tag
        the resulting critical_steps row with this failed criterion, so a
        failing ISC traces straight to the step it went wrong. Returns the
        CriticalStep or None."""
        from kaos.dream.phases.localize import localize
        return localize(self._conn, agent_id, isc_id=isc_id,
                        llm_call_fn=llm_call_fn)

    def list_all(
        self, agent_id: str | None = None, limit: int = 50
    ) -> list[IdealState]:
        params: list[Any] = []
        where = ""
        if agent_id:
            where = "WHERE agent_id = ?"
            params.append(agent_id)
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT isa_id FROM ideal_states {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self.get(r["isa_id"]) for r in rows]  # type: ignore[misc]
