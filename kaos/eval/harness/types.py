"""Per-query / per-arm / per-gate result containers.

These are the atomic units a probe produces. They are deliberately
flat dataclasses (no methods that touch I/O, no implicit aggregation
over arms): the harness composes them; domain code populates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """One judged query against one arm.

    qclass is a free string — the probe chooses its own class taxonomy
    (e.g. ``verbatim_recall``, ``compositional_multihop``, ``malformed_tool_call``).
    correct is the BLIND-JUDGED boolean; helper booleans live in extras.
    """
    qid: str
    qclass: str
    correct: bool
    split: str = "in_dist"
    extras: dict = field(default_factory=dict)


@dataclass
class ArmResults:
    arm: str
    per_query: list[QueryResult] = field(default_factory=list)

    def acc(
        self,
        classes: set[str] | None = None,
        split: str = "in_dist",
    ) -> float:
        rows = [
            q for q in self.per_query
            if q.split == split and (classes is None or q.qclass in classes)
        ]
        if not rows:
            return 0.0
        return sum(q.correct for q in rows) / len(rows)

    def n(
        self,
        classes: set[str] | None = None,
        split: str = "in_dist",
    ) -> int:
        return sum(
            1 for q in self.per_query
            if q.split == split and (classes is None or q.qclass in classes)
        )

    def labels(
        self,
        classes: set[str] | None = None,
        split: str = "in_dist",
    ) -> list[int]:
        return [
            int(q.correct) for q in self.per_query
            if q.split == split and (classes is None or q.qclass in classes)
        ]


@dataclass
class GateOutcome:
    """Outcome of one named gate after the gate function has run.

    kill=True means this gate is a kill-gate: any failure REJECTS the
    binding verdict regardless of how the other gates land. kill=False
    is a sanity/diagnostic gate; failure VOIDS the run (results
    untrustworthy, not a feature verdict).
    """
    gate: str
    name: str
    passed: bool
    kill: bool
    detail: str
