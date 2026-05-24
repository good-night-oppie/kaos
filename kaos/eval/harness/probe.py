"""Probe base class — the lifecycle every KAOS probe implements.

A probe is a concrete mechanism evaluation with five obligations:

  1. ``arms()``         — return the set of arms it ships, including
                          mandatory ablations B0/B1/L1 etc. The lock
                          enumerates which arms are required.
  2. ``workload()``     — build the (deterministic, seeded) workload.
  3. ``queries()``      — frozen pre-registered queries.
  4. ``gates(arms_in)`` — compute the domain-specific kill / sanity
                          gates from arm results.
  5. ``run(...)``       — execute arms, judge blindly, assemble verdict.

The base class implements ``verify()`` (re-runs the lock-hash check
and the gate computation against the saved per-query labels) and
``falsify()`` (the gate-first self-test: substitute FULL := B0 and
prove G1 fires). Concrete probes only fill the obligations above.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kaos.eval.harness.manifest import load_lock
from kaos.eval.harness.types import ArmResults, GateOutcome
from kaos.eval.harness.verdict import compute_verdict


class Probe(ABC):
    """Subclass and set ``lock_path`` + ``known_sha256`` as class vars."""

    lock_path: str | Path
    known_sha256: dict[str, str]
    kappa_min: float = 0.85

    def __init__(self) -> None:
        self.lock = load_lock(self.lock_path, self.known_sha256)

    @abstractmethod
    def arms(self) -> list[str]:
        ...

    @abstractmethod
    def gates(self, arms: dict[str, ArmResults]) -> list[GateOutcome]:
        ...

    @abstractmethod
    def run(self, *, out_dir: str | Path, **kw: Any) -> dict:
        """Execute the probe end-to-end and write results to disk.

        Must return a dict with at least: ``arms`` (per-arm per-query
        results), ``gates`` (list of GateOutcome as dicts), ``verdict``
        (str), and ``judge_kappa`` (float).
        """
        ...

    def verify(self, results_path: str | Path) -> str:
        """Re-compute the verdict from a saved results.json. Confirms
        the verdict on file matches what the current gate code emits.
        """
        data = json.loads(Path(results_path).read_text())
        arms = {
            name: ArmResults(
                arm=name,
                per_query=[
                    _qr_from_dict(q) for q in arr["per_query"]
                ],
            )
            for name, arr in data["arms"].items()
        }
        outcomes = self.gates(arms)
        verdict = compute_verdict(
            outcomes,
            judge_kappa=float(data.get("judge_kappa", 1.0)),
            kappa_min=self.kappa_min,
        )
        return verdict

    def falsify(self) -> tuple[list[GateOutcome], str]:
        """Gate-first self-test: substitute the feature arm by a
        baseline and assert the kill-gate fires. Default substitutes
        ``FULL := B0``; override if the probe uses different names.
        """
        arms = self._stub_arms_for_falsification()
        outcomes = self.gates(arms)
        verdict = compute_verdict(
            outcomes, judge_kappa=1.0, kappa_min=self.kappa_min,
        )
        return outcomes, verdict

    def _stub_arms_for_falsification(self) -> dict[str, ArmResults]:
        """Default falsification: requires the subclass to also provide
        ``synth_stub_arms()`` producing realistic B0..B3/L1..L3 + FULL
        where FULL is byte-identical to B0. Subclasses can override
        this whole method if their arm taxonomy differs.
        """
        raise NotImplementedError(
            "Subclasses must implement either falsify() or "
            "_stub_arms_for_falsification() for the gate-first self-test."
        )


def _qr_from_dict(d: dict) -> Any:
    from kaos.eval.harness.types import QueryResult
    return QueryResult(
        qid=d["qid"],
        qclass=d["qclass"],
        correct=bool(d["correct"]),
        split=d.get("split", "in_dist"),
        extras=d.get("extras", {}),
    )
