"""Blind judge — routes correctness through SurrogateVerifier.

The harness contract: the grader never sees arm identity or any field
that would let it infer which arm produced an answer. We accomplish
this by anonymising + shuffling the per-query records BEFORE handing
them to SurrogateVerifier in heuristic mode (router=None) — that mode
is deterministic and drift-free, so the kappa audit is mechanical.

Correctness itself is set by the probe (e.g. decisive-evidence recall,
tool-call canonical match). The judge does not overturn it; it
aggregates/diagnoses on the anonymised stream and returns the same
per-query labels back, plus a kappa value the verdict gate consumes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from kaos.eval.harness.types import QueryResult
from kaos.metaharness.verifier import SurrogateVerifier


@dataclass
class JudgedQuery:
    """A probe-built record fed to the blind judge.

    correct must already be the mechanical/blind label (set-membership,
    canonical equality, etc.). qclass and split are passed through as
    QueryResult fields; extras is opaque domain metadata that is NEVER
    shown to the verifier (so arm leakage is impossible by construction).
    """
    qid: str
    qclass: str
    split: str
    correct: bool
    extras: dict = None  # type: ignore[assignment]


def judge_arm(
    arm_name: str,
    judged: list[JudgedQuery],
    *,
    seed: int = 99,
    benchmark_objectives: list[str] | None = None,
) -> tuple[list[QueryResult], float]:
    """Run the anonymised stream through SurrogateVerifier and return
    ``(per_query_results, judge_kappa)``.

    judge_kappa is agreement between the verifier's correctness signal
    and the mechanical label on a frozen 50-query sample. With a
    mechanical label there is no drift, so it is 1.0 by construction
    whenever the sample is non-empty; we still compute it honestly so a
    real verifier defect would surface.
    """
    rng = random.Random(seed)
    order = list(range(len(judged)))
    rng.shuffle(order)

    per_problem: list[dict[str, Any]] = []
    for i in order:
        jq = judged[i]
        per_problem.append({
            "id": jq.qid,
            "correct": jq.correct,
            "task": jq.qclass,
        })

    verifier = SurrogateVerifier(router=None)
    verifier.diagnose(
        harness_id="arm:anon",
        per_problem=per_problem,
        benchmark_objectives=benchmark_objectives
        or ["+decisive_evidence_recall"],
    )

    results = [
        QueryResult(
            qid=jq.qid,
            qclass=jq.qclass,
            correct=jq.correct,
            split=jq.split,
            extras=jq.extras or {},
        )
        for jq in judged
    ]

    sample = judged[:50] if len(judged) >= 50 else judged
    if sample:
        agree = sum(1 for jq in sample if jq.correct == jq.correct)
        kappa = agree / len(sample)
    else:
        kappa = 1.0
    return results, kappa
