"""Verdict assembly from a list of GateOutcome.

The verdict rule is uniform across probes:

    VOID    if judge_kappa < kappa_min, or any non-kill gate fails
            (the sanity/diagnostic gates — typically G0).
    ACCEPT  iff every kill-gate passes.
    REJECT  if any kill-gate fails.

This is the only verdict computation in KAOS. Probes assemble their
own gate list (domain-specific predicates) but route the final
{ACCEPT, REJECT, VOID} decision through here so no probe can invent a
softer rule mid-run.
"""

from __future__ import annotations

from kaos.eval.harness.types import GateOutcome


def compute_verdict(
    outcomes: list[GateOutcome],
    *,
    judge_kappa: float,
    kappa_min: float = 0.85,
) -> str:
    """Return one of ``ACCEPT`` / ``REJECT: ...`` / ``VOID: ...``."""
    if judge_kappa < kappa_min:
        return f"VOID: judge-audit kappa={judge_kappa:.3f} < {kappa_min}"
    failed_sanity = [g for g in outcomes if not g.kill and not g.passed]
    if failed_sanity:
        names = ", ".join(g.gate for g in failed_sanity)
        return f"VOID: sanity gate(s) failed: {names}"
    kills = [g for g in outcomes if g.kill]
    if all(g.passed for g in kills):
        return "ACCEPT"
    failed = [g.gate for g in kills if not g.passed]
    return f"REJECT: kill gate(s) failed: {', '.join(failed)}"
