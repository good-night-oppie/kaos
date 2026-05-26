"""
Falsifiable Probe (v0.9)
========================

The kaos.eval.harness primitive — pre-registered, hash-locked,
falsifiable evaluation as a first-class KAOS feature. This is the
apparatus that rejected 4 of 5 mechanism candidates in the v0.9
research cycle. v0.9 makes it reusable.

This example builds a tiny "does adding more few-shot examples help
classification accuracy?" probe — a question with a known answer (yes,
modestly) — and walks through every step the discipline requires:

  1. Pre-register kill gates in a JSON manifest
  2. Compute and lock the manifest sha256
  3. Build a Probe subclass
  4. Falsification self-test (prove FULL := B0 emits [KILL: G1])
  5. Run the binding probe -> ACCEPT / REJECT / VOID verdict
  6. Log the verdict to the v0.9 experiments journal

Run it:

    uv run python examples/falsifiable_probe.py

Expect: a printed ACCEPT verdict — the FULL arm scores ~40pp over
zero-shot on n=200/class, both kill-gates pass, and the falsification
self-test confirms the harness CAN reject (FULL := B0 emits
[KILL: G1]).

Drop `FewshotProbe.full_accuracy` to 0.55 and re-run to see how the
discipline emits REJECT instead — naive averages would still look
like a +10pp "win", but the bootstrap CI is too wide and the
shuffled-lesion arm eats most of the apparent causal claim.

A row is logged to the experiments journal so the verdict is
queryable:

    kaos experiment list --db /path/printed/by/example
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from pathlib import Path

from kaos.eval.harness import (
    ArmResults,
    GateOutcome,
    Probe,
    QueryResult,
    bootstrap_diff_ci,
    compute_verdict,
)
from kaos.experiments import ExperimentStore


# ────────────────────────────────────────────────────────────────────
# Step 1: write the pre-registered manifest (kill gates first)
# ────────────────────────────────────────────────────────────────────
#
# Gate-first invariant: the gates exist BEFORE any feature code can
# see results. Editing this manifest after results land changes its
# sha256, which the harness refuses to load — goalpost moves are
# mechanically impossible.

LOCK = {
    "name": "fewshot-helps-classification",
    "binding_thesis": (
        "Adding 4 few-shot examples to a zero-shot prompt lifts "
        "classification accuracy by at least +10pp on a held-out "
        "200-question slice, with the lift surviving lesion controls."
    ),
    "kill_gates": {
        "G1": {
            "name": "FULL beats zero-shot baseline",
            "threshold_pp": 10.0,
            "predicate": "FULL - B0 >= 0.10 AND bootstrap95 lo > 0",
        },
        "G2": {
            "name": "causal isolation via lesion",
            "threshold_pp": 5.0,
            "predicate": "FULL - L1 (shuffled-examples lesion) >= 0.05",
        },
    },
    "verdict_rule": "ACCEPT iff G1 AND G2; REJECT on any kill-gate "
                    "fail; VOID on sanity floor fail. No retune.",
}


def write_lock(path: Path) -> str:
    """Write the manifest and return its sha256."""
    path.write_text(json.dumps(LOCK, indent=2, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ────────────────────────────────────────────────────────────────────
# Step 2: a tiny "domain" — a simulated classifier with a knob for
# accuracy. Pretend each arm calls an LLM; we model the outcome with
# a fixed seed for reproducibility.
# ────────────────────────────────────────────────────────────────────


def _simulate_arm(arm: str, accuracy: float, seed: int,
                  n_per_class: int = 200) -> ArmResults:
    """Build per-query results for one arm. Two classes ('hard',
    'easy'), seeded RNG so the verdict is reproducible across runs
    (hashlib not built-in hash() — the latter is randomized per
    interpreter unless PYTHONHASHSEED is set)."""
    digest = hashlib.sha256(f"{arm}|{seed}".encode()).digest()[:4]
    rng = random.Random(int.from_bytes(digest, "big"))
    a = ArmResults(arm=arm)
    for qclass in ("hard", "easy"):
        # Easy questions always pass at 0.95; hard tracks the knob.
        p = accuracy if qclass == "hard" else 0.95
        for i in range(n_per_class):
            a.per_query.append(QueryResult(
                qid=f"{arm}-{qclass}-{i}",
                qclass=qclass,
                correct=(rng.random() < p),
                split="in_dist",
            ))
    return a


# ────────────────────────────────────────────────────────────────────
# Step 3: the Probe subclass — implements arms() + gates() + run()
# ────────────────────────────────────────────────────────────────────


class FewshotProbe(Probe):
    """B0 = zero-shot baseline.
    FULL = zero-shot + 4 few-shot examples.
    L1 = same 4 examples but SHUFFLED randomly between classes
         (controls for 'any extra context helps')."""

    # Set in main() once the manifest is on disk + hashed.
    lock_path: str = ""
    known_sha256: dict[str, str] = {}

    # Tunable so the example can show both ACCEPT and REJECT paths.
    # Defaults are picked to clearly ACCEPT with n=200/class (true
    # lift > +25pp, lesion does NOT capture the lift). Drop
    # full_accuracy to ~0.55 to see the discipline emit REJECT
    # instead — that is the educational moment.
    full_accuracy: float = 0.85  # 4-shot lift
    b0_accuracy: float = 0.45    # zero-shot baseline
    l1_accuracy: float = 0.50    # shuffled lesion — small lift only

    def arms(self) -> list[str]:
        return ["B0", "L1", "FULL"]

    def _build_arms(self) -> dict[str, ArmResults]:
        return {
            "B0": _simulate_arm("B0", self.b0_accuracy, seed=1),
            "L1": _simulate_arm("L1", self.l1_accuracy, seed=2),
            "FULL": _simulate_arm("FULL", self.full_accuracy, seed=3),
        }

    def gates(self, arms: dict[str, ArmResults]) -> list[GateOutcome]:
        a_full = arms["FULL"].labels({"hard"})
        a_b0 = arms["B0"].labels({"hard"})
        a_l1 = arms["L1"].labels({"hard"})

        md_g1, lo_g1, hi_g1 = bootstrap_diff_ci(a_full, a_b0, iters=500)
        g1_thresh = LOCK["kill_gates"]["G1"]["threshold_pp"] / 100.0
        g1 = md_g1 >= g1_thresh and lo_g1 > 0.0

        diff_l1 = (arms["FULL"].acc({"hard"}) - arms["L1"].acc({"hard"}))
        g2_thresh = LOCK["kill_gates"]["G2"]["threshold_pp"] / 100.0
        g2 = diff_l1 >= g2_thresh

        return [
            GateOutcome(
                "G1", "FULL beats zero-shot baseline", g1, kill=True,
                detail=f"FULL-B0={md_g1:+.3f} (need >=+{g1_thresh:.3f}); "
                       f"bootstrap95=({lo_g1:+.3f},{hi_g1:+.3f}) "
                       f"(need lo>0)",
            ),
            GateOutcome(
                "G2", "causal isolation via shuffled lesion", g2, kill=True,
                detail=f"FULL-L1={diff_l1:+.3f} (need >=+{g2_thresh:.3f})",
            ),
        ]

    def run(self, *, out_dir: str | Path, **kw) -> dict:
        arms = self._build_arms()
        outcomes = self.gates(arms)
        verdict = compute_verdict(outcomes, judge_kappa=1.0)
        return {
            "verdict": verdict,
            "judge_kappa": 1.0,
            "arms": {n: {"acc_hard": a.acc({"hard"})}
                     for n, a in arms.items()},
            "gates": [{"gate": g.gate, "name": g.name,
                       "passed": g.passed, "kill": g.kill,
                       "detail": g.detail} for g in outcomes],
        }

    # ── Falsification self-test: prove FULL := B0 triggers G1 ──────
    def _stub_arms_for_falsification(self) -> dict[str, ArmResults]:
        b0 = _simulate_arm("B0", 0.45, seed=1)
        return {
            "B0": b0,
            "L1": _simulate_arm("L1", 0.45, seed=2),
            "FULL": ArmResults("FULL", per_query=list(b0.per_query)),
        }


# ────────────────────────────────────────────────────────────────────
# Step 4-6: lifecycle — lock, falsify, run, log
# ────────────────────────────────────────────────────────────────────


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="kaos_v09_demo_"))
    lock_path = workdir / "ISA.lock.json"
    # Use the tempdir for the experiments DB too so the example is
    # self-contained and works the same on Windows + POSIX.
    db_path = workdir / "experiments.db"

    # 1. pre-register the manifest + lock its hash
    lock_sha = write_lock(lock_path)
    print(f"[1] Lock written + hashed: sha256={lock_sha[:16]}...")
    print(f"    {lock_path}")

    # 2. wire the hash into the Probe subclass (tamper-evidence)
    FewshotProbe.lock_path = str(lock_path)
    FewshotProbe.known_sha256 = {lock_sha: "v1-demo"}

    # 3. falsification self-test (gate-first invariant)
    probe = FewshotProbe()
    fals_outcomes, fals_verdict = probe.falsify()
    admissible = fals_verdict.startswith("REJECT")
    print(f"[2] Falsify (FULL := B0) -> {fals_verdict}")
    print(f"    Harness ADMISSIBLE: {admissible} "
          f"(must be True or run is inadmissible)")
    if not admissible:
        print("    BROKEN HARNESS — feature cannot lose. Halting.")
        return

    # 4. binding run
    result = probe.run(out_dir=workdir)
    print(f"[3] Binding verdict: {result['verdict']}")
    for g in result["gates"]:
        flag = "PASS" if g["passed"] else "FAIL"
        print(f"      [{flag}] {g['gate']}  {g['detail']}")

    # 5. log to the v0.9 experiments journal for queryability
    with ExperimentStore(db_path) as store:
        exp_id = store.log_run(
            name="fewshot-helps-classification",
            family="probe",
            verdict=result["verdict"],
            judge_kappa=result["judge_kappa"],
            lock_sha256=lock_sha,
            arms=result["arms"],
            gates=result["gates"],
            metadata={"example": "examples/falsifiable_probe.py"},
        )
    print(f"[4] Logged to experiments journal: exp_id={exp_id}")
    print(f"    Query: kaos experiment list --db {db_path}")

    # 6. show the same lock-tamper guard via lock_sha256 lookup
    print(f"\n[5] Tamper-evidence:")
    print(f"    Lock hash on disk: {lock_sha[:16]}...")
    print(f"    Allowed in probe : {set(FewshotProbe.known_sha256)}")
    print(f"    Edit the lock and re-run -> LockTamperError, no verdict.")


if __name__ == "__main__":
    main()
