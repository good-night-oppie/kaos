"""PR-2 (v0.9) — kaos.eval.harness falsifiable-eval primitive.

Covers: types, stats, manifest tamper-evidence, verdict assembly,
gate-first falsification self-test (the synthesis bench's actual
contract — FULL := B0 MUST emit [KILL: G1]).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaos.eval.harness import (
    ArmResults,
    GateOutcome,
    LockTamperError,
    QueryResult,
    bootstrap_diff_ci,
    compute_verdict,
    load_lock,
    sha256_file,
)


# ─────────────────────────────────────────────────────────────────────
# types
# ─────────────────────────────────────────────────────────────────────


class TestArmResults:
    def test_acc_empty_returns_zero(self):
        a = ArmResults(arm="X")
        assert a.acc() == 0.0
        assert a.n() == 0

    def test_acc_filters_by_class_and_split(self):
        a = ArmResults(arm="X", per_query=[
            QueryResult("q1", "verbatim_recall", True, "in_dist"),
            QueryResult("q2", "verbatim_recall", False, "in_dist"),
            QueryResult("q3", "verbatim_recall", True, "shift"),
            QueryResult("q4", "compositional_multihop", True, "in_dist"),
        ])
        assert a.acc({"verbatim_recall"}, "in_dist") == 0.5
        assert a.acc({"verbatim_recall"}, "shift") == 1.0
        assert a.acc({"compositional_multihop"}, "in_dist") == 1.0
        assert a.n({"verbatim_recall"}, "in_dist") == 2

    def test_labels_returns_zero_one_list(self):
        a = ArmResults(arm="X", per_query=[
            QueryResult("q1", "C", True),
            QueryResult("q2", "C", False),
            QueryResult("q3", "C", True),
        ])
        assert a.labels({"C"}) == [1, 0, 1]


# ─────────────────────────────────────────────────────────────────────
# stats
# ─────────────────────────────────────────────────────────────────────


class TestBootstrapDiffCI:
    def test_empty_inputs_return_zeros(self):
        assert bootstrap_diff_ci([], []) == (0.0, 0.0, 0.0)
        assert bootstrap_diff_ci([1, 1, 1], []) == (0.0, 0.0, 0.0)

    def test_obvious_positive_difference_has_lo_above_zero(self):
        a = [1] * 100
        b = [0] * 100
        md, lo, hi = bootstrap_diff_ci(a, b, iters=500)
        assert md == pytest.approx(1.0)
        assert lo > 0.0
        assert hi >= md

    def test_obvious_negative_difference_has_hi_below_zero(self):
        a = [0] * 100
        b = [1] * 100
        md, lo, hi = bootstrap_diff_ci(a, b, iters=500)
        assert md == pytest.approx(-1.0)
        assert hi < 0.0

    def test_no_difference_ci_contains_zero(self):
        a = [1, 0] * 50
        b = [1, 0] * 50
        md, lo, hi = bootstrap_diff_ci(a, b, iters=500)
        assert lo <= 0.0 <= hi

    def test_seed_reproducibility(self):
        a = [1, 0, 1, 0, 1] * 20
        b = [0, 1, 0, 1, 0] * 20
        r1 = bootstrap_diff_ci(a, b, iters=300, seed=7)
        r2 = bootstrap_diff_ci(a, b, iters=300, seed=7)
        assert r1 == r2


# ─────────────────────────────────────────────────────────────────────
# manifest tamper-evidence
# ─────────────────────────────────────────────────────────────────────


class TestManifestTamperEvidence:
    def test_known_hash_loads_successfully(self, tmp_path: Path):
        lock_path = tmp_path / "X.lock.json"
        lock_path.write_text(json.dumps({"hello": "world"}))
        h = sha256_file(lock_path)
        data = load_lock(lock_path, {h: "test"})
        assert data == {"hello": "world"}

    def test_unknown_hash_raises_LockTamperError(self, tmp_path: Path):
        lock_path = tmp_path / "X.lock.json"
        lock_path.write_text(json.dumps({"hello": "world"}))
        with pytest.raises(LockTamperError) as ei:
            load_lock(lock_path, {"deadbeef" * 8: "test"})
        assert "VOID: lock-tamper" in str(ei.value)

    def test_edited_lock_changes_hash_and_blocks(self, tmp_path: Path):
        lock_path = tmp_path / "X.lock.json"
        lock_path.write_text('{"v": 1}')
        h1 = sha256_file(lock_path)
        load_lock(lock_path, {h1: "v1"})

        lock_path.write_text('{"v": 2}')
        with pytest.raises(LockTamperError):
            load_lock(lock_path, {h1: "v1"})


# ─────────────────────────────────────────────────────────────────────
# verdict assembly
# ─────────────────────────────────────────────────────────────────────


def _g(name: str, passed: bool, kill: bool) -> GateOutcome:
    return GateOutcome(gate=name, name=name, passed=passed, kill=kill, detail="")


class TestVerdictAssembly:
    def test_all_kills_pass_yields_accept(self):
        out = [_g("G0", True, False), _g("G1", True, True),
               _g("G2", True, True)]
        assert compute_verdict(out, judge_kappa=1.0) == "ACCEPT"

    def test_failed_kill_gate_rejects(self):
        out = [_g("G0", True, False), _g("G1", False, True),
               _g("G2", True, True)]
        v = compute_verdict(out, judge_kappa=1.0)
        assert v.startswith("REJECT")
        assert "G1" in v

    def test_failed_sanity_gate_voids(self):
        out = [_g("G0", False, False), _g("G1", True, True)]
        v = compute_verdict(out, judge_kappa=1.0)
        assert v.startswith("VOID")
        assert "G0" in v

    def test_low_kappa_voids(self):
        out = [_g("G1", True, True)]
        v = compute_verdict(out, judge_kappa=0.5)
        assert v.startswith("VOID")
        assert "kappa" in v

    def test_multiple_kill_failures_listed(self):
        out = [_g("G1", False, True), _g("G2", False, True),
               _g("G3", True, True)]
        v = compute_verdict(out, judge_kappa=1.0)
        assert v.startswith("REJECT")
        assert "G1" in v and "G2" in v


# ─────────────────────────────────────────────────────────────────────
# gate-first falsification (Probe contract: feature CAN lose)
# ─────────────────────────────────────────────────────────────────────


class TestGateFirstFalsification:
    """Mirror the synthesis-bench falsify.py invariant: with FULL := B0
    a typical kill-gate (FULL beats B0 by some margin) MUST fail.
    A harness that cannot kill is INADMISSIBLE."""

    def _mk_arm(self, name: str, hard_acc: float, seed: int,
                n_per: int = 200) -> ArmResults:
        import random
        rng = random.Random(hash((name, seed)) & 0xffffffff)
        a = ArmResults(arm=name)
        for cls in ("hard_class",):
            for i in range(n_per):
                a.per_query.append(QueryResult(
                    qid=f"{name}-{cls}-{i}", qclass=cls,
                    correct=(rng.random() < hard_acc),
                    split="in_dist",
                ))
        return a

    def _gates(self, arms: dict[str, ArmResults]) -> list[GateOutcome]:
        full = arms["FULL"].acc({"hard_class"})
        b0 = arms["B0"].acc({"hard_class"})
        md, lo, hi = bootstrap_diff_ci(
            arms["FULL"].labels({"hard_class"}),
            arms["B0"].labels({"hard_class"}),
            iters=500,
        )
        passed = (full - b0 >= 0.10) and lo > 0.0
        return [GateOutcome(
            gate="G1", name="beats baseline", passed=passed, kill=True,
            detail=f"FULL={full:.3f} B0={b0:.3f} diff={full-b0:+.3f} "
                   f"lo={lo:+.3f}",
        )]

    def test_full_equals_b0_triggers_kill_gate(self):
        b0 = self._mk_arm("B0", hard_acc=0.40, seed=1)
        full = ArmResults(arm="FULL", per_query=list(b0.per_query))
        arms = {"B0": b0, "FULL": full}
        outs = self._gates(arms)
        v = compute_verdict(outs, judge_kappa=1.0)
        assert v.startswith("REJECT")
        assert "G1" in v

    def test_genuine_lift_passes(self):
        b0 = self._mk_arm("B0", hard_acc=0.40, seed=1)
        full = self._mk_arm("FULL", hard_acc=0.70, seed=2)
        arms = {"B0": b0, "FULL": full}
        outs = self._gates(arms)
        v = compute_verdict(outs, judge_kappa=1.0)
        assert v == "ACCEPT"


# ─────────────────────────────────────────────────────────────────────
# Probe lifecycle (subclass smoke)
# ─────────────────────────────────────────────────────────────────────


class TestProbeBaseClass:
    def test_subclass_with_locked_manifest_works(self, tmp_path: Path):
        from kaos.eval.harness import Probe

        lp = tmp_path / "demo.lock.json"
        lp.write_text(json.dumps({"v": 1}))
        h = sha256_file(lp)
        lp_str = str(lp)

        class _Demo(Probe):
            lock_path = lp_str
            known_sha256 = {h: "v1"}

            def arms(self) -> list[str]:
                return ["B0", "FULL"]

            def gates(self, arms_in: dict[str, ArmResults]
                      ) -> list[GateOutcome]:
                return [GateOutcome("G1", "smoke", True, True, "")]

            def run(self, *, out_dir, **kw):
                return {"arms": {}, "gates": [], "verdict": "ACCEPT",
                        "judge_kappa": 1.0}

        d = _Demo()
        assert d.arms() == ["B0", "FULL"]
        gs = d.gates({})
        assert gs[0].gate == "G1"

    def test_subclass_rejects_tampered_lock(self, tmp_path: Path):
        from kaos.eval.harness import Probe

        lp = tmp_path / "demo.lock.json"
        lp.write_text(json.dumps({"v": 1}))
        lp_str = str(lp)

        class _Demo(Probe):
            lock_path = lp_str
            known_sha256 = {"00" * 32: "fake"}

            def arms(self):
                return []

            def gates(self, arms_in):
                return []

            def run(self, *, out_dir, **kw):
                return {}

        with pytest.raises(LockTamperError):
            _Demo()
