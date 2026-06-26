"""Tests for the situation index — failure-region navigation aid."""

from __future__ import annotations

from kaos.metaharness.situation import (
    SituationIndex,
    SituationRegion,
    build_situation_index,
    render_situation_brief,
)


def test_build_situation_index_groups_failed_observations_by_region_key() -> None:
    index = build_situation_index(
        [
            {
                "harness_id": "h1",
                "evidence_path": "/harnesses/h1/per_problem.jsonl",
                "per_problem": [
                    {
                        "problem_id": "p1",
                        "correct": False,
                        "scores": {"accuracy": 0.0, "modal_collapse_resistance": 0.0},
                        "region_key": ["triage", "both_wrong"],
                        "diagnostic": {"expected_action": "new_bug"},
                    },
                    {
                        "problem_id": "p2",
                        "correct": True,
                        "scores": {"accuracy": 1.0},
                        "region_key": ["triage", "positive"],
                        "diagnostic": {},
                    },
                ],
            },
            {
                "harness_id": "h2",
                "evidence_path": "/harnesses/h2/per_problem.jsonl",
                "per_problem": [
                    {
                        "problem_id": "p3",
                        "correct": False,
                        "scores": {"accuracy": 0.0},
                        "region_key": ["triage", "both_wrong"],
                        "diagnostic": {"expected_action": "new_bug"},
                    },
                    {
                        "problem_id": "p4",
                        "correct": False,
                        "scores": {"accuracy": 0.5},
                        "region_key": ["triage", "network_ui"],
                        "diagnostic": {"expected_target": "QFS-131804"},
                    },
                ],
            },
        ],
        max_regions=5,
        max_examples_per_region=2,
    )

    assert index == SituationIndex(
        regions=[
            SituationRegion(
                key=("triage", "both_wrong"),
                observation_count=2,
                problem_ids=["p1", "p3"],
                failed_objectives={
                    "accuracy": 2,
                    "modal_collapse_resistance": 1,
                },
                sample_diagnostics=[
                    {"expected_action": "new_bug"},
                    {"expected_action": "new_bug"},
                ],
                evidence_paths=[
                    "/harnesses/h1/per_problem.jsonl",
                    "/harnesses/h2/per_problem.jsonl",
                ],
            ),
            SituationRegion(
                key=("triage", "network_ui"),
                observation_count=1,
                problem_ids=["p4"],
                failed_objectives={"accuracy": 1},
                sample_diagnostics=[{"expected_target": "QFS-131804"}],
                evidence_paths=["/harnesses/h2/per_problem.jsonl"],
            ),
        ],
        total_observations=4,
        omitted_regions=0,
    )


def test_render_situation_brief_is_bounded_evidence_index() -> None:
    index = build_situation_index(
        [
            {
                "harness_id": "h1",
                "evidence_path": "/harnesses/h1/per_problem.jsonl",
                "per_problem": [
                    {
                        "problem_id": "path-mismatch",
                        "correct": False,
                        "scores": {"accuracy": 0.0},
                        "region_key": ["triage", "both_wrong"],
                        "diagnostic": {
                            "failure_code_paths": ["perf/fio/node_down_systest.py"],
                        },
                    }
                ],
            }
        ]
    )

    brief = render_situation_brief(index)

    assert "Search Situation Index" in brief
    assert "index over archived evidence, not ground truth" in brief
    assert "triage / both_wrong" in brief
    assert "observations: 1" in brief
    assert "unique problems: 1" in brief
    assert "/harnesses/h1/per_problem.jsonl" in brief
    assert "path-mismatch" in brief
    assert "perf/fio/node_down_systest.py" in brief


def test_render_empty_index_returns_empty() -> None:
    """Brief render of empty index returns '' so callers can short-circuit."""
    index = SituationIndex(regions=[], total_observations=0)
    assert render_situation_brief(index) == ""


def test_frontier_seed_with_low_score_surfaces_via_situation_brief() -> None:
    """Repro for the original bug: a buggy seed that's Pareto-optimal on cost
    but accuracy=0 was invisible to the proposer because compactor truncation
    dropped it. The situation engine surfaces its evidence_path independently.
    """
    records = [
        {
            "harness_id": "seed_buggy",
            "evidence_path": "/harnesses/seed_buggy/per_problem.jsonl",
            "per_problem": [
                {
                    "problem_id": f"p{i}",
                    "correct": False,
                    "scores": {"accuracy": 0.0, "context_cost": 0.0},
                    "region_key": ["empty_prediction"],
                    "diagnostic": {"observed": ""},
                }
                for i in range(8)
            ],
        },
        # Other harnesses with non-zero scores but some failures
        {
            "harness_id": "h_better",
            "evidence_path": "/harnesses/h_better/per_problem.jsonl",
            "per_problem": [
                {
                    "problem_id": "p9",
                    "correct": False,
                    "scores": {"accuracy": 0.5},
                    "region_key": ["partial_match"],
                    "diagnostic": {},
                }
            ],
        },
    ]
    index = build_situation_index(records)
    brief = render_situation_brief(index)

    # Seed evidence path surfaces even though it has all-zero scores
    assert "/harnesses/seed_buggy/per_problem.jsonl" in brief
    # Failure region key surfaces so proposer can hypothesize bug class
    assert "empty_prediction" in brief
    # Failed-objectives counter aggregates per-problem failures
    assert "accuracy=8" in brief
