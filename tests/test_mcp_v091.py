"""v0.9.1 — MCP exposure of doctor / eval probe / experiment surfaces.

These tests exercise the FULL MCP dispatch path (list_tools + call_tool),
not just the underlying primitives. The point is to catch wiring bugs:
schema mismatches, missing dispatch branches, init order, JSON
serialization. Each test routes through the same code Claude Code
would route through.

Eight new tools (50 -> 58):
  doctor_proposer
  eval_probe_falsify / eval_probe_run / eval_probe_verify
  experiment_log / experiment_list / experiment_show / experiment_compare
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from kaos import Kaos
from kaos.mcp.server import (
    call_tool, init_server, list_tools,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def mcp_server(tmp_path: Path):
    """Initialize the MCP server against a fresh kaos.db (so
    experiment_* dispatch can write to it). Returns the path.

    ccr is set to None — the v0.9.1 tools (doctor/eval/experiment)
    never touch _ccr, only _afs. Other tools that need _ccr would
    fail in isolation, but they're not under test here.
    """
    db_path = tmp_path / "kaos.db"
    afs = Kaos(db_path=str(db_path))
    init_server(afs, None)  # type: ignore[arg-type]
    yield db_path
    afs.close()


def _call(name: str, args: dict) -> dict:
    """Helper: invoke the MCP call_tool handler and return parsed JSON."""
    result = asyncio.run(call_tool(name, args))
    assert len(result) == 1
    text = result[0].text
    return json.loads(text)


# ─────────────────────────────────────────────────────────────────────
# list_tools — surface size + schema sanity
# ─────────────────────────────────────────────────────────────────────


class TestListToolsSurface:
    def test_tool_count_is_58(self):
        tools = asyncio.run(list_tools())
        assert len(tools) == 58, (
            f"v0.9.1 surface must be 58 tools (50 prior + 8 new), "
            f"got {len(tools)}"
        )

    def test_v091_tools_are_registered(self):
        tools = asyncio.run(list_tools())
        names = {t.name for t in tools}
        expected = {
            "doctor_proposer",
            "eval_probe_falsify", "eval_probe_run", "eval_probe_verify",
            "experiment_log", "experiment_list",
            "experiment_show", "experiment_compare",
        }
        missing = expected - names
        assert not missing, f"missing v0.9.1 tools: {missing}"

    def test_every_tool_has_object_schema(self):
        """No tool ships with a missing or malformed schema."""
        tools = asyncio.run(list_tools())
        for t in tools:
            assert t.inputSchema.get("type") == "object", t.name
            assert "properties" in t.inputSchema, t.name

    def test_v091_required_fields_are_declared(self):
        """Tools that need params declare them as required."""
        tools = {t.name: t for t in asyncio.run(list_tools())}
        # falsify/run/verify all need 'probe'
        for n in ("eval_probe_falsify", "eval_probe_run",
                  "eval_probe_verify"):
            assert "probe" in tools[n].inputSchema.get("required", []), n
        assert "out_dir" in tools["eval_probe_run"].inputSchema["required"]
        assert "results_path" in tools[
            "eval_probe_verify"].inputSchema["required"]
        assert "name" in tools["experiment_log"].inputSchema["required"]
        assert "exp_id" in tools[
            "experiment_show"].inputSchema["required"]
        assert "a_id" in tools[
            "experiment_compare"].inputSchema["required"]
        assert "b_id" in tools[
            "experiment_compare"].inputSchema["required"]


# ─────────────────────────────────────────────────────────────────────
# eval_probe_* — real probe via the action-realization adapter
# ─────────────────────────────────────────────────────────────────────


PROBE_SPEC = (
    "demo_action_realization_bench.probe_adapter:ActionRealizationProbe"
)


class TestEvalProbeFalsify:
    def test_real_probe_adapter_emits_admissible_reject(self, mcp_server):
        result = _call("eval_probe_falsify", {"probe": PROBE_SPEC})
        # Contract: harness must be admissible (FULL := B1 emits
        # [KILL: G1]) AND verdict starts with REJECT.
        assert result["admissible"] is True
        assert result["verdict"].startswith("REJECT")
        assert "G1" in result["verdict"]
        # 5 outcomes: G0 + G1..G4
        assert len(result["outcomes"]) == 5
        outcomes_by_gate = {o["gate"]: o for o in result["outcomes"]}
        # G0 must pass (sanity floor), G1 must fail (the kill)
        assert outcomes_by_gate["G0"]["passed"] is True
        assert outcomes_by_gate["G1"]["passed"] is False
        assert outcomes_by_gate["G1"]["kill"] is True

    def test_unknown_probe_returns_error_payload(self, mcp_server):
        """A missing class must surface as an MCP error string, not
        crash the server."""
        result = asyncio.run(call_tool(
            "eval_probe_falsify",
            {"probe": "demo_action_realization_bench:DoesNotExist"},
        ))
        text = result[0].text
        assert text.startswith("Error:")
        # ValueError from _load_probe_class is preferred over crash
        assert ("ValueError" in text or "AttributeError" in text)

    def test_malformed_probe_spec_returns_error(self, mcp_server):
        result = asyncio.run(call_tool(
            "eval_probe_falsify", {"probe": "no_colon_here"},
        ))
        assert result[0].text.startswith("Error:")


class TestEvalProbeRun:
    def test_run_against_real_probe_returns_verdict(
        self, mcp_server, tmp_path
    ):
        """The action-realization probe runs end-to-end and returns
        a binding verdict. In the dev environment with sparse organic
        data, that verdict is VOID#1 — exactly what the lock requires."""
        out_dir = tmp_path / "probe_out"
        result = _call("eval_probe_run", {
            "probe": PROBE_SPEC,
            "out_dir": str(out_dir),
        })
        assert "verdict" in result
        # Either VOID (sparse local DB) or ACCEPT/REJECT (CI w/ data).
        v = result["verdict"]
        assert v.startswith(("VOID", "ACCEPT", "REJECT"))
        # Results file was written
        assert (out_dir / "results.json").exists()


class TestEvalProbeVerify:
    def test_verify_against_a_populated_results_file(self, mcp_server,
                                                      tmp_path):
        """Build a populated results.json (via run, NOT the committed
        VOID-on-empty-workload one) and confirm verify echoes a
        verdict. The committed results.json reflects VOID#1 (empty
        arms map) so a verify against it has no arms to reconstruct —
        that path is a known limitation of verify() under VOID."""
        out_dir = tmp_path / "probe_out"
        # Run first to produce a results.json. If it VOIDs on empty
        # arms (the local-DB case), we cannot exercise verify() — skip
        # rather than asserting an arbitrary error message.
        run_result = _call("eval_probe_run", {
            "probe": PROBE_SPEC,
            "out_dir": str(out_dir),
        })
        if not run_result.get("arms"):
            pytest.skip(
                "VOID#1 on empty workload yields empty arms; verify "
                "needs a populated arms map to reconstruct gates"
            )
        verify_result = _call("eval_probe_verify", {
            "probe": PROBE_SPEC,
            "results_path": str(out_dir / "results.json"),
        })
        assert "verdict" in verify_result
        # Verify should produce the SAME verdict as run (gate code
        # at HEAD).
        assert verify_result["verdict"] == run_result["verdict"]

    def test_verify_with_missing_file_returns_error(self, mcp_server):
        result = asyncio.run(call_tool("eval_probe_verify", {
            "probe": PROBE_SPEC,
            "results_path": "/nonexistent/results.json",
        }))
        # Error containment: the MCP wrapper returns an error string
        # rather than crashing the dispatcher.
        assert result[0].text.startswith("Error:")


# ─────────────────────────────────────────────────────────────────────
# doctor_proposer
# ─────────────────────────────────────────────────────────────────────


class TestDoctorProposer:
    def test_missing_config_returns_error_payload(self, mcp_server,
                                                   tmp_path):
        """Without a config file the tool returns a JSON error rather
        than crashing the dispatcher."""
        result = _call("doctor_proposer", {
            "config_file": str(tmp_path / "nope.yaml"),
        })
        assert "error" in result
        assert "config not found" in result["error"]


# ─────────────────────────────────────────────────────────────────────
# experiment_log / list / show / compare — round-trip via MCP
# ─────────────────────────────────────────────────────────────────────


class TestExperimentMCP:
    def test_log_returns_exp_id(self, mcp_server):
        result = _call("experiment_log", {
            "name": "mcp-smoke",
            "family": "probe",
            "verdict": "ACCEPT",
            "judge_kappa": 1.0,
            "git_sha": "",  # suppress auto-fill
            "arms": {"FULL": {"acc": 0.85}},
            "gates": [{"gate": "G1", "passed": True, "kill": True}],
            "metadata": {"via": "mcp_test"},
        })
        assert result["exp_id"] == 1
        assert result["name"] == "mcp-smoke"

    def test_log_list_round_trip(self, mcp_server):
        for i in range(3):
            _call("experiment_log", {
                "name": f"probe-{i}", "family": "probe",
                "verdict": "ACCEPT" if i % 2 == 0 else "REJECT: G1",
                "git_sha": "",
            })
        listing = _call("experiment_list", {})
        assert len(listing["experiments"]) == 3
        # Newest first
        assert listing["experiments"][0]["name"] == "probe-2"

    def test_list_filters_by_verdict_prefix(self, mcp_server):
        _call("experiment_log", {"name": "a", "verdict": "ACCEPT",
                                 "git_sha": ""})
        _call("experiment_log", {"name": "b", "verdict": "REJECT: G1",
                                 "git_sha": ""})
        _call("experiment_log", {"name": "c", "verdict": "VOID: low n",
                                 "git_sha": ""})

        only_accept = _call("experiment_list",
                            {"verdict_prefix": "ACCEPT"})
        only_reject = _call("experiment_list",
                            {"verdict_prefix": "REJECT"})
        only_void = _call("experiment_list", {"verdict_prefix": "VOID"})
        assert len(only_accept["experiments"]) == 1
        assert len(only_reject["experiments"]) == 1
        assert len(only_void["experiments"]) == 1
        assert only_accept["experiments"][0]["name"] == "a"

    def test_list_filters_by_name(self, mcp_server):
        _call("experiment_log", {"name": "alpha", "git_sha": ""})
        _call("experiment_log", {"name": "alpha", "git_sha": ""})
        _call("experiment_log", {"name": "beta", "git_sha": ""})
        only_alpha = _call("experiment_list", {"name": "alpha"})
        assert len(only_alpha["experiments"]) == 2

    def test_show_returns_full_row(self, mcp_server):
        logged = _call("experiment_log", {
            "name": "show-me", "verdict": "ACCEPT",
            "lock_sha256": "deadbeef" * 8,
            "arms": {"FULL": {"acc": 0.9}}, "git_sha": "",
        })
        exp_id = logged["exp_id"]
        shown = _call("experiment_show", {"exp_id": exp_id})
        assert shown["name"] == "show-me"
        assert shown["verdict"] == "ACCEPT"
        assert shown["lock_sha256"] == "deadbeef" * 8
        assert shown["arms"] == {"FULL": {"acc": 0.9}}

    def test_show_missing_returns_error(self, mcp_server):
        result = _call("experiment_show", {"exp_id": 99999})
        assert "error" in result
        assert "no experiment" in result["error"]

    def test_compare_reports_changed_fields(self, mcp_server):
        a = _call("experiment_log", {
            "name": "x", "verdict": "ACCEPT",
            "arms": {"FULL": {"acc": 0.9}}, "git_sha": "sha-1",
        })["exp_id"]
        b = _call("experiment_log", {
            "name": "x", "verdict": "REJECT: G1",
            "arms": {"FULL": {"acc": 0.6}}, "git_sha": "sha-2",
        })["exp_id"]
        diff = _call("experiment_compare", {"a_id": a, "b_id": b})
        assert "verdict" in diff["changes"]
        assert "git_sha" in diff["changes"]
        assert "arms" in diff["changes"]
        # name didn't change
        assert "name" not in diff["changes"]

    def test_compare_missing_returns_error(self, mcp_server):
        a = _call("experiment_log", {"name": "x", "git_sha": ""})["exp_id"]
        diff = _call("experiment_compare", {"a_id": a, "b_id": 99999})
        assert "error" in diff


# ─────────────────────────────────────────────────────────────────────
# Error containment — bad tool name does NOT crash the dispatcher
# ─────────────────────────────────────────────────────────────────────


class TestErrorContainment:
    def test_unknown_tool_returns_error_string(self, mcp_server):
        result = asyncio.run(call_tool("definitely_not_a_tool", {}))
        assert result[0].text.startswith("Error:")
        assert "Unknown tool" in result[0].text

    def test_missing_required_arg_returns_error(self, mcp_server):
        """experiment_show needs exp_id; missing it must not crash."""
        result = asyncio.run(call_tool("experiment_show", {}))
        assert result[0].text.startswith("Error:")
