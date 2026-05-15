"""Critical-step localizer (Track B2, v0.8.3).

Borrowed from "Where LLM Agents Fail and How They Can Learn From Failures"
(arXiv:2509.25370). When an agent fails, the *visible* error is rarely the
*decisive* one — the agent usually went wrong several steps earlier and the
later error is just where it finally surfaced. This phase reconstructs the
agent's trajectory (tool_calls + shared_log, chronologically) and points at
the earliest step that decided the outcome.

Heuristic-first, LLM-fallback — the same discipline as the diagnoser:
deterministic scoring handles the common shapes for free; the LLM is only
consulted when the heuristic is not confident, and its answer is cached by
trajectory fingerprint so a recurring failure shape pays the model cost at
most once.

Pure read over data KAOS already persists; writes one ``critical_steps``
row. No hot-path cost — this runs in the dream cycle.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TraceStep:
    """One point on the agent's timeline. ``kind`` is 'tool' or 'log'."""
    kind: str
    position: int          # ordering key (monotonic across the merged trace)
    ts: str                # ISO timestamp
    label: str             # human-readable summary
    is_error: bool
    is_decision: bool      # intent / vote / decision / a write-ish tool call
    raw_id: Any            # tool_calls.call_id or shared_log.log_id


@dataclass
class CriticalStep:
    agent_id: str
    log_position: int
    tool_call_id: str | None
    rationale: str
    method: str            # 'heuristic' | 'llm' | 'llm-cached'
    confidence: float
    fingerprint_id: int | None = None
    isc_id: int | None = None


# Tool names / log types that "lock in" a direction — a wrong one here is
# far more decisive than a wrong read-only lookup.
_DECISIVE_LOG_TYPES = {"intent", "vote", "decision", "commit"}
_DECISIVE_TOOL_HINTS = ("write", "save", "delete", "deploy", "apply",
                        "exec", "run", "create", "update", "merge", "push")


def _load_trace(conn: sqlite3.Connection, agent_id: str) -> list[TraceStep]:
    """Merge tool_calls + shared_log for one agent into a single ordered
    timeline. Ordering is by timestamp; ties broken tool-before-log so an
    erroring tool call sorts after the intent that launched it."""
    steps: list[TraceStep] = []

    try:
        rows = conn.execute(
            "SELECT call_id, tool_name, status, error_message, "
            "started_at FROM tool_calls WHERE agent_id = ? "
            "ORDER BY started_at",
            (agent_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    for r in rows:
        call_id, tool_name, status, err, ts = (
            r[0], r[1], r[2], r[3], r[4],
        )
        is_err = (status == "error") or bool(err)
        lname = (tool_name or "").lower()
        decisive = any(h in lname for h in _DECISIVE_TOOL_HINTS)
        label = f"tool:{tool_name}" + (f" ERROR {err[:80]}" if err else "")
        steps.append(TraceStep(
            kind="tool", position=0, ts=ts or "",
            label=label, is_error=is_err, is_decision=decisive,
            raw_id=call_id,
        ))

    try:
        rows = conn.execute(
            "SELECT log_id, position, type, payload, created_at "
            "FROM shared_log WHERE agent_id = ? ORDER BY position",
            (agent_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    for r in rows:
        log_id, pos, ltype, payload, ts = r[0], r[1], r[2], r[3], r[4]
        try:
            pj = json.loads(payload or "{}")
        except (json.JSONDecodeError, TypeError):
            pj = {}
        action = pj.get("action") or pj.get("reason") or pj.get("summary") or ""
        is_err = ltype == "abort"
        steps.append(TraceStep(
            kind="log", position=pos, ts=ts or "",
            label=f"log:{ltype} {str(action)[:80]}".strip(),
            is_error=is_err,
            is_decision=ltype in _DECISIVE_LOG_TYPES,
            raw_id=log_id,
        ))

    # Stable chronological order. Tool calls have no position; sort all by
    # (timestamp, kind) where 'log' < 'tool' so an intent precedes the tool
    # it triggered when timestamps collide.
    steps.sort(key=lambda s: (s.ts, 0 if s.kind == "log" else 1))
    for i, s in enumerate(steps):
        s.position = i
    return steps


def _trace_fingerprint(steps: list[TraceStep]) -> str:
    """Stable hash over the *shape* of the trace (kinds + error flags +
    decision flags), not the contents — so two structurally-identical
    failures share a cached localization."""
    shape = "|".join(
        f"{s.kind}:{int(s.is_error)}{int(s.is_decision)}" for s in steps
    )
    return hashlib.sha256(shape.encode("utf-8", "replace")).hexdigest()[:16]


def _heuristic_localize(
    steps: list[TraceStep],
) -> tuple[int, str, float] | None:
    """Return (index_into_steps, rationale, confidence) or None.

    Logic: find the first error. Walk *backwards* from it collecting
    decision steps. The earliest decision step that precedes the first
    error and was itself not a read-only lookup is the prime suspect —
    a wrong decision that the agent only paid for later. If there is no
    decision before the error, the error step itself is the critical one.
    """
    if not steps:
        return None
    first_err_idx = next(
        (i for i, s in enumerate(steps) if s.is_error), None
    )
    if first_err_idx is None:
        return None  # nothing failed — nothing to localize

    decisions_before = [
        i for i in range(first_err_idx) if steps[i].is_decision
    ]
    if not decisions_before:
        s = steps[first_err_idx]
        return (
            first_err_idx,
            f"No decisive step preceded the failure; the error at "
            f"`{s.label}` is itself the critical step.",
            0.7,
        )

    earliest = decisions_before[0]
    s = steps[earliest]
    # Confidence rises with the distance between the decisive step and the
    # visible error (a bug that festered N steps is a clearer "earliest
    # critical" signal), and with how many decisions piled up after it.
    gap = first_err_idx - earliest
    conf = min(0.9, 0.55 + 0.05 * gap + 0.04 * (len(decisions_before) - 1))
    return (
        earliest,
        f"Earliest decisive step before the failure: `{s.label}` "
        f"({gap} step(s) before the visible error). The agent locked in "
        f"this direction here; later steps inherited it.",
        round(conf, 3),
    )


_LLM_PROMPT = """You are localizing the EARLIEST decisive error in a failed AI agent trajectory.

The visible error is rarely the root cause — the agent usually went wrong earlier and only paid for it later. Identify the single step where the outcome was effectively decided.

Trajectory (chronological, 0-indexed):
{trace}

Respond as STRICT JSON:
  "index":      integer step number that is the earliest critical step
  "rationale":  one sentence explaining why that step decided the outcome
  "confidence": float in [0,1]

Return the JSON object and nothing else."""


def _llm_localize(
    steps: list[TraceStep],
    call_fn: Callable[[str], str],
) -> tuple[int, str, float] | None:
    trace = "\n".join(
        f"{i}: [{s.kind}]{' ERR' if s.is_error else ''}"
        f"{' DECISIVE' if s.is_decision else ''} {s.label}"
        for i, s in enumerate(steps)
    )
    try:
        raw = call_fn(_LLM_PROMPT.format(trace=trace))
    except Exception:
        return None
    from kaos.dream.diagnosis import _safe_parse_llm_json
    parsed = _safe_parse_llm_json(raw)
    if not parsed:
        return None
    try:
        idx = int(parsed.get("index"))
    except (TypeError, ValueError):
        return None
    if not (0 <= idx < len(steps)):
        return None
    return (
        idx,
        str(parsed.get("rationale") or "LLM localization."),
        float(parsed.get("confidence", 0.6) or 0.6),
    )


def localize(
    conn: sqlite3.Connection,
    agent_id: str,
    *,
    fingerprint_id: int | None = None,
    isc_id: int | None = None,
    llm_call_fn: Callable[[str], str] | None = None,
    heuristic_floor: float = 0.6,
    persist: bool = True,
) -> CriticalStep | None:
    """Localize the earliest decisive error in ``agent_id``'s trajectory.

    Heuristic first. The LLM (if ``llm_call_fn`` given) is consulted only
    when the heuristic confidence is below ``heuristic_floor``, and its
    result is cached by trajectory-shape fingerprint in
    ``llm_diagnosis_cache`` (reusing that table's key space with a
    ``localize:`` prefix so it never collides with diagnosis entries).

    Returns the CriticalStep (also written to ``critical_steps`` when
    ``persist``), or None if the agent's trajectory contains no failure.
    """
    steps = _load_trace(conn, agent_id)
    h = _heuristic_localize(steps)
    if h is None:
        return None

    idx, rationale, conf = h
    method = "heuristic"

    if conf < heuristic_floor and llm_call_fn is not None:
        fp = "localize:" + _trace_fingerprint(steps)
        cached = _cache_get(conn, fp)
        if cached is not None:
            idx, rationale, conf = cached
            method = "llm-cached"
        else:
            llm = _llm_localize(steps, llm_call_fn)
            if llm is not None:
                idx, rationale, conf = llm
                method = "llm"
                _cache_put(conn, fp, idx, rationale, conf)

    step = steps[idx]
    cs = CriticalStep(
        agent_id=agent_id,
        log_position=step.position,
        tool_call_id=step.raw_id if step.kind == "tool" else None,
        rationale=rationale,
        method=method,
        confidence=conf,
        fingerprint_id=fingerprint_id,
        isc_id=isc_id,
    )
    if persist:
        _persist(conn, cs)
    return cs


# ── persistence ───────────────────────────────────────────────────


def _persist(conn: sqlite3.Connection, cs: CriticalStep) -> None:
    try:
        conn.execute(
            "INSERT INTO critical_steps "
            "(agent_id, fingerprint_id, log_position, tool_call_id, "
            "rationale, method, confidence, isc_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (cs.agent_id, cs.fingerprint_id, cs.log_position,
             cs.tool_call_id, cs.rationale, cs.method, cs.confidence,
             cs.isc_id),
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass  # pre-v8 DB without critical_steps — best-effort


def _cache_get(
    conn: sqlite3.Connection, fp: str
) -> tuple[int, str, float] | None:
    """Reuse llm_diagnosis_cache: pack (index, rationale, confidence) into
    the existing columns. category holds the int index as text."""
    try:
        row = conn.execute(
            "SELECT category, root_cause, confidence "
            "FROM llm_diagnosis_cache WHERE fingerprint = ?",
            (fp,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    try:
        return int(row[0]), row[1] or "", float(row[2] or 0.6)
    except (TypeError, ValueError):
        return None


def _cache_put(
    conn: sqlite3.Connection, fp: str, idx: int, rationale: str,
    conf: float,
) -> None:
    try:
        conn.execute(
            "INSERT OR REPLACE INTO llm_diagnosis_cache "
            "(fingerprint, category, root_cause, suggested_action, "
            "confidence, model) VALUES (?, ?, ?, ?, ?, 'localizer')",
            (fp, str(idx), rationale, None, conf),
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass


def latest_for_agent(
    conn: sqlite3.Connection, agent_id: str
) -> CriticalStep | None:
    """Return the most recently persisted critical step for an agent."""
    try:
        row = conn.execute(
            "SELECT agent_id, fingerprint_id, log_position, tool_call_id, "
            "rationale, method, confidence, isc_id "
            "FROM critical_steps WHERE agent_id = ? "
            "ORDER BY cs_id DESC LIMIT 1",
            (agent_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    return CriticalStep(
        agent_id=row[0], fingerprint_id=row[1], log_position=row[2],
        tool_call_id=row[3], rationale=row[4], method=row[5],
        confidence=row[6], isc_id=row[7],
    )
