"""Synthesis-as-consolidation (v0.9 candidate — UNDER EVALUATION).

Clusters Hebbian-linked episodic memories and asks an LLM to synthesize a
single generalized `insight` memory per cluster, linked back to its
sources so weighted retrieval can surface the abstraction.

Honest-test invariants enforced here:
  - The synthesizer sees ONLY the cluster's memory texts. It is never
    given queries, canonical answers, or rule-token lists.
  - Deterministic clustering core; the LLM is an opt-in `llm_call_fn`.
  - Fingerprint-cached by cluster shape so a stable cluster pays the LLM
    cost at most once.

Whether this feature survives its pre-registered gates is decided by
demo_synthesis_consolidation_bench/, not by this module.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_MIN_CLUSTER = 4
DEFAULT_EDGE_WEIGHT = 0.0   # any positive Hebbian edge counts


@dataclass
class SynthesisReport:
    clusters_found: int = 0
    insights_written: int = 0
    insight_memory_ids: list[int] = field(default_factory=list)
    llm_calls: int = 0
    llm_failures: int = 0
    cache_hits: int = 0


def _components(conn: sqlite3.Connection,
                min_w: float) -> list[list[int]]:
    """Connected components over memory<->memory association edges."""
    try:
        rows = conn.execute(
            "SELECT id_a, id_b FROM associations "
            "WHERE kind_a='memory' AND kind_b='memory' AND weight > ?",
            (min_w,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    adj: dict[int, set[int]] = {}
    for a, b in rows:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for n in adj:
        if n in seen:
            continue
        stack, comp = [n], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj.get(x, ()))
        comps.append(comp)
    return comps


_PROMPT = (
    "You are consolidating an agent's episodic memory. Below are related "
    "incident notes from one cluster. Write ONE general, reusable insight "
    "(2-3 sentences) stating the recurring underlying condition or pattern "
    "across these incidents. Be concrete and use the exact technical terms "
    "that appear in the notes. If the notes share no real pattern, reply "
    "exactly NONE.\n\nNOTES:\n{notes}\n\nINSIGHT:"
)


def _fingerprint(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in sorted(texts):
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return "synth:" + h.hexdigest()[:20]


def run(
    conn: sqlite3.Connection,
    *,
    llm_call_fn: Callable[[str], str] | None,
    cache: dict[str, str] | None = None,
    min_cluster: int = DEFAULT_MIN_CLUSTER,
    min_edge_weight: float = DEFAULT_EDGE_WEIGHT,
    mask_spans: list[str] | None = None,
    index_insights: bool = True,
) -> SynthesisReport:
    """Synthesize one insight per qualifying cluster.

    mask_spans: substrings stripped from source texts BEFORE the LLM sees
      them (the L3 answer-masked lesion). Never used by the FULL arm.
    index_insights: if False, insights are written but NOT linked into
      associations / left findable (the L1 lesion).
    cache: optional persistent {fingerprint: insight_text} map.
    """
    rep = SynthesisReport()
    cache = cache if cache is not None else {}
    comps = _components(conn, min_edge_weight)
    rep.clusters_found = sum(1 for c in comps if len(c) >= min_cluster)

    from kaos.memory import MemoryStore
    mem = MemoryStore(conn)

    for comp in comps:
        if len(comp) < min_cluster:
            continue
        # Pull member texts (the ONLY thing the synthesizer sees).
        qs = ",".join("?" * len(comp))
        rows = conn.execute(
            f"SELECT memory_id, content FROM memory WHERE memory_id IN ({qs})",
            comp,
        ).fetchall()
        texts = []
        for _mid, content in rows:
            t = content or ""
            for span in (mask_spans or []):
                if span:
                    t = t.replace(span, "[MASKED]")
            texts.append(t)
        if not texts:
            continue

        fp = _fingerprint(texts)
        insight = cache.get(fp)
        if insight is not None:
            rep.cache_hits += 1
        else:
            if llm_call_fn is None:
                continue
            prompt = _PROMPT.format(notes="\n".join(f"- {t}" for t in texts[:40]))
            rep.llm_calls += 1
            try:
                raw = llm_call_fn(prompt)
            except Exception:
                rep.llm_failures += 1
                continue
            if not raw or raw.strip().upper().startswith("NONE"):
                continue
            insight = raw.strip()
            cache[fp] = insight

        # Persist as an `insight` memory authored by consolidation.
        owner = conn.execute(
            "SELECT agent_id FROM memory WHERE memory_id = ? LIMIT 1",
            (comp[0],),
        ).fetchone()
        agent_id = owner[0] if owner else "consolidation"
        mid = mem.write(agent_id=agent_id, content=insight,
                        type="insight", key=fp)
        rep.insights_written += 1
        rep.insight_memory_ids.append(mid)

        if index_insights:
            # Link insight <-> each source so weighted retrieval can reach
            # it via the same cluster the queries co-activate.
            now = "strftime('%Y-%m-%dT%H:%M:%f','now')"
            for src in comp:
                try:
                    conn.execute(
                        f"INSERT INTO associations (kind_a,id_a,kind_b,id_b,"
                        f"weight,uses,first_seen,last_seen) VALUES "
                        f"('memory',?,'memory',?,3.0,1,{now},{now}) "
                        f"ON CONFLICT(kind_a,id_a,kind_b,id_b) DO UPDATE SET "
                        f"weight=weight+1.0,last_seen={now}",
                        (mid, src),
                    )
                except sqlite3.OperationalError:
                    pass
        conn.commit()
    return rep
