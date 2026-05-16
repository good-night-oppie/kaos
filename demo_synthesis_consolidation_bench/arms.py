"""The 8 arms. Each, per query, returns the TEXTS of its retrieved top-K
(the context an agent would get). The blind judge then checks the frozen
decisive predicate over those texts.

Synthesis arms run the REAL feature (kaos.dream.phases.synthesis) with a
cluster-blind llm_call_fn. No arm and no synthesizer ever sees a query or
a canonical answer.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass

from kaos import Kaos
from kaos.memory import MemoryStore
from kaos.dream.phases import synthesis as synth_phase

from world import DOMAINS, canonical_rule
from queries import Query

K_RAW = 8
K_KEEP_ALL = 30   # the long-context arm gets a bigger budget, fairly


@dataclass
class Retrieved:
    texts: list[str]
    any_synth: bool


def _search(conn, query_text: str, *, limit: int,
            include_insight: bool) -> list[tuple[int, str, str]]:
    mem = MemoryStore(conn)
    try:
        # FTS-safe: OR the salient tokens (mirrors realistic UI search).
        toks = [t for t in query_text.lower().replace("?", " ").split()
                if len(t) > 2][:12]
        q = " OR ".join(toks) if toks else query_text
        res = mem.search(q, limit=limit * 3, rank="weighted")
    except Exception:
        res = []
    out = []
    for r in res:
        if not include_insight and r.type == "insight":
            continue
        out.append((r.memory_id, r.type, r.content))
        if len(out) >= limit:
            break
    return out


def _claude_cli(prompt: str, cache: dict, *, timeout: int = 90) -> str:
    """Cluster-blind synthesizer via the Claude Code CLI. Cached on disk by
    the synthesis module's fingerprint (cache dict persisted by run.py)."""
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    try:
        p = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return (p.stdout or "").strip()
    except Exception:
        return ""


def _run_synthesis(db_path: str, cache: dict, *,
                   index_insights: bool, mask_spans=None) -> None:
    k = Kaos(db_path=db_path)
    try:
        synth_phase.run(
            k.conn,
            llm_call_fn=lambda pr: _claude_cli(pr, cache),
            cache=cache,
            index_insights=index_insights,
            mask_spans=mask_spans,
        )
    finally:
        k.close()


# ── arm retrievers ────────────────────────────────────────────────


def retrieve(arm: str, db_path: str, q: Query) -> Retrieved:
    k = Kaos(db_path=db_path)
    try:
        conn = k.conn
        if arm == "B2":
            # keep-everything: budgeted, position-randomized dump.
            rows = conn.execute(
                "SELECT memory_id, content FROM memory "
                "WHERE type != 'insight'"
            ).fetchall()
            rng = random.Random(hash(q.qid) & 0xffffffff)
            rng.shuffle(rows)
            texts = [c for _i, c in rows[:K_KEEP_ALL]]
            return Retrieved(texts, any_synth=False)

        if arm == "B3":
            # cost-matched: deterministic query expansion (no LLM), token
            # budget comparable to a synthesis insight (~2-3 sentences).
            d = next((dm for dm in DOMAINS if dm in q.text), None)
            expand = ""
            if d:
                expand = " " + " ".join(DOMAINS[d]["services"]
                                        + DOMAINS[d]["symptoms"])
            r = _search(conn, q.text + expand, limit=K_RAW,
                        include_insight=False)
            return Retrieved([t for _i, _ty, t in r], any_synth=False)

        if arm in ("B0", "B1"):
            r = _search(conn, q.text, limit=K_RAW, include_insight=False)
            texts = [t for _i, _ty, t in r]
            if arm == "B1":
                # non-LLM dedup: drop near-duplicate texts (Jaccard>0.8).
                from kaos.dream.phases.consolidation import _jaccard
                kept: list[str] = []
                for t in texts:
                    ts = set(t.lower().split())
                    if all(_jaccard(ts, set(x.lower().split())) <= 0.8
                           for x in kept):
                        kept.append(t)
                texts = kept
            return Retrieved(texts, any_synth=False)

        if arm in ("FULL", "L1", "L2", "L3"):
            inc_ins = arm in ("FULL", "L2", "L3")
            r = _search(conn, q.text, limit=K_RAW, include_insight=inc_ins)
            texts = [t for _i, _ty, t in r]
            any_s = any(ty == "insight" for _i, ty, _t in
                        _search(conn, q.text, limit=K_RAW,
                                include_insight=inc_ins))
            if arm == "L2":
                # append a synthesized insight from an UNRELATED cluster.
                ins = conn.execute(
                    "SELECT content FROM memory WHERE type='insight' "
                    "ORDER BY memory_id DESC LIMIT 1"
                ).fetchone()
                if ins:
                    texts = texts + [ins[0]]
                    any_s = True
            return Retrieved(texts, any_synth=any_s)

        raise ValueError(arm)
    finally:
        k.close()


def retrieve_all(arm: str, db_path: str,
                 queries: list[Query]) -> list[Retrieved]:
    """Open the arm's DB ONCE, retrieve for every query. Much faster than
    per-query opens; identical results."""
    k = Kaos(db_path=db_path)
    out: list[Retrieved] = []
    try:
        conn = k.conn
        all_rows = None
        if arm == "B2":
            all_rows = conn.execute(
                "SELECT memory_id, content FROM memory "
                "WHERE type != 'insight'"
            ).fetchall()
        for q in queries:
            if arm == "B2":
                rng = random.Random(hash(q.qid) & 0xffffffff)
                rows = list(all_rows)
                rng.shuffle(rows)
                out.append(Retrieved([c for _i, c in rows[:K_KEEP_ALL]],
                                      any_synth=False))
                continue
            if arm == "B3":
                d = next((dm for dm in DOMAINS if dm in q.text), None)
                expand = (" " + " ".join(DOMAINS[d]["services"]
                                         + DOMAINS[d]["symptoms"])) if d else ""
                r = _search(conn, q.text + expand, limit=K_RAW,
                            include_insight=False)
                out.append(Retrieved([t for _i, _ty, t in r],
                                      any_synth=False))
                continue
            if arm in ("B0", "B1"):
                r = _search(conn, q.text, limit=K_RAW,
                            include_insight=False)
                texts = [t for _i, _ty, t in r]
                if arm == "B1":
                    from kaos.dream.phases.consolidation import _jaccard
                    kept: list[str] = []
                    for t in texts:
                        ts = set(t.lower().split())
                        if all(_jaccard(ts, set(x.lower().split())) <= 0.8
                               for x in kept):
                            kept.append(t)
                    texts = kept
                out.append(Retrieved(texts, any_synth=False))
                continue
            # FULL / L1 / L2 / L3
            inc_ins = arm in ("FULL", "L2", "L3")
            r = _search(conn, q.text, limit=K_RAW, include_insight=inc_ins)
            texts = [t for _i, _ty, t in r]
            any_s = any(ty == "insight" for _i, ty, _t in r)
            if arm == "L2":
                ins = conn.execute(
                    "SELECT content FROM memory WHERE type='insight' "
                    "ORDER BY memory_id DESC LIMIT 1"
                ).fetchone()
                if ins:
                    texts = texts + [ins[0]]
                    any_s = True
            out.append(Retrieved(texts, any_synth=any_s))
        return out
    finally:
        k.close()


def prepare_synthesis_arms(db_paths: dict[str, str], cache: dict) -> None:
    """Run the real synthesis feature into the synth arms' DBs. Each arm
    has its OWN forked DB so arms don't contaminate each other.

    FULL/L2: insights indexed.  L1: insights written but NOT indexed.
    L3: answer-masked (canonical rule PHRASES masked from sources; by
    world construction those phrases never appear in a single source, so
    masking is a no-op iff there is no verbatim-answer to steal — the
    leakage control)."""
    mask = []
    for dm in DOMAINS:
        for resolved in (True, False):
            _toks, phrase = canonical_rule(dm, resolved=resolved)
            mask.append(phrase)
    _run_synthesis(db_paths["FULL"], cache, index_insights=True)
    _run_synthesis(db_paths["L1"], cache, index_insights=False)
    _run_synthesis(db_paths["L2"], cache, index_insights=True)
    _run_synthesis(db_paths["L3"], cache, index_insights=True,
                   mask_spans=mask)
