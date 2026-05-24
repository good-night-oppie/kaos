"""Bootstrap CI helpers for the harness.

bootstrap_diff_ci is the only inferential statistic any KAOS gate uses.
Keeping it in one place — and seeded — guarantees that "lo > 0" means
the same thing across every probe.
"""

from __future__ import annotations

import random


def bootstrap_diff_ci(
    a: list[int],
    b: list[int],
    *,
    iters: int = 2000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Two-sample bootstrap CI for mean(a) - mean(b).

    a, b are 0/1 label lists (e.g. correctness per query). Returns
    ``(mean_diff, lo, hi)`` for a symmetric (1-alpha) CI. Empty inputs
    return ``(0.0, 0.0, 0.0)`` so callers can treat empty arms as
    "no claim". The seed is fixed so the verdict is reproducible.
    """
    if not a or not b:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n_a, n_b = len(a), len(b)
    diffs: list[float] = []
    for _ in range(iters):
        ra = sum(rng.choice(a) for _ in range(n_a)) / n_a
        rb = sum(rng.choice(b) for _ in range(n_b)) / n_b
        diffs.append(ra - rb)
    diffs.sort()
    md = sum(a) / n_a - sum(b) / n_b
    lo_idx = int((alpha / 2.0) * len(diffs))
    hi_idx = int((1.0 - alpha / 2.0) * len(diffs))
    lo = diffs[lo_idx]
    hi = diffs[min(hi_idx, len(diffs) - 1)]
    return md, lo, hi
