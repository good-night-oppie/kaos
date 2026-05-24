"""Hash-locked manifest loader — tamper-evidence for pre-registration.

A probe ships an ISA.lock.json (the pre-registered protocol) plus a
small allow-list of sha256 hashes encoding the commit(s) at which that
lock was registered. Any edit to the lock after that commit changes
its hash and load_lock refuses to run — closing the goalpost-move
hole. The only legitimate way to amend a lock is a new commit + a new
hash entry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


class LockTamperError(RuntimeError):
    """Raised when a lock file's sha256 is not in the pre-registered set."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_lock(
    path: str | Path,
    known_sha256: Mapping[str, str],
) -> dict:
    """Read a JSON lock file, verifying its sha256 is pre-registered.

    Parameters
    ----------
    path
        Path to the ISA.lock.json (or equivalent).
    known_sha256
        Mapping ``{sha256_hex: label}`` of accepted hashes. ``label`` is
        free text shown in audit (e.g. ``"v1"``, ``"v2-manifests-filled"``).

    Raises
    ------
    LockTamperError
        If the on-disk hash is not in ``known_sha256``. The harness must
        refuse to compute a verdict in this case; that is the entire
        tamper-evidence contract.
    """
    p = Path(path)
    h = sha256_file(p)
    if h not in known_sha256:
        raise LockTamperError(
            f"[VOID: lock-tamper] {p.name} sha256={h} is not in the "
            f"pre-registered set {set(known_sha256)}. A changed lock "
            f"after results are seen is a goalpost move; supersede by a "
            f"new pre-registration commit + a new hash entry instead."
        )
    return json.loads(p.read_text())
