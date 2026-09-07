#!/usr/bin/env python3
"""Give the CHECKPOINT's page cache back to the system, after the boot.

Why this instead of patching the loader: the default safetensors path is **mmap**, so the
tensors ARE the page cache. Dropping "as soon as the tensors are consumed" is either a no-op
(the kernel will not evict a mapped page) or a silent re-read — in the one file where a
mistake never faults on unified memory. AFTER the boot the question disappears: the weights
are resident engine-side and no checkpoint page is useful any more. A fadvise(DONTNEED) is
then safe, needs no privilege, and returns the same bytes.

Measured here: 10.29 GiB on the head and 16.71 GiB on the worker, just after a boot.

What it changes: MemAvailable barely moves (the cache already counted there). **MemFree**
does, and MemFree is what matters twice here — vLLM's startup guard reads it, and the CUDA
driver allocates against FREE pages, not reclaimable ones.

Never confuse this with a global drop_caches run BEFORE a boot: this one is targeted,
unprivileged, and runs AFTER. It also never calls sync(): a global sync costs in proportion to
dirty pages and has cost 18.7 -> 1.6 tok/s during a download.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import sys


def _exige(nom: str, option: str = "") -> str:
    """No path from the machine this was written on survives here: the variable is REQUIRED.

    Called AFTER argparse, never as an argparse default: a default is evaluated at import
    time, which would make the very flag suggested in the message impossible to use.
    """
    import sys
    ou = f" or pass {option}" if option else ""
    sys.exit(f"FAIL CLOSED: {nom} is not set. Do `set -a; . recipe.env; set +a`{ou}.")

POSIX_FADV_DONTNEED = 4
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_int]


def larguer(chemin: str) -> None:
    fd = os.open(chemin, os.O_RDONLY)
    try:
        rc = libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        if rc != 0:
            raise SystemExit(f"FAIL CLOSED : posix_fadvise a rendu {rc} sur {chemin}")
    finally:
        os.close(fd)


def memfree() -> tuple[float, float]:
    d = {}
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for l in fh:
            p = l.split()
            if p[0] in ("MemFree:", "MemAvailable:"):
                d[p[0]] = int(p[1]) / 1048576
    return d["MemFree:"], d["MemAvailable:"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--motif", default="*.safetensors")
    a = ap.parse_args()
    a.dir = a.dir or os.environ.get("CHECKPOINT_DIR") or _exige("CHECKPOINT_DIR", "--dir")

    fichiers = sorted(glob.glob(os.path.join(a.dir, a.motif)))
    if not fichiers:
        raise SystemExit(f"FAIL CLOSED : aucun fichier {a.motif} dans {a.dir}")
    f0, a0 = memfree()
    for f in fichiers:
        larguer(f)
    f1, a1 = memfree()
    print(f"  {len(fichiers)} shard(s) · MemFree {f0:.2f} -> {f1:.2f} GiB "
          f"(+{f1-f0:.2f}) · MemAvailable {a0:.2f} -> {a1:.2f} GiB (+{a1-a0:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
