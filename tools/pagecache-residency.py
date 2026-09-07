#!/usr/bin/env python3
"""How many bytes of the CHECKPOINT are resident in the page cache? (mmap + mincore)

Why not MemFree: measured after a boot it varies by +/-1.3 GiB between identical boots (logs,
docker, shells). For a lever whose expected effect is "give the checkpoint's pages back",
MemFree mostly measures what everything ELSE put in the cache.

The clean observable is the residency of the CHECKPOINT's pages, read with mincore(2): exactly
what a per-shard POSIX_FADV_DONTNEED is meant to drive to zero, and nothing else moves it.

This only READS, and never calls read() on the data: mmap + mincore bring no page in.
Measuring it does not change it.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import mmap
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

PAGE = os.sysconf("SC_PAGE_SIZE")
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def residence(chemin: str) -> tuple[int, int]:
    """(resident bytes, total bytes) for one file, without reading it."""
    taille = os.path.getsize(chemin)
    if taille == 0:
        return 0, 0
    fd = os.open(chemin, os.O_RDONLY)
    try:
        # MAP_PRIVATE + PROT_WRITE : `ctypes.from_buffer` exige un tampon INSCRIPTIBLE pour
        # rendre l'adresse. En MAP_PRIVATE toute écriture serait une copie-à-l'écriture
        # privée — et on n'écrit jamais. Le fichier n'est pas touché, et `mincore` rend bien
        # la résidence des pages DU FICHIER.
        mm = mmap.mmap(fd, taille, flags=mmap.MAP_PRIVATE,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    try:
        npages = (taille + PAGE - 1) // PAGE
        vec = ctypes.create_string_buffer(npages)
        adresse = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        if libc.mincore(ctypes.c_void_p(adresse), ctypes.c_size_t(taille), vec) != 0:
            raise SystemExit(f"FAIL CLOSED : mincore a échoué sur {chemin} : "
                             f"{os.strerror(ctypes.get_errno())}")
        # `vec.raw` est un bytes : chaque octet a le bit 0 à 1 si la page est résidente
        resident = sum(1 for o in vec.raw[:npages] if o & 1)
        return resident * PAGE, taille
    finally:
        mm.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--motif", default="*.safetensors")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    a.dir = a.dir or os.environ.get("CHECKPOINT_DIR") or _exige("CHECKPOINT_DIR", "--dir")

    fichiers = sorted(glob.glob(os.path.join(a.dir, a.motif)))
    if not fichiers:
        raise SystemExit(f"FAIL CLOSED : aucun fichier {a.motif} dans {a.dir}")
    G = 2 ** 30
    tr = tt = 0
    for f in fichiers:
        r, t = residence(f)
        tr += r
        tt += t
        if a.detail:
            print(f"  {os.path.basename(f):<40} {r/G:7.2f} / {t/G:7.2f} GiB "
                  f"({r/t*100 if t else 0:5.1f} %)")
    print(f"  {len(fichiers)} shard(s) · resident in page cache: "
          f"**{tr/G:.2f} GiB** of {tt/G:.2f} ({tr/tt*100:.1f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
