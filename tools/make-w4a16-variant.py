#!/usr/bin/env python3
"""Build the W4A16 variant of the NVIDIA checkpoint. A LABEL change, not a requantization.

vLLM's modelopt path states it: NVFP4 (W4A4) and W4A16_NVFP4 share the same packing. Flipping
`quant_algo` from `NVFP4` to `W4A16_NVFP4` therefore touches **no weight bit**: it changes the
linear method that gets elected, and for the fused MoE it sets the 16-bit-activation path.

The only lever that works, and two proven no-ops to avoid:
  - `hf_quant_config.json` is read ONLY when `config.json` has no `quantization_config`.
    NVIDIA's has one, so editing that file is a silent no-op.
  - the mixed path never reads `config_groups` or `input_activations` either.
The lever is `config.json -> quantization_config.quantized_layers`, and nothing else.

Why RELATIVE symlinks: the weights are used as-is, byte for byte. The variant contains one
rewritten config.json; everything else is a relative link, so the directory resolves both on
the host and inside the container, where the tree is mounted under another prefix.

Fails closed: refuses unless the switched count is exactly what is expected, refuses an NVFP4
entry that is not a routed-expert family, refuses if FP8 / FP8_BLOCK_SCALES counts change,
refuses a dangling link, and refuses destination == source or either nested in the other --
that last guard exists because without it `--dest <the source> --refaire` deleted the weights.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def _exige(nom: str, option: str = "") -> str:
    """No path from the machine this was written on survives here: the variable is REQUIRED.

    Called AFTER argparse, never as an argparse default: a default is evaluated at import
    time, which would make the very flag suggested in the message impossible to use.
    """
    import sys
    ou = f" or pass {option}" if option else ""
    sys.exit(f"FAIL CLOSED: {nom} is not set. Do `set -a; . recipe.env; set +a`{ou}.")

ATTENDU = 48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--attendu", type=int, default=ATTENDU)
    ap.add_argument("--refaire", action="store_true")
    a = ap.parse_args()
    a.source = a.source or os.environ.get("CHECKPOINT_DIR") or _exige("CHECKPOINT_DIR", "--source")
    a.dest = a.dest or os.environ.get("SERVED_DIR") or _exige("SERVED_DIR", "--dest")

    src, dst = pathlib.Path(a.source), pathlib.Path(a.dest)
    if not (src / "config.json").is_file():
        sys.exit(f"FAIL CLOSED : {src}/config.json introuvable")
    if dst.exists() and not a.refaire:
        sys.exit(f"FAIL CLOSED : {dst} existe déjà (--refaire pour reconstruire)")

    # GARDE. Sans elle,
    # `--dest <la source> --refaire` faisait `cible.unlink()` sur les poids EUX-MÊMES :
    # 123 GiB détruits, et 3,3 h de re-téléchargement à 206 Mb/s. Le `resolve()` est
    # indispensable — un lien symbolique ou un `..` suffisait à contourner une égalité
    # de chaînes. On refuse aussi l'imbrication dans les deux sens.
    rs, rd = src.resolve(), dst.resolve()
    if rs == rd:
        sys.exit(f"FAIL CLOSED : destination == source ({rs}). Ce script SUPPRIME des "
                 "entries in the destination: it would have destroyed the checkpoint.")
    if rs in rd.parents or rd in rs.parents:
        sys.exit(f"FAIL CLOSED : {rd} et {rs} sont imbriqués — refus, le nettoyage de la "
                 "destination toucherait la source.")

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    qc = cfg.get("quantization_config")
    if not qc:
        sys.exit("FAIL CLOSED : config.json sans `quantization_config`")
    ql = qc.get("quantized_layers")
    if not ql:
        sys.exit("FAIL CLOSED : `quantized_layers` absent — ce n'est pas un MIXED_PRECISION")

    avant = {}
    for k, v in ql.items():
        avant.setdefault(v.get("quant_algo"), []).append(k)
    print("  état AVANT :")
    for algo, ks in sorted(avant.items()):
        print(f"    {algo:<20} {len(ks):>3} entrées   ex. {ks[0]}")

    cibles = avant.get("NVFP4", [])
    if len(cibles) != a.attendu:
        sys.exit(f"FAIL CLOSED : {len(cibles)} entrées NVFP4, {a.attendu} attendues. "
                 "This is not the checkpoint this script knows how to handle.")
    for k in cibles:
        if not k.endswith(".mlp.experts"):
            sys.exit(f"FAIL CLOSED : entrée NVFP4 inattendue `{k}` — ce script ne bascule "
                     "que les familles d'experts routés.")
        ql[k]["quant_algo"] = "W4A16_NVFP4"

    apres = {}
    for k, v in ql.items():
        apres.setdefault(v.get("quant_algo"), []).append(k)
    print("  état APRÈS :")
    for algo, ks in sorted(apres.items()):
        print(f"    {algo:<20} {len(ks):>3} entrées")

    intacts = {"FP8", "FP8_BLOCK_SCALES"}
    for algo in intacts:
        if len(avant.get(algo, [])) != len(apres.get(algo, [])):
            sys.exit(f"FAIL CLOSED : {algo} a changé de compte — effet de bord interdit")
    if "NVFP4" in apres:
        sys.exit("FAIL CLOSED : des entrées NVFP4 subsistent")
    if len(apres.get("W4A16_NVFP4", [])) != a.attendu:
        sys.exit("FAIL CLOSED : compte W4A16_NVFP4 incorrect après bascule")

    dst.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, dst)
    liens = 0
    for p in sorted(src.iterdir()):
        if p.name == "config.json" or p.name.endswith(".part"):
            continue
        cible = dst / p.name
        if cible.is_symlink() or cible.exists():
            cible.unlink()
        cible.symlink_to(os.path.join(rel, p.name))
        liens += 1

    tmp = dst / "config.json.tmp"
    tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(dst / "config.json")

    morts = [p.name for p in dst.iterdir() if p.is_symlink() and not p.resolve().exists()]
    if morts:
        sys.exit(f"FAIL CLOSED : {len(morts)} lien(s) mort(s), ex. {morts[0]}")
    print(f"\n  ✅ {dst}")
    print(f"     {liens} liens relatifs vivants + 1 config.json réécrit, "
          f"{a.attendu} familles d'experts en W4A16_NVFP4")
    print("     aucun octet de poids copié ni modifié")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
