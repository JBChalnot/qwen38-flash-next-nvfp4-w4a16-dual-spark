#!/usr/bin/env python3
"""Aggregate throughput against concurrency, 1..N streams.

The aggregate reported here is **total output tokens / wall time to completion**.

It is NOT `output / (wall - mean TTFT)`. That form leaves the OTHER requests' prefill inside
the window it calls "decode", and once manufactured a 31 -> 1.9 tok/s "collapse" that did not
exist: the machine was prefill-dominated, not slow. Before concluding that decode regressed
under load, check what fraction of the wall clock is prefill.

The whole warm-up sweep is discarded, not just the first shot of each rung.

Also reported per rung: TTFT and mean accepted length, because a concurrency change can move
either one without touching throughput.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import urllib.request


def _resultats() -> str:
    """Where JSON goes: RESULTS_DIR, else `results/` beside the repo. Never an absolute path."""
    import os
    import pathlib as _pl
    d = os.environ.get("RESULTS_DIR") or str(_pl.Path(__file__).resolve().parent.parent / "results")
    _pl.Path(d).mkdir(parents=True, exist_ok=True)
    return d

BASE = "http://127.0.0.1:8000"
UNITE = ("Le systeme enregistre des mesures de bande passante, de latence et d'acceptation "
         "speculative sur deux noeuds relies par un lien point-a-point. ")


def metrique(nom: str) -> float | None:
    t = urllib.request.urlopen(f"{BASE}/metrics", timeout=20).read().decode()
    for l in t.split("\n"):
        if l.startswith(f"vllm:{nom}") and "created" not in l:
            return float(l.rsplit(" ", 1)[1])
    return None   # absent: an UNOBSERVED state, decided by the caller, never a zero


def spec() -> tuple[float, float, float] | None:
    """The three speculation counters, or None when the engine exports NONE of them
    (a serve booted with SPEC_TOKENS=0 is a documented configuration). A PARTIAL absence is
    refused: reporting a mean accepted length from a missing counter is how a dead drafter
    goes unnoticed."""
    v = (metrique("spec_decode_num_drafts_total"),
         metrique("spec_decode_num_draft_tokens_total"),
         metrique("spec_decode_num_accepted_tokens_total"))
    if all(x is None for x in v):
        return None
    if any(x is None for x in v):
        raise SystemExit("FAIL CLOSED: only some spec_decode counters are exported; refusing to "
                         "compute a mean accepted length from a partial set.")
    return v  # type: ignore[return-value]


def une_requete(idx: int, rep: int, max_tokens: int, modele: str, t_zero: float):
    """Rend (jetons de sortie, TTFT, instant de fin)."""
    prompt = (f"Flux {idx}. " + UNITE * rep +
              "\n\nResume ce qui precede, puis enumere dix consequences pratiques.")
    corps = {"model": modele, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": max_tokens, "temperature": 0, "stream": True,
             "stream_options": {"include_usage": True},
             "chat_template_kwargs": {"enable_thinking": False}}
    d = json.dumps(corps).encode()
    premier = None
    sortie = 0
    with urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/v1/chat/completions", d, {"Content-Type": "application/json"}),
            timeout=3600) as r:
        for ligne in r:
            if not ligne.startswith(b"data: "):
                continue
            c = ligne[6:].strip()
            if c == b"[DONE]":
                break
            o = json.loads(c)
            u = o.get("usage")
            if u:
                sortie = u.get("completion_tokens") or sortie
            ch = o.get("choices") or []
            if ch and (ch[0].get("delta") or {}) and premier is None:
                premier = time.time()
    return sortie, (premier - t_zero) if premier else float("nan"), time.time()


def un_point(c: int, rep: int, max_tokens: int, modele: str):
    d0 = spec()
    t_zero = time.time()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        futs = [ex.submit(une_requete, i, rep, max_tokens, modele, t_zero) for i in range(c)]
        res = [f.result() for f in futs]
    fin = max(r[2] for r in res)
    d1 = spec()
    sortie_tot = sum(r[0] for r in res)
    mur = fin - t_zero
    if d0 is None or d1 is None:
        mal = None   # no speculation on this serve: reported as null, not as 1.0 or nan
    else:
        drafts = d1[0] - d0[0]
        mal = 1 + (d1[2] - d0[2]) / drafts if drafts else float("nan")
    return {"concurrence": c,
            "agrege_tok_s": sortie_tot / mur if mur > 0 else 0.0,
            "par_flux_tok_s": sortie_tot / mur / c if mur > 0 and c else 0.0,
            "sortie_totale": sortie_tot, "mur_s": mur,
            "ttft_mediane": statistics.median([r[1] for r in res]),
            "mal": mal}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bras", required=True)
    p.add_argument("--max", type=int, default=6, help="concurrence maximale")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--rep-prompt", type=int, default=120, help="~3200 jetons d'entrée")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--modele", default="q38")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    if urllib.request.urlopen(f"{BASE}/health", timeout=15).getcode() != 200:
        print("🔴 /health is not 200 - refusing to measure."); return 2

    print("  ── balayage de CHAUFFE, jeté (le premier balayage entier) ──")
    for c in range(1, a.max + 1):
        r = un_point(c, a.rep_prompt, a.max_tokens, a.modele)
        print(f"     c={c} agrégé {r['agrege_tok_s']:6.1f} tok/s")

    par_c = {}
    for c in range(1, a.max + 1):
        pts = [un_point(c, a.rep_prompt, a.max_tokens, a.modele) for _ in range(a.repetitions)]
        ag = [x["agrege_tok_s"] for x in pts]
        pf = [x["par_flux_tok_s"] for x in pts]
        tt = [x["ttft_mediane"] for x in pts]
        ml = [x["mal"] for x in pts if x["mal"] is not None]
        par_c[c] = {"agrege_mediane": statistics.median(ag),
                    "agrege_min": min(ag), "agrege_max": max(ag),
                    "par_flux_mediane": statistics.median(pf),
                    "ttft_mediane": statistics.median(tt),
                    "mal_mediane": statistics.median(ml) if ml else None,
                    "sortie_totale": pts[0]["sortie_totale"]}

    print(f"\n  {'c':>3}{'agrégé tok/s':>15}{'étendue':>16}{'par flux':>11}"
          f"{'TTFT s':>9}{'MAL':>7}{'échelle':>9}")
    base = par_c[1]["agrege_mediane"]
    for c, d in par_c.items():
        print(f"  {c:>3}{d['agrege_mediane']:>15.1f}{d['agrege_min']:>8.1f}-{d['agrege_max']:<7.1f}"
              f"{d['par_flux_mediane']:>11.1f}{d['ttft_mediane']:>9.2f}"
              f"{(f"{d['mal_mediane']:.3f}" if d['mal_mediane'] is not None else 'n/a'):>7}"
              f"{d['agrege_mediane']/base:>8.2f}x")

    res = {"bras": a.bras, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "repetitions": a.repetitions, "rep_prompt": a.rep_prompt,
           "max_tokens": a.max_tokens, "par_concurrence": par_c}
    out = a.json or f"{_resultats()}/concurrence-{a.bras}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    os.replace(out + ".tmp", out)
    print(f"  JSON : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
