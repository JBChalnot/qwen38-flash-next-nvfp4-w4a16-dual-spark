#!/usr/bin/env python3
"""Decode throughput AND mean accepted length, from the SAME run.

Why both from one run: on a speculative decoder the two move together and reading them from
different runs has produced contradictory conclusions here. Throughput alone cannot tell a
slow kernel from an acceptance collapse — a dead drafter looks exactly like a slow engine.

Why four content types: the spread between them reaches 47 % on this model (prose is the
slowest, repetitive text the fastest). A single-content benchmark on a speculative decoder
measures the content, not the engine.

The first sequence after a boot is warm-up and is discarded.

Output: median AND range per content type. Never a single shot — at temperature 0.7 the real
spread of one task reaches 185 %, which measures the length of the answer and nothing else.
"""
from __future__ import annotations

import argparse
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

# Quatre contenus, parce que l'acceptation spéculative en dépend fortement.
CONTENUS = {
    "prose": "Explique en detail, en prose continue et sans listes, comment un lien RoCE "
             "point-a-point entre deux machines a memoire unifiee change le dimensionnement "
             "d'un cache de cles-valeurs. Developpe au moins six paragraphes.",
# ⚠️ THE BENCHMARK PROMPTS BELOW STAY IN FRENCH, ON PURPOSE. Translating them would
# change the measurement: different tokens, different tokenizer fertility (3.258 vs
# 3.116 chars/token between two models we compare = 4.5 % of unit), different answer
# lengths. Every number published for this recipe was taken with THESE strings. If you
# translate them you have a new instrument, not the same one, and you must re-baseline.
    "code": "Ecris une fonction Python complete qui lit les en-tetes de plusieurs fichiers "
            "safetensors, agrege les octets de donnees par famille de tenseurs a partir d'une "
            "expression reguliere fournie, et rend un tableau trie. Commente chaque etape.",
    "structure": "Rends un objet JSON valide decrivant huit mesures de performance, chacune "
                 "avec les champs nom, valeur, unite, methode, et une liste de trois reserves. "
                 "Uniquement du JSON, rien avant, rien apres.",
    "repetitif": "Enumere les entiers de 1 a 120, un par ligne, au format 'ligne N : valeur N', "
                 "sans commentaire ni introduction.",
}


def metrique(nom: str) -> float:
    t = urllib.request.urlopen(f"{BASE}/metrics", timeout=20).read().decode()
    for l in t.split("\n"):
        if l.startswith(f"vllm:{nom}") and "created" not in l:
            return float(l.rsplit(" ", 1)[1])
    # ⚠️ FAIL CLOSED, and this line is the whole point. Returning 0.0 for an ABSENT metric
    # made the mean accepted length come out as nan and the tool exit 0 -- so the single
    # control that catches a silently dead drafter could not fail. An absent metric is an
    # UNOBSERVED state, not a healthy one.
    raise SystemExit(f"FAIL CLOSED: metric vllm:{nom} is absent from /metrics. Speculation "
                     f"may be disabled, or this engine does not export it. Refusing to "
                     f"report a mean accepted length computed from a missing counter.")


def spec() -> tuple[float, float, float]:
    return (metrique("spec_decode_num_drafts_total"),
            metrique("spec_decode_num_draft_tokens_total"),
            metrique("spec_decode_num_accepted_tokens_total"))


def tir(prompt: str, modele: str, max_tokens: int) -> tuple[float, int, float]:
    """Rend (débit de décode tok/s, jetons de sortie, mur total)."""
    req = {"model": modele, "messages": [{"role": "user", "content": prompt}],
           "max_tokens": max_tokens, "temperature": 0,
           "chat_template_kwargs": {"reasoning_effort": "low"}, "stream": True,
           "stream_options": {"include_usage": True}}
    d = json.dumps(req).encode()
    t0 = time.time()
    premier = None
    sortie = 0
    with urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/v1/chat/completions", d, {"Content-Type": "application/json"}),
            timeout=1800) as r:
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
            if ch and (ch[0].get("delta") or {}):
                if premier is None:
                    premier = time.time()
    fin = time.time()
    # On ne soustrait PAS une TTFT moyenne d'un mur agrégé. Ici on est en
    # série à concurrence 1, donc (fin − premier jeton) EST la fenêtre de décode de CETTE
    # requête, sans prefill d'autrui dedans.
    fenetre = fin - (premier or t0)
    return (sortie / fenetre if fenetre > 0 else 0.0), sortie, fin - t0


def main() -> int:
    MORT = False   # zero-acceptance verdict; must reach the exit code, not just stdout
    p = argparse.ArgumentParser()
    p.add_argument("--bras", required=True)
    p.add_argument("--tirs", type=int, default=5, help="tirs RETENUS par contenu (+1 jeté)")
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--modele", default="q38")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    if urllib.request.urlopen(f"{BASE}/health", timeout=15).getcode() != 200:
        print("🔴 /health is not 200 - refusing to measure."); return 2

    d0, t0, a0 = spec()
    par_contenu = {}
    for nom, prompt in CONTENUS.items():
        tir(prompt, a.modele, a.max_tokens)          # le premier est jeté
        v, s = [], []
        for _ in range(a.tirs):
            deb, sortie, _mur = tir(prompt, a.modele, a.max_tokens)
            v.append(deb); s.append(sortie)
        par_contenu[nom] = {"debit_mediane": statistics.median(v),
                            "debit_min": min(v), "debit_max": max(v),
                            "sortie_mediane": statistics.median(s)}
        print(f"  {nom:<11} {statistics.median(v):6.2f} tok/s  "
              f"[{min(v):.2f}-{max(v):.2f}]  sortie {statistics.median(s):.0f}")
    d1, t1, a1 = spec()

    drafts, proposes, acceptes = d1 - d0, t1 - t0, a1 - a0
    mal = 1 + acceptes / drafts if drafts else float("nan")
    taux = acceptes / proposes * 100 if proposes else float("nan")
    tous = [par_contenu[n]["debit_mediane"] for n in par_contenu]
    print()
    print(f"  débit, médiane des contenus : {statistics.median(tous):.2f} tok/s  "
          f"[{min(tous):.2f}-{max(tous):.2f}]")
    print(f"  MAL  : {mal:.3f}   ({int(drafts)} brouillons, {int(proposes)} proposés, "
          f"{int(acceptes)} acceptés, taux {taux:.1f} %)")
    if drafts and acceptes == 0:
        print("  🔴 ZERO tokens accepted: speculation is DEAD. A low throughput then reads "
              "like a slow kernel when the cause is zero acceptance. See docs/MODS.md.")
        MORT = True

    res = {"bras": a.bras, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "tirs_par_contenu": a.tirs, "max_tokens": a.max_tokens,
           "par_contenu": par_contenu,
           "debit_mediane_globale": statistics.median(tous),
           "mal": mal, "taux_acceptation_pct": taux,
           "brouillons": drafts, "proposes": proposes, "acceptes": acceptes}
    out = a.json or f"{_resultats()}/decode-mal-{a.bras}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    os.replace(out + ".tmp", out)
    print(f"  JSON : {out}")
    # A dead drafter is a FAILED measurement, not a successful one. Exit 3 so a caller in a
    # shell pipeline cannot mistake it for a healthy run.
    return 3 if MORT else 0


if __name__ == "__main__":
    sys.exit(main())
