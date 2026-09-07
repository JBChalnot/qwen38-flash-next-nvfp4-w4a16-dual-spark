#!/usr/bin/env python3
"""Logprob MAE against the engine's OWN noise. The control for a kernel change.

The server is NOT deterministic at temperature=0 (exact logprob ties plus numeric drift), so a
kernel cannot be validated by output identity. It is validated by comparing its logprob
deviation to the engine's self-noise.

That noise depends on LENGTH, and the two channels must never be aggregated:
  - short prompts (11-89 tokens): self-noise is exactly 0.000000 across repeated captures, so
    any deviation there is pure configuration difference and an ABSOLUTE threshold applies;
  - long prompts (~30,000 tokens): self-noise is 0.19-0.40 nat and the server disagrees with
    itself on the top-1 token for identical input. Its cause is the long prefill (reduction
    order), not sampling. There the deviation is compared to the noise.
Aggregating the two drowns the only clean channel.

To force a cold prefill on IDENTICAL tokens, use `cache_salt`: it changes the cache key without
touching a token. Without it two captures of the same prompt return 0.000000 BY CACHE HIT and
the test cannot be failed.

⚠️ WHAT THIS TOOL DOES NOT DO. It measures whether the arithmetic is what you think. It does
NOT measure quality. A build whose gates were all green here was later rejected on an external
agentic benchmark. And on a deliberate precision change — fp8 KV, say — this tool WILL report
a structural deviation, correctly: use it to see how much your configuration moves, not to be
told it is fine.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request
import uuid

import os as _os


def _resultats() -> str:
    """Where JSON goes: RESULTS_DIR, else `results/` beside the repo. Never an absolute path."""
    import os
    import pathlib as _pl
    d = os.environ.get("RESULTS_DIR") or str(_pl.Path(__file__).resolve().parent.parent / "results")
    _pl.Path(d).mkdir(parents=True, exist_ok=True)
    return d
_M = _os.environ.get("BANC_MODELE", "q38")
_U = _os.environ.get("BANC_URL", "http://127.0.0.1:8000")

QUAL = pathlib.Path(_resultats())

# Quatre types de contenu. Le texte est neutre : on mesure des numériques,
# pas une compétence.
GRAINES = {
# ⚠️ THE BENCHMARK PROMPTS BELOW STAY IN FRENCH, ON PURPOSE. Translating them would
# change the measurement: different tokens, different tokenizer fertility (3.258 vs
# 3.116 chars/token between two models we compare = 4.5 % of unit), different answer
# lengths. Every number published for this recipe was taken with THESE strings. If you
# translate them you have a new instrument, not the same one, and you must re-baseline.
    "prose": (
        "La bibliothèque municipale occupait l'ancien entrepôt à grains, dont les poutres "
        "portaient encore les marques des sangles. Le personnel y avait installé des rayonnages "
        "bas pour ne pas masquer les fenêtres hautes, et la lumière de fin d'après-midi "
        "traversait la salle de lecture en diagonale. "
    ),
    "code": (
        "def fusionner(intervalles):\n"
        "    intervalles = sorted(intervalles)\n"
        "    sortie = []\n"
        "    for debut, fin in intervalles:\n"
        "        if sortie and debut <= sortie[-1][1]:\n"
        "            sortie[-1][1] = max(sortie[-1][1], fin)\n"
        "        else:\n"
        "            sortie.append([debut, fin])\n"
        "    return sortie\n\n"
    ),
    "structure": (
        '{"poste": "convoyeur B", "releves": [{"heure": "06:15", "debit": 412, "unite": "t/h"}, '
        '{"heure": "06:30", "debit": 408, "unite": "t/h"}], "operateur": "quart 1", '
        '"anomalies": [], "visa": null}\n'
    ),
    "repetitif": "point de contrôle conforme · point de contrôle conforme · ",
}

QUESTIONS = {
    "prose": "En un mot, quel bâtiment abritait la bibliothèque ?",
    "code": "En un mot, comment s'appelle la fonction ?",
    "structure": "En un mot, quel poste est relevé ?",
    "repetitif": "En un mot, quel est le verdict répété ?",
}


def poste(url: str, chemin: str, corps: dict, delai: int = 900) -> dict:
    r = urllib.request.Request(
        url + chemin,
        data=json.dumps(corps).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=delai) as f:
        return json.load(f)


def compte_jetons(url: str, texte: str) -> int:
    return poste(url, "/tokenize", {"model": _M, "prompt": texte})["count"]


def construis(url: str, cle: str, cible: int) -> str:
    """Calibre au tokenizer, jamais par un ratio caractères/jetons."""
    graine = GRAINES[cle]
    if cible <= 400:
        return graine
    par_bloc = compte_jetons(url, graine)
    n = max(1, cible // max(1, par_bloc))
    return graine * n


def capture(url: str, etiquette: str) -> pathlib.Path:
    salt = uuid.uuid4().hex  # prefill FROID à jetons IDENTIQUES
    resultat = {"etiquette": etiquette, "cache_salt": salt, "prompts": {}}
    for cle in GRAINES:
        for cible, nom in ((300, "court"), (30000, "long")):
            corps_texte = construis(url, cle, cible)
            reel = compte_jetons(url, corps_texte)
            corps = {
                "model": _M,
                "messages": [
                    {"role": "user", "content": corps_texte + "\n\n" + QUESTIONS[cle]}
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False},
                "cache_salt": salt,
            }
            try:
                r = poste(url, "/v1/chat/completions", corps)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"  ÉCHEC {cle}/{nom} : {e}")
                return None
            lp = r["choices"][0].get("logprobs")
            if not lp or not lp.get("content"):
                print(f"  ÉCHEC {cle}/{nom} : aucun logprob rendu")
                return None
            top = lp["content"][0].get("top_logprobs") or []
            resultat["prompts"][f"{cle}/{nom}"] = {
                "jetons_prompt": reel,
                # `prompt_tokens_details` est PRESENT et vaut None quand il n'y a rien a
                # detailler : `.get(k, {})` rend alors None, pas {}. Meme piege que aiguille.py.
                "caches": ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                ),
                "top": [(e["token"], e["logprob"]) for e in top],
            }
            print(f"  {cle}/{nom:5} · {reel:>6} jet · caches {resultat['prompts'][f'{cle}/{nom}']['caches']:>6} · {len(top)} logprobs")
    QUAL.mkdir(parents=True, exist_ok=True)
    dest = QUAL / f"logprobs-{etiquette}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.rename(dest)  # jamais open("w") direct : ça tronque AVANT d'échouer
    return dest


def ecart(a: dict, b: dict, classe: str | None = None) -> tuple[float, float, int]:
    """MAE sur les jetons COMMUNS, et accord du top-1. `classe` filtre court/long.

    MESURE DU 2026-08-28 QUI JUSTIFIE LE DÉCOUPAGE : le bruit propre du serveur vaut
    **exactement 0,000000** sur les prompts courts (11 à 89 jetons, 4/4) et **0,19 à 0,40**
    sur les prompts de ~30 000 jetons, où il retourne même le top-1 sur une entrée
    identique. Les deux classes ne mesurent donc PAS la même chose :
      · court -> canal à BRUIT NUL : tout écart non nul est du pur écart de configuration ;
      · long  -> canal bruyant : il faut comparer au bruit, jamais à zéro.
    Agréger les deux noie le signal du canal propre dans le bruit de l'autre.
    """
    total = 0.0
    n = 0
    accords = 0
    prompts = 0
    for cle in a["prompts"]:
        if cle not in b["prompts"]:
            continue
        if classe and not cle.endswith("/" + classe):
            continue
        ta = dict(a["prompts"][cle]["top"])
        tb = dict(b["prompts"][cle]["top"])
        communs = set(ta) & set(tb)
        if not communs:
            continue
        for jeton in communs:
            total += abs(ta[jeton] - tb[jeton])
            n += 1
        pa = a["prompts"][cle]["top"][0][0] if a["prompts"][cle]["top"] else None
        pb = b["prompts"][cle]["top"][0][0] if b["prompts"][cle]["top"] else None
        accords += int(pa == pb)
        prompts += 1
    return (total / n if n else float("nan"), accords / prompts if prompts else 0.0, n)


def charge(etiquette: str) -> dict:
    p = QUAL / f"logprobs-{etiquette}.json"
    if not p.exists():
        raise SystemExit(f"🔴 ÉCHEC FERMÉ : capture absente — {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["capture", "compare"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--url", default=_U)
    a = ap.parse_args()

    if a.action == "capture":
        print(f"++ capture de logprobs — {a.args[0]} (prefill froid par cache_salt)")
        d = capture(a.url, a.args[0])
        if d is None:
            return 1
        print(f"  -> {d}")
        return 0

    if len(a.args) != 3:
        raise SystemExit("compare attend : <réf-a> <réf-b> <candidat>")
    ra, rb, cand = (charge(x) for x in a.args)
    print(f"++ MAE de logprobs · bruit propre = {a.args[0]} vs {a.args[1]} · candidat = {a.args[2]}")
    print(f"  {'classe':7} {'bruit propre':>13} {'candidat':>10} {'rapport':>8} {'top-1':>7} {'jetons':>7}")
    verdicts = []
    for classe in ("court", "long"):
        bruit, acc_bruit, n1 = ecart(ra, rb, classe)
        ca, acc_a, n2 = ecart(ra, cand, classe)
        cb, acc_b, _ = ecart(rb, cand, classe)
        cross = (ca + cb) / 2
        acc = (acc_a + acc_b) / 2
        if bruit <= 0.0:
            # canal a bruit nul : le seuil n'est plus un rapport mais une valeur absolue.
            # 0,05 nat sur un logprob est en dessous de toute consequence de sampling.
            ok = cross <= 0.05 and acc >= 1.0
            rap = "n/a"
        else:
            ok = cross <= 3.0 * bruit and acc >= acc_bruit
            rap = f"x{cross / bruit:.2f}"
        verdicts.append(ok)
        print(f"  {classe:7} {bruit:>13.6f} {cross:>10.6f} {rap:>8} {acc:>6.0%} {n2:>7}")
    verdict = "INDISCERNABLE" if all(verdicts) else "ECART STRUCTUREL"
    print(f"  >>> {verdict}")
    return 0 if verdict == "INDISCERNABLE" else 1


if __name__ == "__main__":
    sys.exit(main())
