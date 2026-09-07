#!/usr/bin/env python3
"""Multimodal probe: does `max-num-batched-tokens` cap the image encoder?

Sends N images of a given size in ONE request and reports, per rung: the HTTP code,
`prompt_tokens` (the real token cost of the multimodal input), and whether the shape->colour
mapping is right FOR EACH image.

Judging the mapping is necessary. A 200 with wrong colours is exactly what an encoder budget
overflow would produce (a truncated image), and a token count alone would not see it.

A negative arm is mandatory: the same request WITHOUT the images. If the model gets the
mapping right without seeing anything, the probe measures a prior of the prompt, not vision.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("FAIL CLOSED: this probe needs Pillow to synthesise its images.\n"
                     "  pip install pillow\n"
                     "It is the only dependency in tools/ outside the standard library,\n"
                     "and README.md's `vision - proven` row is reproduced with it.")

FORMES = ("cercle", "carre", "triangle")
COULEURS = {"rouge": (220, 30, 30), "vert": (30, 170, 60), "bleu": (40, 80, 220)}


def dessine(taille: int, appariement: dict) -> bytes:
    img = Image.new("RGB", (taille, taille), (255, 255, 255))
    d = ImageDraw.Draw(img)
    u = taille // 8
    for i, f in enumerate(FORMES):
        c = COULEURS[appariement[f]]
        y = u + i * (taille - 2 * u) // 3
        if f == "cercle":
            d.ellipse([u, y, u + 2 * u, y + 2 * u], fill=c)
        elif f == "carre":
            d.rectangle([u, y, u + 2 * u, y + 2 * u], fill=c)
        else:
            d.polygon([(u, y + 2 * u), (u + u, y), (u + 2 * u, y + 2 * u)], fill=c)
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def demande(base: str, cle: str, modele: str, images: list[bytes], n: int) -> dict:
    contenu = []
    for i, png in enumerate(images):
        contenu.append({"type": "text", "text": f"Image {i + 1} :"})
        contenu.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode()}})
    contenu.append({"type": "text", "text":
                    f"Pour CHACUNE des {n} images, donne la couleur du cercle, du carre et du "
                    "triangle. Reponds UNIQUEMENT en JSON : "
                    '{"1":{"cercle":"...","carre":"...","triangle":"..."}, ...}'})
    corps = {"model": modele, "max_tokens": 700, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": False},
             "messages": [{"role": "user", "content": contenu}]}
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(corps).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {cle}"} if cle else {})})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        return {"http": 200, "s": time.time() - t0, "d": d}
    except urllib.error.HTTPError as e:
        return {"http": e.code, "s": time.time() - t0,
                "err": e.read().decode(errors="ignore")[:300]}
    except Exception as e:  # noqa: BLE001
        return {"http": 0, "s": time.time() - t0, "err": f"{type(e).__name__}: {e}"[:300]}


def juge(txt: str, verites: list[dict]) -> tuple[int, int]:
    try:
        s = txt[txt.index("{"):txt.rindex("}") + 1]
        got = json.loads(s)
    except Exception:  # noqa: BLE001
        return 0, len(verites) * 3
    ok = 0
    for i, v in enumerate(verites, 1):
        g = got.get(str(i)) or got.get(i) or {}
        if isinstance(g, dict):
            ok += sum(1 for f in FORMES if str(g.get(f, "")).lower().strip() == v[f])
    return ok, len(verites) * 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BANC_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default="q38")
    ap.add_argument("--images", default="1,2,3,4")
    ap.add_argument("--taille", type=int, default=768)
    ap.add_argument("--bras", default="sonde")
    ap.add_argument("--json")
    a = ap.parse_args()
    cle = os.environ.get("API_KEY") or os.environ.get("VLLM_API_KEY") or ""

    rng = random.Random(20260907)
    res = []
    print(f"  sonde multi-image · {a.taille}x{a.taille} px · bras {a.bras} · {a.base}")
    for n in [int(x) for x in a.images.split(",")]:
        verites, pngs = [], []
        for _ in range(n):
            cs = list(COULEURS)
            rng.shuffle(cs)
            v = dict(zip(FORMES, cs))
            verites.append(v)
            pngs.append(dessine(a.taille, v))
        octets = sum(len(p) for p in pngs)
        r = demande(a.base, cle, a.model, pngs, n)
        if r["http"] != 200:
            print(f"  {n} image(s) · {octets/1024:6.1f} Kio · 🔴 HTTP {r['http']} "
                  f"en {r['s']:.1f}s · {r.get('err','')[:120]}")
            res.append({"n": n, "http": r["http"], "err": r.get("err")})
            continue
        d = r["d"]
        pt = d["usage"]["prompt_tokens"]
        c = d["choices"][0]
        txt = (c["message"].get("content") or "")
        ok, tot = juge(txt, verites)
        print(f"  {n} image(s) · {octets/1024:6.1f} Kio · 200 en {r['s']:5.1f}s · "
              f"prompt {pt:6d} jetons ({pt//max(n,1):5d}/image) · "
              f"appariement {ok}/{tot} · finish={c['finish_reason']}")
        res.append({"n": n, "http": 200, "prompt_tokens": pt, "juste": ok, "total": tot,
                    "finish": c["finish_reason"], "secondes": r["s"]})

    # ── bras NÉGATIF : sans image, l'appariement doit être FAUX (sinon la sonde ne mesure rien)
    verites = [dict(zip(FORMES, ["rouge", "vert", "bleu"]))]
    r = demande(a.base, cle, a.model, [], 1)
    if r["http"] == 200:
        ok, tot = juge(r["d"]["choices"][0]["message"].get("content") or "", verites)
        print(f"  bras NÉGATIF (sans image) · appariement {ok}/{tot} "
              f"{'[OK] the probe discriminates' if ok < tot else '🔴 correct WITHOUT seeing: probe is worthless'}")
        res.append({"n": 0, "negatif": True, "juste": ok, "total": tot})

    if a.json:
        with open(a.json + ".tmp", "w", encoding="utf-8") as fh:
            json.dump({"bras": a.bras, "taille": a.taille, "res": res}, fh,
                      indent=1, ensure_ascii=False)
        os.replace(a.json + ".tmp", a.json)
        print(f"  JSON : {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
