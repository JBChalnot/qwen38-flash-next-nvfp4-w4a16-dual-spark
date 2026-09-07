#!/bin/bash
# probe-qsa-kv-range — MESURE la plage dynamique des K/V de QSA. N'change RIEN au calcul.
#
# POURQUOI CE MOD EXISTE
#   Le bras P1 (2026-08-28) a établi que le KV fp8 par tenseur est bloqué pour une raison
#   précise : le checkpoint FP8 (`Qwen3.8-Flash-Next-FP8`) ne porte **aucun** `k_scale`/`v_scale` (les 75 421
#   tenseurs `*_scale` sont tous des `weight_scale_inv` d'experts), et
#   `model_executor/layers/attention/attention.py:131-132` enregistre donc
#   `_k_scale = _v_scale = torch.tensor(1.0)`. Or `float8_e4m3fn` sature à **±448**.
#   Les six `NotImplementedError` de `models/qwen4_exp/nvidia/qsa.py` (:108, :146, :183, :185,
#   :187, :280) protègent de ce fait un `inf` silencieux, ils ne sont pas de la prudence.
#
#   MAIS on ne sait pas ce que valent RÉELLEMENT nos K/V. Deux mondes très différents :
#     • absmax < 448  ⇒ l'échelle 1.0 suffit, le port fp8 devient beaucoup plus simple ;
#     • absmax >> 448 ⇒ il faut une échelle, et sa valeur doit être MESURÉE, pas devinée.
#   Concevoir la quantification avant de connaître ce chiffre serait deviner. Ce mod le mesure.
#
# CE QU'IL FAIT, ET CE QU'IL NE FAIT PAS
#   Il enveloppe `FlashAttentionImpl.do_kv_cache_update` (`v1/attention/backends/flash_attn.py`,
#   appelée par `qwen4_exp/nvidia/qsa.py:372`) et accumule un MAX GLISSANT de |K| et |V| par
#   couche, **sur le GPU**, sans jamais synchroniser dans la boucle chaude. Il journalise un
#   résumé toutes les N mises à jour. Le tenseur d'origine est passé INTACT à l'implémentation
#   d'origine : aucune valeur n'est modifiée, aucune décision n'est prise à partir de la mesure.
#
# 🔴 POURQUOI IL EST SÛR, ET LA LIMITE QUI VA AVEC
#   `torch.amax` sur un tenseur déjà résident ne fait qu'ajouter un kernel de réduction par
#   appel : le calcul, l'ordre et les sorties sont inchangés. En revanche il **coûte du débit**
#   (un kernel de plus par couche et par pas) ⇒ **ce mod est un instrument de campagne, jamais
#   un réglage de production.** Aucun chiffre de vitesse relevé sous ce mod n'est comparable.
#
# CE QU'IL FAUT LIRE APRÈS (un boot propre ne prouve rien)
#   1. `QSA-KV-RANGE` dans le journal : le max de |K| et |V| par couche, et le MAX GLOBAL ;
#   2. comparer ce max à 448 (e4m3) et à 57344 (e5m2) ;
#   3. l'échelle sûre est `absmax / 448` arrondie à la puissance de 2 supérieure — une puissance
#      de 2 rend la déquantification exacte au bit près sur la mantisse.
set -euo pipefail
say() { printf '  [probe-qsa-kv-range] %s\n' "$*"; }

# -- resolution du repertoire vllm : `VLLM_DIR` d'abord ------------------------
# POURQUOI CETTE VARIABLE. Un controle anti-derive extrait l'arbre vllm dans un TEMPORAIRE et
# lance ce script dessus ; sans `VLLM_DIR` le script resout `import vllm` et patche l'arbre du
# systeme, donc le controle ne mesure pas ce qu'il annonce. Dans le conteneur, le defaut
# resout exactement le chemin servi.
P="${VLLM_DIR:-$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')}"
[ -d "$P/model_executor" ] || { echo "ECHEC FERME : $P n'est pas un arbre vllm."; exit 1; }
F="$P/v1/attention/backends/flash_attn.py"
[ -f "$F" ] || { say "🔴 ECHEC FERME : $F introuvable"; exit 1; }

MARK="__QSA_KV_RANGE_PROBE__"
if grep -q "$MARK" "$F"; then say "déjà appliqué (idempotent)"; exit 0; fi

grep -q "def do_kv_cache_update" "$F" || { say "🔴 ECHEC FERME : do_kv_cache_update absente de $F"; exit 1; }

# On écrit dans un TEMPORAIRE puis on renomme : `open(p,"w")` tronque AVANT d'échouer, et une
# recette a déjà été laissée à 0 octet par un UnicodeEncodeError.
python3 - "$F" "$MARK" <<'PY'
import io, pathlib, sys

f = pathlib.Path(sys.argv[1]); mark = sys.argv[2]
src = f.read_text(encoding="utf-8")

anchor = "    def do_kv_cache_update(\n"
if anchor not in src:
    raise SystemExit("ECHEC FERME : ancre do_kv_cache_update introuvable")

shim = f'''
# {mark} — sonde de plage dynamique des K/V, posée par mods/probe-qsa-kv-range.
# N'altère aucune valeur : elle observe et journalise. Voir l'en-tête du mod.
def _qsa_kv_range_install():
    import os, torch, logging
    _log = logging.getLogger("vllm.qsa_kv_range")
    every = int(os.environ.get("QSA_KV_RANGE_EVERY", "200"))
    state = {{"n": 0, "max": {{}}}}
    _orig = FlashAttentionImpl.do_kv_cache_update

    def wrapped(self, layer, key, value, kv_cache, slot_mapping):
        try:
            nom = getattr(layer, "layer_name", None) or getattr(self, "layer_name", "?")
            if key.numel() and value.numel():
                mk = torch.amax(key.detach().abs())
                mv = torch.amax(value.detach().abs())
                cur = state["max"].get(nom)
                if cur is None:
                    state["max"][nom] = [mk, mv]
                else:
                    cur[0] = torch.maximum(cur[0], mk)
                    cur[1] = torch.maximum(cur[1], mv)
                state["n"] += 1
                if state["n"] % every == 0:
                    # UNE seule synchronisation, hors boucle chaude
                    items = sorted(
                        (n, float(v[0]), float(v[1])) for n, v in state["max"].items()
                    )
                    gk = max(k for _, k, _ in items)
                    gv = max(v for _, _, v in items)
                    _log.info(
                        "QSA-KV-RANGE apres %d ecritures · %d couches · "
                        "MAX GLOBAL |K|=%.3f |V|=%.3f · e4m3 sature a 448 => "
                        "echelle sure |K| %.4f |V| %.4f",
                        state["n"], len(items), gk, gv, gk / 448.0, gv / 448.0,
                    )
                    for n, k, v in items[:4]:
                        _log.info("QSA-KV-RANGE   %s |K|=%.3f |V|=%.3f", n, k, v)
        except Exception as e:                      # une sonde ne doit JAMAIS casser le serve
            _log.warning("QSA-KV-RANGE desactivee : %s", e)
            FlashAttentionImpl.do_kv_cache_update = _orig
        return _orig(self, layer, key, value, kv_cache, slot_mapping)

    FlashAttentionImpl.do_kv_cache_update = wrapped
    _log.info("QSA-KV-RANGE installee (resume toutes les %d ecritures)", every)


try:
    _qsa_kv_range_install()
except Exception as _e:                             # échec = pas de sonde, pas de serve cassé
    import logging
    logging.getLogger("vllm.qsa_kv_range").warning("QSA-KV-RANGE non installee : %s", _e)
'''

out = src.rstrip() + "\n" + shim
tmp = f.with_suffix(".py.tmp")
tmp.write_text(out, encoding="utf-8")
tmp.replace(f)
print("  patch ecrit")
PY

python3 -c "import ast, pathlib, sys; ast.parse(pathlib.Path('$F').read_text()); print('  syntaxe OK')" \
  || { say "🔴 ECHEC FERME : $F ne parse plus"; exit 1; }
grep -q "$MARK" "$F" || { say "🔴 ECHEC FERME : marqueur absent apres patch"; exit 1; }
say "applique — chercher QSA-KV-RANGE dans le journal"
