#!/usr/bin/env bash
# mtp-modelopt-mixed — rend le drafter MTP du checkpoint NVIDIA chargeable sous `modelopt_mixed`.
#
# LES DEUX TROUS, VÉRIFIÉS À LA SOURCE DANS NOTRE IMAGE
#   Le checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4` déclare, pour son drafter :
#       mtp.layers.0.mlp.experts -> {quant_algo: FP8_BLOCK_SCALES, group_size: 128}
#
#   B3 — L'INDICE EST LOCAL AU DRAFTER. `mtp.py:204` construit les modules sous
#        `mtp.layers.<num_hidden_layers>.*`, soit `mtp.layers.48.*` ici, tandis que modelopt
#        indexe `mtp.layers.0`. `_quantized_layer_prefix_candidates` (`modelopt.py:2355-2370`)
#        ne propose que la bascule `language_model.model.` <-> `model.language_model.` ⇒ le
#        préfixe réel ne trouve JAMAIS son entrée.
#
#   B2 — `FP8_BLOCK_SCALES` EST INCONNU DU DISPATCH, ET LE REPLI EST SILENCIEUX.
#        `grep -rc FP8_BLOCK_SCALES` sur tout l'arbre vLLM : 0 occurrence. Et rien ne lève :
#        `_from_config` (`:2196-2234`) n'y cherche qu'un `group_size` et ignore le reste ;
#        `get_quant_method` retombe sur `UnquantizedLinearMethod()` pour un `LinearBase` et sur
#        `return None` pour un `RoutedExperts` (`:2423`). ⇒ les experts du drafter sont créés
#        **BF16 non quantifiés** alors que le checkpoint les porte en FP8 par blocs 128x128.
#
# 🔴 CE QUE ÇA DONNE SANS LE MOD, ET POURQUOI AUCUN INSTRUMENT NE LE VOIT
#   Le serve boote, répond 200, et les réponses restent JUSTES — une proposition MTP est
#   toujours vérifiée par la cible, donc un drafter faux coûte du DÉBIT, jamais de la
#   correction. C'est mot pour mot notre cicatrice du 2026-08-31 : 30 993 jetons proposés,
#   **0 accepté**, MAL 1,000, débit 19,4 contre 44,7. ⇒ le contrôle d'un bras MTP est le
#   **MAL**, jamais le débit.
#
# LE CORRECTIF, ET SON ORIGINE
#   Porté depuis `tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark`,
#   `single-spark-vllm-tp1/patch/upstream-overlays/modelopt.py:1710-1722` et `:1780-1796`
#   (Kai / 2Wild, 2026-09-05, **Apache-2.0**). Deux autres implémentations indépendantes du
#   même jour existent (sfxnz, MIT ; MiaAI-Lab, AGPL-3.0) — nous portons la version Apache.
#   Signatures vérifiées dans NOTRE image avant portage (règle : aucune hypothèse) :
#     `Fp8MoEMethod.__init__(self, quant_config: Fp8Config, layer: RoutedExperts)` (`fp8.py:480`)
#     `Fp8Config.__init__(..., activation_scheme="dynamic", weight_block_size=None, ...)` (`:98`)
#   ⇒ l'appel positionnel de la référence transpose tel quel.
#
# POURQUOI CE MOD EST INERTE PAR CONSTRUCTION, SANS DRAPEAU
#   Il ne touche que `quantization/modelopt.py`. Notre serve de production charge
#   le checkpoint FP8 (`Qwen3.8-Flash-Next-FP8`), dont le `quantization_config.quant_method` vaut `FP8` ⇒ le chemin élu est
#   `quantization/fp8.py`, jamais modelopt. Aucun drapeau n'est donc nécessaire : le code
#   ajouté n'est atteignable que par un checkpoint `MIXED_PRECISION`. C'est un fait de
#   dispatch, pas une intention — et c'est ce qui le rend vérifiable.
set -euo pipefail
say() { printf '  [mtp-modelopt-mixed] %s\n' "$*"; }

P="${VLLM_DIR:-$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')}"
[ -d "$P/model_executor" ] || { echo "ECHEC FERME : $P n'est pas un arbre vllm."; exit 1; }
F="$P/model_executor/layers/quantization/modelopt.py"
[ -f "$F" ] || { say "🔴 ECHEC FERME : $F introuvable"; exit 1; }

MARK="__MTP_MODELOPT_MIXED__"
if grep -q "$MARK" "$F"; then say "deja applique (idempotent)"; exit 0; fi

python3 - "$F" "$MARK" <<'PY'
import ast
import pathlib
import re
import sys

f = pathlib.Path(sys.argv[1])
mark = sys.argv[2]


def ecrire(p, texte):
    """`open(p,"w")` tronque AVANT d'echouer : temporaire puis rename."""
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(texte, encoding="utf-8")
    t.replace(p)


def subn_exig(motif, remplacement, texte, attendu, quoi):
    out, n = re.subn(motif, remplacement, texte, flags=re.M)
    if n != attendu:
        raise SystemExit(f"ECHEC FERME : {quoi} — {n} substitution(s), {attendu} attendue(s)")
    return out


t = f.read_text(encoding="utf-8")

# ── B3 : offrir l'indice LOCAL au drafter comme candidat supplementaire.
#    Ancre : la premiere ligne du corps de `_quantized_layer_prefix_candidates`.
#    ⚠️ Le code injecte n'utilise PAS `re` : modelopt.py ne l'importe pas, et injecter un
#    import serait une piece mobile de plus (ancre d'AST a trouver, portee a verifier). La
#    reference amont s'en sert ; nous faisons le meme decoupage en chaines pures.
#    Cicatrice : ma v1 GARDAIT contre l'absence de `import re` — donc elle refusait TOUT,
#    y compris la source saine, et les quatre controles negatifs etaient verts pour la
#    mauvaise raison. Seul le controle POSITIF l'a demasquee (2026-09-03, meme piege).
ANCRE_B3 = (
    r"    def _quantized_layer_prefix_candidates\(prefix: str\) -> tuple\[str, \.\.\.\]:\n"
    r"        candidates = \[prefix\]\n"
)
t = subn_exig(
    ANCRE_B3,
    "    def _quantized_layer_prefix_candidates(prefix: str) -> tuple[str, ...]:\n"
    "        candidates = [prefix]\n"
    f"        # {mark} B3 : vLLM numerote les couches du drafter a partir de\n"
    "        # num_hidden_layers (mtp.layers.48.*), modelopt les indexe en LOCAL\n"
    "        # (mtp.layers.0.*). Sans ce candidat le prefixe reel ne trouve jamais son\n"
    "        # entree, et le repli est SILENCIEUX (experts crees non quantifies).\n"
    "        _jeton = \"mtp.layers.\"\n"
    "        _pos = prefix.find(_jeton)\n"
    "        if _pos >= 0:\n"
    "            _tete = prefix[: _pos + len(_jeton)]\n"
    "            _num, _sep, _queue = prefix[_pos + len(_jeton) :].partition(\".\")\n"
    "            if _sep and _num.isdigit():\n"
    "                _k = int(_num)\n"
    "                for _j in range(0, min(_k, 8) + 1):\n"
    "                    if _j != _k:\n"
    "                        candidates.append(f\"{_tete}{_j}.{_queue}\")\n",
    t, 1, "B3 candidats de prefixe")

# ── B2 : router FP8_BLOCK_SCALES vers la methode MoE FP8 par blocs de vLLM.
#    Ancre : la branche W4A16_NVFP4 du bloc `RoutedExperts` — unique dans le fichier.
ANCRE_B2 = (
    r"            if quant_algo == \"W4A16_NVFP4\":\n"
    r"                return ModelOptNvFp4FusedMoE\(\n"
    r"                    quant_config=self\.w4a16_nvfp4_config,\n"
    r"                    moe_config=layer\.moe_config,\n"
    r"                \)\n"
)
t = subn_exig(
    ANCRE_B2,
    "            if quant_algo == \"W4A16_NVFP4\":\n"
    "                return ModelOptNvFp4FusedMoE(\n"
    "                    quant_config=self.w4a16_nvfp4_config,\n"
    "                    moe_config=layer.moe_config,\n"
    "                )\n"
    f"            if quant_algo in (\"FP8_BLOCK_SCALES\", \"FP8_BLOCK\"):\n"
    f"                # {mark} B2 : les experts routes du MTP sont du FP8 a echelles de\n"
    "                # bloc 128x128 (weight_scale_inv). Le dispatch mixte n'avait aucune\n"
    "                # branche et retombait sur `return None` = non quantifie, en silence.\n"
    "                from vllm.model_executor.layers.quantization.fp8 import (\n"
    "                    Fp8Config as _Fp8Config,\n"
    "                    Fp8MoEMethod as _Fp8MoEMethod,\n"
    "                )\n"
    "                _gs = 128\n"
    "                for _c in self._quantized_layer_prefix_candidates(prefix):\n"
    "                    _i = self.quantized_layers.get(_c)\n"
    "                    if _i and _i.get(\"group_size\"):\n"
    "                        _gs = int(_i[\"group_size\"])\n"
    "                        break\n"
    "                return _Fp8MoEMethod(\n"
    "                    _Fp8Config(\n"
    "                        is_checkpoint_fp8_serialized=True,\n"
    "                        activation_scheme=\"dynamic\",\n"
    "                        weight_block_size=[_gs, _gs],\n"
    "                    ),\n"
    "                    layer,\n"
    "                )\n",
    t, 1, "B2 branche FP8_BLOCK_SCALES")

# ── controle de PORTEE : `ast.parse` valide la syntaxe, jamais les noms.
#    Chaque nom lu par le code injecte doit etre assigne dans SA fonction hote.
arbre = ast.parse(t)


def fonction_contenant(arbre, marqueur_ligne):
    trouvee = None
    for n in ast.walk(arbre):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.lineno <= marqueur_ligne <= (n.end_lineno or n.lineno):
                if trouvee is None or n.lineno > trouvee.lineno:
                    trouvee = n
    return trouvee


lignes = t.splitlines()
sites = [i + 1 for i, l in enumerate(lignes) if mark in l]
if len(sites) != 2:
    raise SystemExit(f"ECHEC FERME : {len(sites)} marqueur(s) injecte(s), 2 attendus")

EXIGENCES = {
    "_quantized_layer_prefix_candidates": {"prefix", "candidates", "_jeton", "_pos",
                                           "_tete", "_num", "_sep", "_queue", "_k", "_j"},
    "get_quant_method": {"prefix", "quant_algo", "layer", "_gs", "_c", "_i"},
}
for ligne in sites:
    fn = fonction_contenant(arbre, ligne)
    if fn is None:
        raise SystemExit(f"ECHEC FERME : injection ligne {ligne} hors de toute fonction")
    besoins = EXIGENCES.get(fn.name)
    if besoins is None:
        raise SystemExit(f"ECHEC FERME : injection dans `{fn.name}`, fonction non prevue")
    assignes = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.posonlyargs}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            assignes.add(n.id)
        elif isinstance(n, ast.comprehension) and isinstance(n.target, ast.Name):
            assignes.add(n.target.id)
    manquants = besoins - assignes
    if manquants:
        raise SystemExit(f"ECHEC FERME : dans `{fn.name}`, noms hors portee : "
                         f"{sorted(manquants)}")

ecrire(f, t)
print("  modelopt.py patche (2/2), portee verifiee par AST dans les 2 fonctions hotes")
PY

say "OK — atteignable uniquement par un checkpoint MIXED_PRECISION"
