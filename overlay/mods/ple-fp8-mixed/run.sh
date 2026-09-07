#!/bin/bash
# ple-fp8-mixed — garde la table PLE en FP8 quand le checkpoint n'est plus un `Fp8Config`.
#
# ─────────────────────────────────────────────────────────────────────────────────────────
# POURQUOI
#   `models/qwen4_exp/nvidia/ple_layer.py:193` :  `if not isinstance(quant_config, Fp8Config)`
#   C'est le SEUL portail vers `Qwen4ExpPLEFp8EmbeddingMethod` (unique appel, ligne 288).
#   `Fp8Config` est importe ligne 30 depuis `quantization/fp8.py` ; ni
#   `CompressedTensorsConfig`, ni `ModelOptMixedPrecisionConfig` (`modelopt.py:2143`,
#   base `ModelOptQuantConfigBase`), ni `HummingConfig` (`humming.py:148`) n'en heritent.
#   Des que la greffe change `quant_method`, la fonction rend None,
#   `vocab_parallel_embedding.py:290-291` pose `UnquantizedEmbeddingMethod()`, et la table
#   est allouee en `model_config.dtype` (bf16, passe ligne 576) : 47,75 -> 95,43 GiB.
#
# ECHEC FERME, TROIS ETAGES
#   1. le mod n'agit QUE si `VLLM_QWEN4EXP_PLE_FP8=1` : a defaut il est inerte, donc le serve
#      fp8 actuel est bit-a-bit inchange ;
#   2. chaque substitution est COMPTEE : si une ligne amont bouge, 0 substitution => SystemExit
#      et le mod ne patche pas a moitie ;
#   3. sous le drapeau, `load_weights` EXIGE d'avoir charge `ngram_embedding.weight_scale`.
#      Pointe sur un checkpoint dont la PLE n'est pas fp8, le boot MEURT au lieu de
#      transtyper en silence (`common/ple.py` fait `target.copy_(source.to(dtype))`).
#
# CE QUE CE MOD NE FAIT PAS
#   Il ne touche NI le dispatch des experts, NI le MTP, NI l'attention. Une seule fonction.
set -euo pipefail
say() { printf '  [ple-fp8-mixed] %s\n' "$*"; }

P="${VLLM_DIR:-$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')}"
[ -d "$P/model_executor" ] || { echo "ECHEC FERME : $P n'est pas un arbre vllm."; exit 1; }
PLE="$P/models/qwen4_exp/nvidia/ple_layer.py"
[ -f "$PLE" ] || { say "🔴 ECHEC FERME : $PLE introuvable"; exit 1; }

MARK="__PLE_FP8_MIXED__"
if grep -q "$MARK" "$PLE"; then say "deja applique (idempotent)"; exit 0; fi

python3 - "$PLE" "$MARK" <<'PY'
import pathlib, re, sys

ple = pathlib.Path(sys.argv[1]); mark = sys.argv[2]

def ecrire(p, texte):
    """Temporaire puis rename : `open(p,"w")` tronque AVANT d'echouer."""
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(texte, encoding="utf-8"); t.replace(p)

def subn_exig(motif, remplacement, texte, attendu, quoi):
    out, n = re.subn(motif, remplacement, texte)
    if n != attendu:
        raise SystemExit(f"ECHEC FERME : {quoi} — {n} substitution(s), {attendu} attendue(s)")
    return out

t = ple.read_text(encoding="utf-8")

# 1. `os` n'est pas importe par ce module.
t = subn_exig(r"(?m)^import math\n", "import math\nimport os\n", t, 1, "1 import os")

# 2. LE portail. On laisse le chemin `Fp8Config` d'origine INTACT ; on ajoute la branche
#    mixte AVANT lui, sous drapeau. `is_checkpoint_fp8_serialized` / `ignored_layers` /
#    `ignored_layers_match_mode` n'existent que sur `Fp8Config`, donc la branche mixte ne
#    peut pas les lire : elle rend la methode fp8 telle quelle.
t = subn_exig(
    r"    if not isinstance\(quant_config, Fp8Config\):\n        return None\n",
    "    if quant_config is not None and not isinstance(quant_config, Fp8Config):\n"
    f"        # {mark} : greffe MIXED_PRECISION — le quant_config n'est plus un Fp8Config,\n"
    "        # mais la table PLE reste F8_E4M3 sur disque. Sans ceci elle repasse en bf16.\n"
    "        if os.environ.get(\"VLLM_QWEN4EXP_PLE_FP8\") == \"1\":\n"
    "            return Qwen4ExpPLEFp8EmbeddingMethod()\n"
    "    if not isinstance(quant_config, Fp8Config):\n        return None\n",
    t, 1, "2 portail ple_layer.py:193")

# 3. Le controle positif : sous le drapeau, la PLE DOIT avoir une echelle globale UTILISABLE.
#
# Le garde teste la VALEUR, pas l'appartenance : tester `"ngram_embedding.weight_scale" not in loaded`,
# donc la COMPTABILITE du chargeur. Elle a REFUSE le checkpoint NVIDIA alors que l'echelle
# y est bien presente (`...ngram_embedding.weight_scale`, BF16 [1], verifie aux en-tetes,
# nom et forme IDENTIQUES aux notres) : faux positif, boot mort a 2 min 34.
# La cause de fond est notre propre cicatrice : `default_loader.py:436-464` DESACTIVE le
# suivi des poids pour tout modele quantifie et marque comme charges les parametres des
# modules a `process_weights_after_loading` ⇒ l'appartenance a `loaded` n'est pas un
# observable fiable. Et la v1 n'avait jamais ete eprouvee en ACCEPTATION sous
# `modelopt_mixed` : le boot du 08-31 qui la validait tournait sous `compressed-tensors`.
#
# ⇒ On teste desormais le RESULTAT, ce qui est a la fois plus simple et plus fort : l'echelle
# doit etre finie et strictement positive. Ca attrape les DEUX modes de defaillance —
# l'echelle jamais chargee (elle reste a sa sentinelle `torch.finfo(float32).min`, exactement
# ce qui a tue la spec le 08-31) ET l'echelle chargee a une valeur absurde.
t = subn_exig(
    r"        if regular_weights:\n"
    r"            loaded\.update\(AutoWeightsLoader\(self\)\.load_weights\(regular_weights\)\)\n"
    r"        return loaded\n",
    "        if regular_weights:\n"
    "            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))\n"
    f"        # {mark} : echec ferme sur la VALEUR de l'echelle, pas sur la comptabilite\n"
    "        # du chargeur (non fiable pour un modele quantifie).\n"
    "        _sc = getattr(self.ngram_embedding, \"weight_scale\", None)\n"
    "        if _sc is not None:\n"
    "            _v = float(_sc.detach().reshape(-1)[0].float())\n"
    "            if not (_v > 0.0) or _v != _v or _v in (float(\"inf\"), float(\"-inf\")):\n"
    "                raise ValueError(\n"
    "                    \"PLE fp8 : echelle globale inutilisable \"\n"
    f"                    f\"(weight_scale={{_v!r}}) — jamais chargee, ou absurde. \"\n"
    "                    \"Sans elle la table serait lue avec une echelle sentinelle.\"\n"
    "                )\n"
    "        return loaded\n",
    t, 1, "3 controle positif de l'echelle")

ecrire(ple, t)
print("  ple_layer.py patche (3/3)")
PY

say "OK — inerte tant que VLLM_QWEN4EXP_PLE_FP8 != 1"
