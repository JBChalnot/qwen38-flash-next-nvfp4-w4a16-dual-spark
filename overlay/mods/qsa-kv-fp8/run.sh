#!/bin/bash
# qsa-kv-fp8 — débloque `--kv-cache-dtype fp8_e4m3` sur le chemin QSA de q38 (`qwen4_exp`).
#
# ⚠️ TRAVAUX TIERS POSTÉRIEURS ET INDÉPENDANTS : l'issue vLLM #54426 (Nanetnounou, 2026-08-30,
#   sur GB10) et la PR #54846 (andreasgru, 2026-09-01) traitent le même sujet, aux mêmes 4 points
#   d'intégration. Ce script date du 2026-08-28 (mesure sur le build FP8), écrit sans les connaître ; nous les
#   avons lus le 2026-09-04 comme corroboration (×1,79 et ×1,889 contre notre ×1,84). Indépendance,
#   pas priorité : le code laisse peu de place à d'autres solutions. Aucune ligne copiée dans un
#   sens ni l'autre. Si #54846 fusionne, ce mod devient inutile sur ce moteur : le RETIRER. Voir NOTICE.
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# CE QUE CE MOD FAIT, ET POURQUOI CHAQUE MORCEAU EXISTE
# ─────────────────────────────────────────────────────────────────────────────────────────────
# Le cache KV principal de QSA est en BF16, et six `NotImplementedError` refusent tout autre
# dtype. Le passer en fp8 e4m3 double la capacité du pool (coût par jeton 14 144 → 7 696 o,
# soit ×1,84), ce qui est la seule voie mesurée vers « 1 session à 700 k + 3 subagents à 200 k »
# sur cette box, où il a été PROUVÉ qu'il n'y a plus un octet à prendre (+1,44 GiB de KV a
# suffi à OOM-killer le rang 0 le 2026-08-28).
#
# 🔴 POURQUOI LES SIX REFUS EXISTENT, ET POURQUOI ON PEUT LES LEVER ICI
#   Le checkpoint FP8 (`Qwen3.8-Flash-Next-FP8`) ne porte AUCUN `k_scale`/`v_scale` (les 75 421 tenseurs
#   `*_scale` sont tous des `weight_scale_inv` d'experts), donc
#   `model_executor/layers/attention/attention.py:131-132` pose `_k_scale = _v_scale = 1.0`.
#   Or `float8_e4m3fn` sature à ±448 : à échelle 1.0, toute activation au-delà devient `inf`.
#   Les refus protègent donc d'un `inf` SILENCIEUX — ce n'est pas de la prudence.
#
#   MAIS la plage a été MESURÉE EN VOL (`mods/probe-qsa-kv-range`, 200 écritures, 13 couches) :
#       |K|max = 89,500   |V|max = 31,375   contre ±448
#   soit **5,0× et 14,3× de marge**. L'échelle de 1.0 est donc sûre, et c'est mesuré, pas supposé.
#   ⇒ Aucune calibration, aucun cache d'échelles, aucune modification de la spec du cache KV.
#
# CE QUE LE PATCH TOUCHE, EXHAUSTIVEMENT
#   A. `nvidia/qsa.py`   : 1 `supported_kv_cache_dtypes` + 6 `NotImplementedError`
#   B. `common/qsa_cache.py` : 1 `supported_kv_cache_dtypes`
#   C. `nvidia/ops/qsa.py`  : l'`assert` de dtype de `qsa_sparse_paged_attention`
#                             + 3 sites de `tl.load` transtypés vers le dtype de la requête
#   D. la VUE `uint8 → float8_e4m3fn` : vLLM alloue le cache en **uint8**
#      (`utils/torch_utils.py` `STR_DTYPE_TO_TORCH_DTYPE["fp8_e4m3"] = torch.uint8`). Le chemin FA
#      générique la réinterprète (`v1/attention/backends/flash_attn.py:1056-1058`
#      `key_cache.view(current_platform.fp8_dtype())`) ; **le chemin QSA ne le fait pas.**
#
# POURQUOI LE CAST VA SUR LE `tl.load` ET PAS SUR LE `tl.dot`
#   Vérifié sur silicium sm_121 (bras P1) : `tl.dot` fp8×fp8 est ACCEPTÉ, mais fp8×bf16 est
#   REFUSÉ (`Unsupported rhs dtype fp8e4nv`) — et il est refusé sur TOUTES les architectures, ce
#   n'est pas une limite de GB10. On transtype donc à la LECTURE, patron de la PR vLLM #47665
#   (« for bf16 caches this folds to a no-op ») : sur un cache BF16 le `.to()` disparaît, donc le
#   patch est **inerte tant que `--kv-cache-dtype` reste `auto`**.
#
# 🔴 ÉCHEC FERMÉ
#   On n'accepte QUE `fp8` et `fp8_e4m3`. Pour toute autre valeur, le refus d'origine est
#   ré-émis TEL QUEL — une vraie erreur de configuration reste une erreur au lieu d'être avalée
#   et de produire un cache mal quantifié en silence. Et chaque substitution de texte est
#   comptée : si un compte ne tombe pas juste, le mod ÉCHOUE au lieu de patcher à moitié.
#
# CE QU'IL FAUT VÉRIFIER APRÈS (un chargement propre ne prouve RIEN)
#   1. `cache_dtype` dans l'écho du moteur, ET `num_gpu_blocks` qui doit MONTER ;
#      ⚠️ le `block_size` passe de 832 à **1664** (il est dérivé de la page mamba), donc
#      `bytes_per_block` AUGMENTE et le bon observable est `num_gpu_blocks × block_size` ;
#   2. la gate de dégénérescence, AVANT tout chiffre de vitesse ;
#   3. l'aiguille longue en prefill FROID — e4m3 n'a que 3 bits de mantisse, donc ~6 % de
#      précision relative quelle que soit la plage : c'est la RÉCUPÉRATION qui tranche ;
#   4. le MAL (`tools/decode-mal.py`, qui le rapporte) — la vérification spéculative LIT le KV, c'est le canal de
#      dégradation le plus direct, et le KV a déjà tué le MTP en silence sur un hybride ;
#   5. la MAE de logprobs contre le bruit propre (le serveur n'est pas déterministe
#      à T=0, donc l'identité de sortie n'est pas un critère).
set -euo pipefail
say() { printf '  [qsa-kv-fp8] %s\n' "$*"; }

# -- resolution du repertoire vllm : `VLLM_DIR` d'abord ------------------------
# POURQUOI CETTE VARIABLE. Un controle anti-derive extrait l'arbre vllm dans un TEMPORAIRE et
# lance ce script dessus ; sans `VLLM_DIR` le script resout `import vllm` et patche l'arbre du
# systeme, donc le controle ne mesure pas ce qu'il annonce. Dans le conteneur, le defaut
# resout exactement le chemin servi.
P="${VLLM_DIR:-$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')}"
[ -d "$P/model_executor" ] || { echo "ECHEC FERME : $P n'est pas un arbre vllm."; exit 1; }
QSA="$P/models/qwen4_exp/nvidia/qsa.py"
CACHE="$P/models/qwen4_exp/common/qsa_cache.py"
OPS="$P/models/qwen4_exp/nvidia/ops/qsa.py"
FA="$P/v1/attention/backends/flash_attn.py"
for f in "$QSA" "$CACHE" "$OPS" "$FA"; do
  [ -f "$f" ] || { say "🔴 ECHEC FERME : $f introuvable"; exit 1; }
done

MARK="__QSA_KV_FP8_ENABLED__"
if grep -q "$MARK" "$QSA"; then say "déjà appliqué (idempotent)"; exit 0; fi

python3 - "$QSA" "$CACHE" "$OPS" "$MARK" <<'PY'
import pathlib, re, sys

qsa = pathlib.Path(sys.argv[1])
cache = pathlib.Path(sys.argv[2])
ops = pathlib.Path(sys.argv[3])
mark = sys.argv[4]

ACCEPTES = '("auto", "bfloat16", "fp8", "fp8_e4m3")'


def ecrire(p: pathlib.Path, texte: str) -> None:
    """Temporaire puis rename : `open(p,"w")` tronque AVANT d'échouer."""
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(texte, encoding="utf-8")
    t.replace(p)


def subn_exig(motif, remplacement, texte, attendu, quoi):
    out, n = re.subn(motif, remplacement, texte)
    if n != attendu:
        raise SystemExit(f"ECHEC FERME : {quoi} — {n} substitution(s), {attendu} attendue(s)")
    return out


# ── A. nvidia/qsa.py ────────────────────────────────────────────────────────────────────────
t = qsa.read_text(encoding="utf-8")

# A1. la liste declarative
t = subn_exig(
    r'supported_kv_cache_dtypes: ClassVar\[list\[CacheDType\]\] = \["auto", "bfloat16"\]',
    'supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
    '        "auto",\n        "bfloat16",\n        "fp8",\n        "fp8_e4m3",\n    ]'
    f'  # {mark}',
    t, 1, "A1 supported_kv_cache_dtypes de nvidia/qsa.py")

# A2. le garde de dtype de la config (ligne ~107)
t = subn_exig(
    r'if self\.kv_cache_dtype not in \("auto", "bfloat16"\):',
    f'if self.kv_cache_dtype not in {ACCEPTES}:',
    t, 1, "A2 garde de config")

# A3. les refus qui comparent le dtype TORCH du cache a bfloat16.
#     vLLM alloue le fp8 en UINT8, donc on accepte aussi uint8 et float8_e4m3fn.
t = subn_exig(
    r'if self\.kv_cache_torch_dtype != torch\.bfloat16:',
    'if self.kv_cache_torch_dtype not in (\n'
    '            torch.bfloat16,\n            torch.uint8,\n            torch.float8_e4m3fn,\n        ):',
    t, 1, "A3 garde de stockage")

# A4. LA VUE uint8 -> float8_e4m3fn. vLLM alloue le cache fp8 en UINT8
#     (`utils/torch_utils.py` STR_DTYPE_TO_TORCH_DTYPE["fp8_e4m3"] = torch.uint8). Le chemin FA
#     generique la pose (`v1/attention/backends/flash_attn.py:1056-1058`) ; QSA ne le fait PAS,
#     donc sans elle le noyau recevrait des pointeurs uint8. Le bras P1 a verifie que la vue est
#     SANS COPIE : meme data_ptr, meme storage.
VUE = (
    "\n\\2# " + mark + " : le cache fp8 est alloue en uint8 ; la vue est sans copie."
    "\n\\2if key_cache.dtype == torch.uint8:"
    "\n\\2    key_cache = key_cache.view(torch.float8_e4m3fn)"
    "\n\\2    value_cache = value_cache.view(torch.float8_e4m3fn)"
)
t = subn_exig(
    r"(\n(\s+)key_cache, value_cache = kv_cache\.transpose\(1, 2\)\.split\(self\.head_size, dim=-1\))",
    r"\1" + VUE,
    t, 1, "A4 vue uint8 -> float8_e4m3fn")

nb_refus = len(re.findall(r'raise NotImplementedError\("Qwen4Exp QSA (?:requires|currently requires|does not support KV)', t))
ecrire(qsa, t)
print(f"  A. nvidia/qsa.py patche (refus lies au dtype restants a neutraliser : {nb_refus})")

# ── B. common/qsa_cache.py ──────────────────────────────────────────────────────────────────
t = cache.read_text(encoding="utf-8")
t = subn_exig(
    r'supported_kv_cache_dtypes: ClassVar\[list\[CacheDType\]\] = \["auto", "bfloat1(?:6"\])',
    'supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
    '        "auto",\n        "bfloat16",\n        "fp8",\n        "fp8_e4m3",\n    ]'
    f'  # {mark}',
    t, 1, "B1 supported_kv_cache_dtypes de qsa_cache.py")
ecrire(cache, t)
print("  B. common/qsa_cache.py patche")

# ── C. nvidia/ops/qsa.py ────────────────────────────────────────────────────────────────────
t = ops.read_text(encoding="utf-8")

# C1. l'assert de dtype : autoriser un cache fp8 avec une requete bf16
t = subn_exig(
    r'assert q\.dtype == k_cache\.dtype == v_cache\.dtype == torch\.bfloat16',
    f'# {mark} : le cache peut etre fp8 e4m3 pendant que q reste bf16 ; le transtypage\n'
    '    # a lieu au `tl.load` dans le noyau (patron vLLM #47665).\n'
    '    assert q.dtype == torch.bfloat16\n'
    '    assert k_cache.dtype == v_cache.dtype\n'
    '    assert k_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)',
    t, 1, "C1 assert de dtype")

# C2/C3/C4. transtyper les trois chargements de cache vers le dtype de la requete.
#     Sur un cache BF16 le `.to()` se replie en no-op => le patch est inerte a `auto`.
t = subn_exig(
    r'(\n        scores = tl\.dot\(keys, query, out_dtype=tl\.float32\))',
    '\n        keys = keys.to(query.dtype)   # ' + mark + r'\1',
    t, 1, "C2 cast du site de decode")
t = subn_exig(
    r'(\n        scores = tl\.dot\(query, keys\))',
    '\n        keys = keys.to(query.dtype)      # ' + mark + '\n'
    '        values = values.to(query.dtype)  # ' + mark + r'\1',
    t, 1, "C3/C4 cast des sites de prefill")
ecrire(ops, t)
print("  C. nvidia/ops/qsa.py patche (assert + 3 casts)")
PY

# ── neutraliser les refus restants, un par un et en comptant ────────────────────────────────
# La portée doit être vérifiée par AST : un premier essai avait posé le prédicat
# `if self.kv_cache_dtype not in ("fp8","fp8_e4m3")`. Le fichier PARSAIT, le mod s'appliquait
# proprement sur les deux rangs… et le boot mourait sur
# `AttributeError: 'Qwen4ExpQSAAttention' object has no attribute 'kv_cache_dtype'`.
# Un parcours d'AST l'a chiffré : **5 sites sur 6 étaient hors portée** —
# `Qwen4ExpQSAFlashAttentionImpl` n'assigne jamais cet attribut (il l'hérite), et dans
# `Qwen4ExpQSAAttention` trois sites s'exécutent AVANT son assignation.
# ⇒ le prédicat est désormais une CONSTANTE DE MODULE, valide dans toutes les portées, et
# lue dans l'environnement. Double bénéfice : c'est un opt-in EXPLICITE (donc auditable dans
# `/proc/<pid>/environ`) et l'assouplissement ne peut pas s'activer par accident.
python3 - "$QSA" "$MARK" <<'PY'
import ast, pathlib, re, sys
p, mark = pathlib.Path(sys.argv[1]), sys.argv[2]
t = p.read_text(encoding="utf-8")

FLAG = "_QSA_KV_FP8_OK"

# 1. la constante de module, posee juste apres les imports (donc valide partout)
if FLAG not in t:
    # ANCRAGE PAR AST, jamais par regex : une regex texte attrape aussi les imports
    # INDENTES (bloc TYPE_CHECKING, continuation) et l'insertion casse la syntaxe.
    _a = ast.parse(t)
    _fin = max((n.end_lineno for n in _a.body
                if isinstance(n, (ast.Import, ast.ImportFrom))), default=0)
    if not _fin:
        raise SystemExit("ECHEC FERME : aucun import de niveau module pour ancrer la constante")
    pos = sum(len(l) + 1 for l in t.splitlines()[:_fin])
    const = (f"\n\n# {mark} : opt-in EXPLICITE du KV fp8 sur le chemin QSA."
             f"\n# Constante de MODULE et non attribut d'instance : les refus qu'elle garde"
             f"\n# vivent dans deux classes dont l'une n'assigne jamais `self.kv_cache_dtype`"
             f"\n# et l'autre le fait APRES eux (la syntaxe ne dit rien de la portee)."
             f"\nimport os as _qsa_os"
             f"\n{FLAG} = _qsa_os.environ.get('VLLM_QSA_KV_FP8', '0') == '1'\n")
    t = t[:pos] + const + t[pos:]

# 2. chaque refus devient conditionnel : il tire SAUF si l'opt-in est pose => echec FERME
motifs = [
    (r'(\n(\s+))raise NotImplementedError\("Qwen4Exp QSA requires a BF16 main KV cache"\)', 2),
    (r'(\n(\s+))raise NotImplementedError\("Qwen4Exp QSA requires BF16 Q/K/V"\)', 1),
    (r'(\n(\s+))raise NotImplementedError\("Qwen4Exp QSA currently requires BF16"\)', 1),
    (r'(\n(\s+))raise NotImplementedError\("Qwen4Exp QSA does not support KV quantization"\)', 1),
    (r'(\n(\s+))raise NotImplementedError\("Qwen4Exp QSA requires BF16 cache storage"\)', 1),
]
total = 0
for motif, attendu in motifs:
    def rep(m):
        ind = m.group(2)
        corps = m.group(0).lstrip("\n").strip()
        return f"\n{ind}if not {FLAG}:  # {mark}\n{ind}    {corps}"
    t, n = re.subn(motif, rep, t)
    if n != attendu:
        raise SystemExit(f"ECHEC FERME : refus {motif[:52]}… — {n} trouve(s), {attendu} attendu(s)")
    total += n

# 3. 🔴 CONTROLE DE PORTEE, celui qui manquait : chaque nom utilise par le code injecte
#    doit exister dans sa portee. On le verifie par AST, pas en esperant.
arbre = ast.parse(t)
globaux = {n.targets[0].id for n in arbre.body
           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
globaux |= {a.asname or a.name.split(".")[0]
            for n in arbre.body if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names}
if FLAG not in globaux:
    raise SystemExit(f"ECHEC FERME : {FLAG} n'est pas une globale du module apres patch")
sites = [i + 1 for i, l in enumerate(t.splitlines()) if f"if not {FLAG}:" in l]
if len(sites) != total:
    raise SystemExit(f"ECHEC FERME : {len(sites)} garde(s) posee(s) pour {total} refus")

tmp = p.with_suffix(".py.tmp"); tmp.write_text(t, encoding="utf-8"); tmp.replace(p)
print(f"  refus neutralises : {total} · garde = constante de module {FLAG} · portee VERIFIEE")
PY

# ── E. le garde de FlashAttention, celui qui bloque APRÈS les six de QSA ─────────────────────
# `flash_attn.py:918-929` refuse tout cache KV quantifié dès que
# `flash_attn_supports_kv_cache_dtype()` rend False. Or `fa_utils.py:241-243` rend
#     (fa_version in (3,4) and family(90)) or (fa_version == 4 and family(100))
# et sm_121 est **famille 120** ⇒ False INCONDITIONNELLEMENT sur GB10.
#
# POURQUOI ON PEUT LE LEVER POUR QSA, ET SEULEMENT POUR QSA
#   Sur le chemin QSA, FlashAttention ne calcule AUCUNE attention : la lecture est
#   100 % Triton (`ops/qsa.py` → `qsa_sparse_paged_attention`), et FA n'y sert qu'aux
#   MÉTADONNÉES (tables de blocs, `slot_mapping`). Le garde protège donc un chemin de calcul
#   que nous n'emprunterons pas.
# ⚠️ MAIS `flash_attn.py` est PARTAGÉ par tous les backends FA du processus. C'est pour cela
#   que le garde n'est levé que sous l'opt-in EXPLICITE `VLLM_QSA_KV_FP8=1` : dans un serve où
#   un autre modèle utiliserait vraiment FA avec un KV quantifié, la variable ne serait pas
#   posée et le refus d'origine tiendrait.
python3 - "$FA" "$MARK" <<'PY'
import ast, pathlib, re, sys
p, mark = pathlib.Path(sys.argv[1]), sys.argv[2]
t = p.read_text(encoding="utf-8")
FLAG = "_QSA_KV_FP8_OK"

if FLAG not in t:
    _a = ast.parse(t)
    _fin = max((n.end_lineno for n in _a.body
                if isinstance(n, (ast.Import, ast.ImportFrom))), default=0)
    if not _fin:
        raise SystemExit("ECHEC FERME : aucun import de niveau module dans flash_attn.py")
    pos = sum(len(l) + 1 for l in t.splitlines()[:_fin])
    t = t[:pos] + (
        f"\n\n# {mark} : opt-in du KV quantifie pour le chemin QSA, ou FlashAttention ne"
        f"\n# calcule aucune attention (lecture 100 % Triton) et ne sert qu'aux metadonnees."
        f"\nimport os as _qsa_fa_os"
        f"\n{FLAG} = _qsa_fa_os.environ.get('VLLM_QSA_KV_FP8', '0') == '1'\n"
    ) + t[pos:]

motif = r'(\n(\s+))if is_quantized_kv_cache\(\n'
mm = re.search(motif, t)
if not mm:
    raise SystemExit("ECHEC FERME : ancre is_quantized_kv_cache introuvable dans flash_attn.py")
ind = mm.group(2)
t2, n = re.subn(
    r'(\n\s+)if is_quantized_kv_cache\(\n',
    f'\\1if not {FLAG} and is_quantized_kv_cache(  # {mark}\n',
    t, count=1)
if n != 1:
    raise SystemExit(f"ECHEC FERME : garde FA — {n} substitution(s), 1 attendue")

# controle de portee : la constante doit etre une GLOBALE du module
a = ast.parse(t2)
gl = {x.targets[0].id for x in a.body
      if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name)}
if FLAG not in gl:
    raise SystemExit(f"ECHEC FERME : {FLAG} n'est pas une globale de flash_attn.py")

tmp = p.with_suffix(".py.tmp"); tmp.write_text(t2, encoding="utf-8"); tmp.replace(p)
print(f"  E. flash_attn.py patche (garde de KV quantifie sous opt-in) · portee VERIFIEE")
PY

for f in "$QSA" "$CACHE" "$OPS" "$FA"; do
  python3 -c "import ast,pathlib,sys; ast.parse(pathlib.Path(sys.argv[1]).read_text())" "$f" \
    || { say "🔴 ECHEC FERME : $f ne parse plus"; exit 1; }
done
grep -q "$MARK" "$QSA" && grep -q "$MARK" "$CACHE" && grep -q "$MARK" "$OPS" && grep -q "$MARK" "$FA" \
  || { say "🔴 ECHEC FERME : marqueur absent d'un des trois fichiers"; exit 1; }
say "applique — fp8_e4m3 accepte sur le chemin QSA (echelle 1.0, plage 89,5/31,4 contre 448 mesuree sur le build FP8 du 08-28, PAS sur ce checkpoint : lancez probe/)"
