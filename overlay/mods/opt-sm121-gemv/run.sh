#!/bin/bash
# opt-sm121-gemv — remplace le GEMV BF16 de cuBLAS par un kernel Triton, au seul regime m=1.
#
# LE FAIT MESURE (2026-07-30, GPU nu, jeu de travail 600 MiB >> L2 de 24 MiB) :
# cuBLAS n'atteint que 161-166 GB/s sur les projections d'attention de ce modele, alors que
# la lecture pure de la puce vaut 238 GB/s et qu'un GEMV Triton de vingt lignes en atteint 218.
#
#     forme (par rang TP=2)   cuBLAS m=1   Triton    gain
#     in_proj_qkv [6144,4096]   305,0 us   223,1 us  1,37x   x45 couches
#     in_proj_z   [4096,4096]   207,5 us   152,6 us  1,36x   x45
#     out_proj    [4096,4096]   208,4 us   152,8 us  1,36x   x45
#     o_proj      [4096,4096]   208,8 us   152,7 us  1,37x   x15
#     lm_head   [125696,4096]  5876,9 us  4278,6 us  1,37x   x1
#     q_proj      [8192,4096]   295,4 us   292,0 us  1,01x   x15  (cuBLAS y est deja bon)
#
# Total 11,15 ms sur un pas de decode mesure a 62,3 ms => 16,05 -> 19,55 tok/s attendus (+21,8 %).
#
# POURQUOI cuBLAS est mauvais ici : a m=1 le GEMV est latency-bound, pas bandwidth-bound. Preuve
# par l'absurde mesuree : sur [4096,4096], un GEMM a m=2 prend 157,9 us contre 208,0 a m=1 — il
# est PLUS RAPIDE en calculant DEUX FOIS plus. Le kernel m=1 de cuBLAS laisse 31 % de la bande
# passante sur la table. Ce n'est pas un probleme d'octets : quantifier ne le corrigerait pas.
#
# CORRECTION VERIFIEE, pas supposee : l'erreur relative du kernel Triton contre une verite fp32
# est IDENTIQUE a celle de cuBLAS au chiffre pres (1,698e-03 contre 1,698e-03, ratio 1,00 sur
# six formes). Meme arrondi bf16, aucune perte de precision.
#
# CONCEPTION, et pourquoi elle est minuscule :
#   * UN prédicat, pas de liste blanche de formes : `un seul token && N>=1024 && K>=1024`.
#     Mesure a l'appui : sous 1024 le gain disparait (shared experts 1,01x, k/v_proj nul) et
#     q_proj, la seule forme ou cuBLAS etait deja optimal, ne perd RIEN (1,01x). Une liste
#     blanche serait plus fine et beaucoup plus fragile a relire.
#   * UNE config figee (BN=32, BK=1024, num_warps=8). Pas d'autotune : il se declencherait au
#     premier appel, donc possiblement pendant la capture des graphes CUDA, et il violerait la
#     regle maison « aucun recalcul par boot ». Les cinq configs candidates rendent 1,25-1,27x,
#     donc le choix n'est pas sur le fil du couteau.
#   * enregistre en `torch.ops.vllm.*` avec une impl `_fake`, exactement comme le fait deja
#     `rocm_unquantized_gemm` (utils.py:212). C'est ce qui permet a inductor de le traiter comme
#     opaque et aux graphes CUDA de le capturer.
#   * repli sur `F.linear` pour TOUT le reste : biais, dtype non-bf16, m>1, poids non contigu.
#
# Revert = redemarrer le conteneur sans ce mod. Rien n'est ecrit dans l'image.
set -euo pipefail

# -- resolution du repertoire vllm : `VLLM_DIR` d'abord ------------------------
# POURQUOI CETTE VARIABLE. Un controle anti-derive extrait l'arbre vllm dans un TEMPORAIRE et
# lance ce script dessus ; sans `VLLM_DIR` le script resout `import vllm` et patche l'arbre du
# systeme, donc le controle ne mesure pas ce qu'il annonce. Dans le conteneur, le defaut
# resout exactement le chemin servi.
P="${VLLM_DIR:-$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')}"
[ -d "$P/model_executor" ] || { echo "ECHEC FERME : $P n'est pas un arbre vllm."; exit 1; }
V="$P"
U="$V/model_executor/layers/utils.py"
MOD="$V/model_executor/layers/gemv_sm121.py"
say() { echo "[opt-sm121-gemv] $*"; }

if grep -q "gemv_sm121" "$U" 2>/dev/null; then
  say "deja applique, rien a faire (idempotent)"
  exit 0
fi

say "patching $P"

# --------------------------------------------------------------------------- le kernel
cat > "$MOD" <<'PY'
"""GEMV BF16 pour sm_121 au regime m=1. Voir mods/opt-sm121-gemv/run.sh pour les mesures."""

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# Config FIGEE, choisie par balayage sur les formes reelles du modele (BN/BK/warps).
# Pas d'autotune a l'execution : voir le run.sh.
_BN, _BK, _WARPS = 32, 1024, 8
# Sous ces bornes le gain disparait (mesure) et cuBLAS reste meilleur ou equivalent.
_MIN_N, _MIN_K = 1024, 1024
# SEUIL DE TAILLE : 6 MiB, pas le L2 (24 MiB). Un banc qui alloue UN seul poids et le reutilise
# rend une forme de 12 MiB RESIDENTE EN L2 des le second tir (cuBLAS y mesure 660 GB/s, ~3x la
# DRAM) ; en production rien n'est chaud : 27 instances de chaque famille QSA et 96 hyper-
# connexions (1,18 GiB) defilent entre deux visites d'un meme poids. Mesure au micro-banc a
# ROTATION sur ~1,2 GiB de tampons distincts (methode dans docs/HARDWARE.md)
# (medianes sur 30 tirs, formes ENUMEREES des en-tetes safetensors du checkpoint servi) :
#     forme                MiB  1 tampon   en rotation
#     hc_mixer  [ 2560,2560]  12,5   0,97x  ->  1,31x   <- le cas qui avait pose le garde
#     qsa       [ 1152,4304]   9,5   0,68x  ->  1,29x   <- changeait meme de SIGNE
#     qsa       [ 4304,1152]   9,5   1,06x  ->  1,25x
#     qsa       [ 3456,1152]   7,6   0,95x  ->  1,23x
#     proj      [ 2560,4608]  22,5   1,30x  ->  1,38x
#     qsa       [ 1152,1152]   2,5   0,80x  ->  0,90x   <- PERD dans LES DEUX regimes
# Precision identique partout (err_rel 1,37e-03 a 1,45e-03), donc rien a arbitrer de ce cote.
# Le vrai discriminant n'est donc PAS le L2 mais le cout de LANCEMENT : a 2,5 MiB le noyau dure
# 22-24 us, l'ordre du lancement lui-meme, et l'appel Triton paie son surcout sans rien gagner.
# On garde donc un PLANCHER, mesure, tres en-dessous du L2. 6 MiB rejette le seul perdant
# (2,5 MiB, 0,90x) et le quasi-nul (5,1 MiB, 1,03x), et admet les cinq gains de 23 a 38 %.
# ⚠️ Effet attendu bout-en-bout : +1,78 % (162 MiB effectifs sur 9 089 par pas) -- SOUS notre
# plancher de bruit inter-boot de +-4 %, donc NON validable par une paire de boots. Ce garde
# est adopte sur la preuve de NOYAU, pas sur une mesure de bout en bout ; le boot ne sert qu'a
# verifier l'absence de REGRESSION et l'integrite du MAL.
_PLANCHER_OCTETS = 6 * 1024 * 1024

_kernel = None


def _get_kernel():
    """Compilation paresseuse : importer triton au chargement du module allongerait tout
    demarrage, y compris ceux qui n'emprunteront jamais ce chemin."""
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _gemv(Wp, xp, yp, N, K, sW0, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rn = pid * BN + tl.arange(0, BN)
        mn = rn < N
        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < K
            w = tl.load(
                Wp + rn[:, None] * sW0 + rk[None, :],
                mask=mn[:, None] & mk[None, :],
                other=0.0,
            ).to(tl.float32)
            xv = tl.load(xp + rk, mask=mk, other=0.0).to(tl.float32)
            acc += tl.sum(w * xv[None, :], axis=1)
        tl.store(yp + rn, acc.to(tl.bfloat16), mask=mn)

    _kernel = (triton, _gemv)
    return _kernel


def _eligible(x: torch.Tensor, weight: torch.Tensor, bias) -> bool:
    return (
        bias is None
        and x.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
        and weight.ndim == 2
        and weight.is_contiguous()
        # exactement un token : couvre [K], [1,K] et [1,1,K] sans supposer le rang
        and x.numel() == x.shape[-1]
        and weight.shape[0] >= _MIN_N
        and weight.shape[1] >= _MIN_K
        # plancher de LANCEMENT, pas de L2 : sous ~6 MiB le noyau dure l'ordre du lancement
        # lui-meme et le surcout Triton domine (mesure en rotation ci-dessus)
        and weight.numel() * weight.element_size() > _PLANCHER_OCTETS
    )


def sm121_gemv_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    if not _eligible(x, weight, bias):
        return torch.nn.functional.linear(x, weight, bias)
    N, K = weight.shape
    triton, gemv = _get_kernel()
    y = torch.empty((*x.shape[:-1], N), dtype=x.dtype, device=x.device)
    gemv[(triton.cdiv(N, _BN),)](
        weight, x.reshape(-1), y.reshape(-1), N, K, weight.stride(0),
        BN=_BN, BK=_BK, num_warps=_WARPS,
    )
    return y


def sm121_gemv_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="sm121_gemv",
    op_func=sm121_gemv_impl,
    fake_impl=sm121_gemv_fake,
)


def sm121_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.sm121_gemv(x, weight, bias)
PY

# --------------------------------------------------------------------------- l'aiguillage
# On n'insere qu'UNE branche dans dispatch_unquantized_gemm, avant le repli `else`.
python3 - "$U" <<'PY'
import re
import sys

p = sys.argv[1]
s = open(p).read()
old = """    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm"""
new = """    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    elif current_platform.is_cuda():
        # opt-sm121-gemv : cuBLAS ne tient que 161-166 GB/s sur les GEMV m=1 de ce modele
        # contre 218 pour un kernel Triton (238 en lecture pure). Le kernel retombe lui-meme
        # sur F.linear pour tout ce qui n'est pas eligible.
        from vllm.model_executor.layers.gemv_sm121 import sm121_unquantized_gemm

        return sm121_unquantized_gemm
    else:
        return default_unquantized_gemm"""
assert s.count(old) == 1, f"motif d'aiguillage introuvable ou ambigu ({s.count(old)})"
open(p, "w").write(s.replace(old, new))
print("  aiguillage insere")
PY

# --------------------------------------------------------------------------- auto-validation
say "auto-validation"
python3 - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    print("  [SKIP] pas de GPU visible : validation numerique impossible")
    sys.exit(0)

from vllm.model_executor.layers.gemv_sm121 import _eligible, sm121_unquantized_gemm
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

# 1. l'aiguillage renvoie bien notre fonction
got = dispatch_unquantized_gemm()
assert got is sm121_unquantized_gemm, f"aiguillage inactif : {got}"
print("  OK aiguillage -> sm121_unquantized_gemm")

# 2. correction numerique contre F.linear, sur les formes REELLES du modele
ok = True
for N, K in ((6144, 4096), (4096, 4096), (8192, 4096), (125696, 4096)):
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
    ref = torch.nn.functional.linear(x, W)
    got = sm121_unquantized_gemm(None, x, W, None)
    assert got.shape == ref.shape, f"forme {got.shape} != {ref.shape}"
    # les deux encaissent le meme arrondi bf16 ; on borne l'ecart RELATIF
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    good = rel < 5e-3
    ok &= good
    print(f"  {'OK' if good else 'ECHEC'} [{N},{K}] ecart relatif {rel:.2e}")
    del W, x, ref, got
    torch.cuda.empty_cache()

# 3. les cas NON eligibles doivent retomber sur F.linear, a l'identique bit pour bit
W = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
b = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
for lab, x, bias in (
    ("m=4 (lot)", torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda"), None),
    ("avec biais", torch.randn(1, 4096, dtype=torch.bfloat16, device="cuda"), b),
    ("fp16", torch.randn(1, 4096, dtype=torch.float16, device="cuda"), None),
):
    Wl = W.half() if lab == "fp16" else W
    assert not _eligible(x, Wl, bias), f"{lab} devrait etre INELIGIBLE"
    r = sm121_unquantized_gemm(None, x, Wl, bias)
    e = torch.nn.functional.linear(x, Wl, bias)
    same = torch.equal(r, e)
    ok &= same
    print(f"  {'OK' if same else 'ECHEC'} repli {lab} : identique a F.linear = {same}")

# 4. petite forme : doit etre ineligible (mesure : aucun gain sous 1024)
assert not _eligible(
    torch.randn(1, 4096, dtype=torch.bfloat16, device="cuda"),
    torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda"), None)
print("  OK N=512 exclu par le predicat")

print("  === opt-sm121-gemv VALIDE ===" if ok else "  === ECHEC ===")
sys.exit(0 if ok else 1)
PY
say "applique"
