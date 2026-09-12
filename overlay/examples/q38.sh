#!/usr/bin/env bash
# Profil de lancement pour Qwen3.8-Flash-Next (arch Qwen4Exp), NVFP4 servi en W4A16.
#
# POURQUOI CE FICHIER EXISTE
# Sans profil dedie, `up.sh` retombe sur le profil generique, qui pose les parseurs de
# Hunyuan-3 : le serve DEMARRE, /health rend 200, puis la premiere completion meurt sur
#   « HYV3 Tool parser could not locate tool call start/end tokens in the tokenizer! »
# « loads + /health 200 » n'est pas « marche ».
#
# CONTRAINTES PROPRES A CETTE ARCHITECTURE, chacune lue dans le code de la branche :
#   - QSA refuse `--kv-cache-dtype fp8` par six `NotImplementedError` (`nvidia/ops/qsa.py`) ;
#     la recette les leve avec `mods/qsa-kv-fp8` (KV_DTYPE=fp8_e4m3, voir le bloc KV plus bas).
#   - le chemin rapide `cooperative_topk` est **exclu nommement pour la famille sm_120**
#     (`nvidia/ops/qsa.py:788-792`) : on tourne sur `persistent_topk`, cout non documente.
#   - le MTP est present (`mtp_num_hidden_layers: 1`) et ses experts sont **deja en F8_E4M3**
#     dans le checkpoint officiel. A TP=2 le backend MoE du drafter est force a Triton par
#     `fp8.py:520-533`.
set -euo pipefail

# EAGER=1 (defaut) : pas de compilation Inductor.
# POURQUOI CE DEFAUT — mesure du 2026-08-26. Avec la compilation active, le boot atteint
# `Application startup complete` puis meurt a l'autotuning :
#   InductorError: CUDA out of memory. Tried to allocate 47.69 GiB.
#   GPU 0 has a total capacity of 121.69 GiB of which 23.56 GiB is free.
#   this process has 86.71 GiB memory in use.
# Mesure sur le build FP8 (86,43 GiB/noeud) ; en W4A16 les poids font 63,70 GiB/rang, mais le
# pic de compilation reste hors du budget KV. Ce n'est donc
# PAS un manque de budget KV mais un pic de COMPILATION, et il ne se resout pas en baissant
# `gpu-memory-utilization` — le pic est hors de ce budget.
# Le journal donne aussi le fait qui l'explique : « Checkpoint size: 170.29 GiB.
# Available RAM: 19.57 GiB » — le chargement a rempli le page cache juste avant.
# ⚠️ Ce defaut a un COUT en debit, non chiffre ici. Sur un autre modele, le bras eager valait 13,71 tok/s
# contre 14,75 avec graphes (+7,6 %). A re-mesurer proprement une fois le serve stable, en
# comparant des bras appariés (`EAGER=0` quand la memoire le permettra).
ARGS_EAGER=()
[ "${EAGER:-1}" = "1" ] && ARGS_EAGER=(--enforce-eager)

# SPEC_TOKENS=N : active le MTP du checkpoint (`mtp_num_hidden_layers: 1`).
# ⚠️ Le plafond est STRUCTUREL a n=1 couche de draft : ne pas demander plus que ce que le
# checkpoint porte. Les experts du drafter sont DEJA en F8_E4M3 dans le checkpoint officiel,
# donc aucune consigne `SPEC_MOE` a transposer ; a TP=2 le
# backend MoE est de toute facon force a Triton par `fp8.py:520-533`.
# ── IDX_SHARE=1 : reutiliser les indices creux QSA du premier pas sur les pas de BROUILLON.
# Champ `index_share_for_mtp_iteration` (`config/speculative.py:141`, `bool | None = None`).
# A None il retombe sur le config du checkpoint, et notre checkpoint ne le porte PAS => OFF.
# Ce qu'il economise : quand il est actif, `indexer_qsa.py:337` fait `return out` et saute
# entierement `qsa_select_paged_tokens`, donc l'alloc `torch.empty(rows, columns, float32)`
# de `ops/qsa.py:636` ET le top-k, sur les 12 couches QSA, pour 3 pas de brouillon sur 4.
# C'est du travail retire des 76,1 % du mur qui ne sont PAS de la lecture de poids.
# ⚠️ RISQUE : reutiliser des indices creux sur les jetons brouillonnes est une APPROXIMATION.
# Elle peut baisser l'acceptation (MAL) ou degrader la sortie => valider MAL + gate + aiguille,
# jamais le seul debit.
ARGS_SPEC=()
if [ "${SPEC_TOKENS:-0}" -gt 0 ] 2>/dev/null; then
  SPEC_JSON="{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}"
  [ -n "${IDX_SHARE:-}" ] && SPEC_JSON="${SPEC_JSON},\"index_share_for_mtp_iteration\":$([ "$IDX_SHARE" = "1" ] && echo true || echo false)"
  # SPEC_MODEL : le checkpoint du DRAFTER, quand il doit differer de la cible.
  # POURQUOI. `get_draft_quant_config` resout la config de quantification depuis
  # `draft_model_config` -- sa docstring le dit : « Draft models should use their own
  # quantization config instead of the verifier/target model's config ». Et le post-init de
  # `config/speculative.py` ne force `self.model = self.target_model_config.model` QUE
  # `if self.model is None`. Donc renseigner `model` donne au drafter SON checkpoint et SA
  # config, independamment de la cible.
  # MESURE DU 2026-08-31 qui rend ce champ necessaire : avec la cible ET le drafter sur
  # `q38-experts4bit` (experts MTP quantifies en 4 bits), le drafter a rendu 30 993 jetons de
  # brouillon pour ZERO accepte, a toutes les positions => MAL 1,000 contre 2,728, debit
  # 19,4 tok/s contre 44,7. Le prefill, lui, gagnait 4,8x : la greffe 4 bits de la CIBLE est
  # saine, seul le drafter etait casse.
  [ -n "${SPEC_MODEL:-}" ] && SPEC_JSON="${SPEC_JSON},\"model\":\"${SPEC_MODEL}\""
  ARGS_SPEC=(--speculative-config "${SPEC_JSON}}")
fi

# ── EP=1 : expert parallelism. LE levier du rendement kernel (mesure du 2026-08-27).
# POURQUOI. `quantization/fp8.py:520-533` force Triton des que
# `intermediate_size_per_partition % 128 != 0`, APRES le choix de l'oracle :
#   refine = gcd(128,128,320,2560) = 64  ->  moe_block_shape=[64,64] + fp8_backend=TRITON
# Chez nous 640/2 = 320 a TP=2, donc l'override tire TOUJOURS. L'EP pose tp_size=1/ep_size=2
# pour le MoE (`fused_moe/config.py:1208-1252`) => partition 640, 640 % 128 == 0, l'override
# ne tire plus, le bloc reste [128,128] et BLOCK_SIZE_K repasse de 64 a 128 sur 48 couches.
# ⚠️ Ce n'est PAS un gain de bande passante : a dp_size=1 il n'y a pas d'all-to-all
# (`use_all2all_kernels() == False`). Une objection de topologie a ete faite puis retiree.
# Corrobore : @zzw09773 sur vllm#53896 (« Expert parallelism is required » pour ce checkpoint)
# et la recette 2x DGX Spark FP8 du forum NVIDIA 381228/97 qui l'embarque deja.
ARGS_EP=()
[ "${EP:-0}" = "1" ] && ARGS_EP=(--enable-expert-parallel
                                 --all2all-backend "${ALL2ALL:-allgather_reducescatter}")

# ── MAMBA_DTYPE : dtype de l'etat recurrent GDN. Il FIXE le block size.
# `interface.py:917-925` : block = 16 x ceil(page_mamba / 16384). L'etat SSM en float32 fait
# 1 572 864 o => page 1 634 304 => block 1600. En bfloat16 : page 847 872 => block 832,
# soit +5,5 % de pool ET la granularite du prefix-cache divisee par deux (1600 -> 832).
# ⚠️ Le code dit « Only float32 is known to have no accuracy issues by default » : ce bras
# EXIGE la gate de degenerescence et une sonde de recuperation longue avant d'etre garde.
ARGS_MAMBA=()
[ -n "${MAMBA_DTYPE:-}" ] && ARGS_MAMBA=(--mamba-ssm-cache-dtype "$MAMBA_DTYPE")

# ── CG=1 : graphes CUDA SANS Inductor.
# `config/vllm.py:1269-1275` : --enforce-eager pose mode=NONE **et** cudagraph_mode=NONE.
# Or le pic de 47,69 GiB est de l'AUTOTUNING Inductor, pas de la capture. Donc mode=0 +
# cudagraph_mode=FULL_DECODE_ONLY donne les graphes sans jamais toucher ce pic.
# `compilation.py:62` FULL_DECODE_ONLY = (FULL, NONE) => mixed_mode()==NONE, legal avec MTP.
# ⚠️ RISQUE NOMME : vllm#46253 « illegal memory access at capture_end — host-staged NCCL
# all-reduce on GB10 (no GPUDirect) », et notre .env pose NCCL_NET_GDR_DISABLE=1 : nous
# remplissons la condition d'echec. Et quand ils ont obtenu la capture, le REPLAY rendait
# des logits nuls. Un boot qui demarre n'est donc PAS une preuve : gate obligatoire.
ARGS_CG=()
if [ "${CG:-0}" = "1" ]; then
  ARGS_EAGER=()
  # ── CG_SIZES : les tailles de lot CAPTUREES. Le defaut de vLLM en pose 50, jusqu'a
  # 272+. MESURE LE 2026-08-27 : ce pic de capture a fait OOM-KILLER LA BOX (rang 0 tue
  # par le noyau, `global_oom`, `NVRM ... NV_ERR_NO_MEMORY` sur les DEUX nœuds) alors que
  # la memoire FINALE des graphes ne valait que **0,26 GiB** — c'est le PIC qui tue, pas
  # le residu. Aggravant : hors eager la profilation mesure un pic plus bas, donc vLLM
  # avait deja alloue **0,52 GiB de KV EN PLUS** (10,6 contre 10,08 GiB, pool 728 869
  # contre 710 200) avant meme de capturer.
  # Le defaut ci-dessous suit le regime mesure : `Running=1` 93,4 % du temps
  # et MTP n=3 => lot de decode ~4. Les lots plus grands que la derniere taille capturee
  # retombent en eager : c'est le comportement PREVU de vLLM, pas une perte silencieuse.
  ARGS_CG=(--compilation-config \
    "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CG_SIZES:-1,2,4,8,16,32}]}")
fi

# ── KV_BYTES : plafonner explicitement le pool KV.
# `KV_BYTES` doit etre LU ici : une variable presente dans la liste blanche du lanceur mais
# non consommee par le profil atteint le conteneur et n'y change rien. C'est le piege
# « blanchi mais non consomme » : la valeur atteignait le conteneur sans rien changer.
# POURQUOI IL Y A DU GRAIN A PRENDRE : `gpu-memory-utilization` budgete sur le TOTAL
# (0,85 x 121,69 = 103,44 GiB) alors que le LIBRE au demarrage vaut 106,46 GiB. vLLM le
# signale de lui-meme (`gpu_worker.py:920`) : « Replace gpu_memory_utilization config with
# --kv-cache-memory=14169529856 (13.2 GiB) to fully utilize gpu memory. Current kv cache
# memory in use is 10.56 GiB. »
# ⚠️ POURQUOI NE PAS PRENDRE LE MAXIMUM : ces 2,64 GiB sont HORS du budget gmu, pris sur
# ~3 GiB de jeu PHYSIQUE. Or `MemAvailable` vaut deja 1,8-2,1 GiB en regime sur A, le
# transitoire du warmup multimodal vaut ~4 GiB, et un tiers mesure 15/48 cellules tuees par
# earlyoom a gmu 0,85 contre 0/48 a 0,80. Ce flag COURT-CIRCUITE la profilation memoire
# (`gpu_worker.py:605-625`) : il n'y a plus de garde-fou automatique. Toute valeur posee ici
# doit etre validee par la vision ET une aiguille longue, pas par le seul boot.
ARGS_KV=()
[ -n "${KV_BYTES:-}" ] && ARGS_KV=(--kv-cache-memory-bytes "$KV_BYTES")

# ── BATCHED_TOKENS / LONG_PREFILL : decoupler le BUDGET AGREGE du CHUNK PAR REQUETE.
# `BATCHED_TOKENS` doit etre LU ici, pour la meme raison. Sans cette lecture le budget valait
# donc 2048, le defaut de la branche « autre materiel » (`arg_utils.py:2592-2614`) : GB10 et
# ses 121,69 GiB devraient prendre la branche >= 70 GiB et recevoir **8192**, mais
# `get_device_total_memory()` echoue sous Ray (le code nomme lui-meme le cas) et rend 0.
# vLLM le reclame de son cote : « max_num_scheduled_tokens is set to 2048 based on the
# speculative decoding settings. This may lead to suboptimal performance. Consider
# increasing max_num_batched_tokens to accommodate the additional draft token slots. »
# POURQUOI CA COMPTE ICI : la charge mesuree est a **40:1 entree/sortie**, donc
# le prefill n'est marginal que quand le cache de prefixes absorbe ; sur du contenu froid il
# pese autant que le decode.
# 🔴 RETRACTE : une version precedente de ce commentaire bridait le chunk en invoquant « la
# falaise de l'indexeur QSA : transitoire fp32 [chunk x history], 8,00 GiB a chunk 8192 ».
# CETTE FALAISE N'EXISTE PAS DANS CE MOTEUR : `nvidia/ops/qsa.py:14` epingle
# `_LOGITS_WORKSPACE_BYTES = 128 MiB`, consomme en BORNE DE BOUCLE a `:770`
# (`rows_per_chunk = _LOGITS_WORKSPACE_BYTES // (columns * 4)`), donc le transitoire est
# INDEPENDANT du chunk ET du contexte. Le 8-10 GiB decrit ailleurs est le noyau TileLang de
# SGLang, pas ce chemin. Les deux vraies raisons de `LONG_PREFILL` sont ci-dessous.
# D'ou LONG_PREFILL : `scheduler.py:563-564` plafonne DUREMENT les jetons par pas d'une
# requete (`if 0 < seuil < num_new_tokens: num_new_tokens = seuil`), ce qui laisse le budget
# agrege a 8192 pour BATCHER plusieurs requetes tout en gardant le chunk d'UNE requete
# longue a 2048. Les deux doivent donc bouger ENSEMBLE.
# Cout persistant a 8192 : `topk_indices_buffer` passe de 192 a 769 MiB/rang (~6 % du pool).
ARGS_BATCH=()
[ -n "${BATCHED_TOKENS:-}" ] && ARGS_BATCH+=(--max-num-batched-tokens "$BATCHED_TOKENS")
[ -n "${LONG_PREFILL:-}" ] && ARGS_BATCH+=(--long-prefill-token-threshold "$LONG_PREFILL")

# ── KV_DTYPE : le dtype du cache KV. NON CABLE JUSQU'AU 2026-08-28.
# `up.sh` exportait `-e KV_DTYPE` dans le conteneur depuis toujours, et ce profil ne
# le lisait JAMAIS : quatre bras d'une campagne KV auraient boote identiques au controle.
# ⚠️ `fp8` exige un PATCH : `STR_DTYPE_TO_TORCH_DTYPE["fp8_e4m3"] = torch.uint8`, le chemin
# FA generique reinterprete la vue (`v1/attention/backends/flash_attn.py:1056-1058`
# `.view(current_platform.fp8_dtype())`) mais **le chemin QSA ne le fait pas**
# (`models/qwen4_exp/nvidia/qsa.py:142-144`), et `qsa_sparse_paged_attention` n'a AUCUN
# parametre `k_scale`/`v_scale`. Sans le patch, sept `NotImplementedError` refusent :
# `nvidia/qsa.py:108,146,183,185,187,280` + `common/qsa_cache.py:658`.
# ⚠️ Et le block size est DERIVE de la page mamba : en fp8 il passe de 832 a **1664**
# (`platforms/interface.py:906-911`), donc `bytes_per_block` AUGMENTE et `num_gpu_blocks`
# BAISSE — la capacite monte quand meme (blocs x block_size), mais l'observable a suivre est
# `num_gpu_blocks x block_size`, jamais `bytes_per_block`.
ARGS_KVD=()
[ -n "${KV_DTYPE:-}" ] && [ "${KV_DTYPE}" != "auto" ] && ARGS_KVD=(--kv-cache-dtype "$KV_DTYPE")

# ── ROPE_FACTOR : etendre le contexte par YaRN.
# Il n'existe AUCUN flag `--rope-scaling` dans cette version (0 occurrence dans
# `engine/arg_utils.py`) : la seule voie est `--hf-overrides`.
# ⚠️ TROIS PIEGES, tous verifies dans le code :
# 1. La cible est `text_config.rope_parameters`. Posee a la racine, la surcharge atterrit sur
#    un attribut dict (`models/qwen4_exp/config.py:242-244`) et n'atteint jamais
#    `hf_text_config`, seul lu par `config/model.py:1955`. Sous la cle `rope_scaling`, elle est
#    silencieusement ignoree.
# 2. `config/model.py:488` (`_update_nested`) fait un `setattr` qui REMPLACE la feuille sans
#    fusionner => il faut RE-POSER `mrope_section`, `mrope_interleaved`, `partial_rotary_factor`
#    et `rope_theta`. Sans `mrope_section`, `rotary_embedding/__init__.py:273` bascule sur
#    `YaRNScalingRotaryEmbedding`, `supports_mrope` devient False et **LA VISION CASSE**.
#    Sans `partial_rotary_factor`, l'assert `sum(mrope_section) == rotary_dim // 2` saute.
# 3. Le facteur doit satisfaire `262144 x factor >= MAX_LEN`, sinon `config/model.py:2461`
#    leve une `ValueError`. Pour 700 000 le minimum EXACT est 2.6702880859375
#    (262144 = 2**18, donc exactement representable). 2.67 donne 699 924,48 et ECHOUE.
ARGS_ROPE=()
if [ -n "${ROPE_FACTOR:-}" ]; then
  ARGS_ROPE=(--hf-overrides "{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":${ROPE_FACTOR},\"original_max_position_embeddings\":262144,\"mrope_section\":[11,11,10],\"mrope_interleaved\":true,\"partial_rotary_factor\":0.25,\"rope_theta\":10000000}}}")
fi

# ── MOE_BACKEND : choix EXPLICITE du noyau MoE, et c'est une SURETE, pas un reglage.
# 🔴 POURQUOI CETTE BLOC EXISTE (2026-09-05). `up.sh` met `MOE_BACKEND` dans les `-e`
# depuis toujours -- et AUCUN profil ne la lisait : la valeur atteignait le conteneur et n'y
# changeait rien. C'est exactement le meme defaut que celui documente plus bas pour
# `EXTRA_ARGS` (2026-08-30), sur la variable d'a cote. Verifier les DEUX bouts,
# la liste blanche ET le consommateur.
# ⚠️ Et il ne suffit PAS de passer par `EXTRA_ARGS` : celui-ci est de-quote a l'interieur de
# `launch-cluster.sh`, donc un EXTRA_ARGS de DEUX mots voit son second mot mange par
# `docker run` lui-meme (`unknown flag: --moe-backend`, mesure le 2026-09-05). EXTRA_ARGS ne
# peut porter qu'UN seul drapeau ; toute option supplementaire a besoin de sa variable.
# CE QUE CA ACHETE : sur un checkpoint NVFP4, l'oracle a 5 chemins A4 devant Marlin
# (FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL, ..., VLLM_CUTLASS). Marlin est le seul W4A16 =>
# le seul agentiquement admissible ici. Temoin dans l'echo du moteur :
#   [nvfp4.py:244] Using 'MARLIN' NvFp4 MoE backend out of potential backends: [...]
# Controle du drapeau lui-meme (et non de l'env) :
#   docker inspect vllm_node --format '{{join .Config.Cmd " "}}' | grep moe-backend
ARGS_MOE=()
if [ -n "${MOE_BACKEND:-}" ]; then
  ARGS_MOE=(--moe-backend="${MOE_BACKEND}")
fi

exec vllm serve "${MODEL_PATH:?}" \
  --served-model-name q38 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  --max-model-len "${MAX_LEN:-65536}" \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --max-num-seqs "${MAX_SEQS:-256}" \
  --no-enable-flashinfer-autotune \
  "${ARGS_EP[@]}" \
  "${ARGS_MAMBA[@]}" \
  "${ARGS_EAGER[@]}" \
  "${ARGS_CG[@]}" \
  "${ARGS_KV[@]}" \
  "${ARGS_BATCH[@]}" \
  "${ARGS_KVD[@]}" \
  "${ARGS_ROPE[@]}" \
  "${ARGS_SPEC[@]}" \
  "${ARGS_MOE[@]}" \
  --no-enable-log-requests \
  ${EXTRA_ARGS:-}
# ⚠️ ${EXTRA_ARGS} EST NON QUOTE ICI, mais ca ne lui permet PAS de porter plusieurs drapeaux.
# Un EXTRA_ARGS a plusieurs drapeaux ne marche PAS : `launch-cluster.sh` de-quote la valeur bien AVANT ce
# point, dans sa propre ligne `docker run` : un EXTRA_ARGS de deux mots fait echouer le boot
# en 23 s sur `unknown flag: --moe-backend` / `Usage: docker run [OPTIONS]`, le second mot
# ayant ete pris pour une option de docker. ⇒ **EXTRA_ARGS ne porte qu'UN seul drapeau**, et
# toujours en `--flag=valeur` (jamais `--flag valeur`). Toute option supplementaire exige sa
# propre variable, cote liste blanche ET cote consommateur — voir ARGS_MOE ci-dessus.
# 🔴 POURQUOI CETTE LIGNE EXISTE (2026-08-30) : `up.sh` met EXTRA_ARGS dans sa liste
# blanche depuis toujours, la variable atteignait donc le conteneur — mais ce profil ne la
# LISAIT PAS. Un `EXTRA_ARGS=--moe-backend=marlin` a traverse tout le chemin, s'est retrouve
# dans `docker inspect .Config.Env`, et n'est JAMAIS arrive a argparse : `marlin` apparaissait
# 0 fois dans `.Config.Cmd`. Le boot a reussi, le serve a repondu 200, et il a mesure la
# REFERENCE en croyant mesurer un bras. C'est la regle « une variable dans la liste blanche
# n'est pas une variable CONSOMMEE » — verifier les DEUX bouts, la liste blanche ET le
# consommateur. Controle : `docker inspect vllm_node --format '{{join .Config.Cmd " "}}'`
# doit contenir le drapeau, pas seulement `.Config.Env`.
