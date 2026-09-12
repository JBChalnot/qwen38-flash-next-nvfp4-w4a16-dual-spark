# Qwen3.8-Flash-Next NVFP4 as W4A16 on two DGX Sparks (vLLM, TP=2)

Serves [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
with its calibrated 4-bit weights and **16-bit activations** (W4A16) instead of the W4A4 the
checkpoint is designed for. No weight byte is copied or modified: `tools/make-w4a16-variant.py`
flips 48 labels in `config.json`, and `--moe-backend=marlin` selects the W4A16 kernel.

Per rank: **63.70 GiB of weights**, KV pool **3,547,511 tokens** (26 GiB, fp8 KV), MTP mean
accepted length **2.99**, decode **51.1 tok/s** median single-stream, **99.4 tok/s** aggregate at
4 streams. Tested on one topology: 2× GB10, TP=2, RoCE. No support.

## Prerequisites

- 2× DGX Spark (GB10, sm_121, 121.69 GiB unified memory, **no swap**), ConnectX ports linked.
- Docker with the NVIDIA runtime on both. Key-based ssh from the head to the worker.
- ~124 GiB of disk per box: the checkpoint must exist on **both** ranks.
- **This repository cloned at the same absolute path on both boxes.** `up.sh` runs the
  page-cache release on the worker at `$ICI/tools/...` over ssh and refuses to boot otherwise.
- Python 3 standard library for `tools/`, plus **Pillow** for `tools/multi-image.py`.

`./install.sh --check` verifies ssh, the RDMA device, free memory and the checkpoint on both
ranks. It does not check the clone path on the worker (`up.sh` does, and refuses) nor Pillow.

## Quick start

```bash
cp recipe.env recipe.env.local && $EDITOR recipe.env.local   # the lines marked CHANGE ME
./install.sh --check      # prerequisites; builds nothing
./install.sh              # clone the launcher, build the engine from vLLM's Dockerfile at a pinned
                          # commit (3 stages, hours; skipped if $IMAGE already holds that commit).
                          # Stages 1 and 3 were re-run from this tree; stage 2 was not (docs/HARDWARE.md)
CHECKPOINT_DIR=... SERVED_DIR=... python3 tools/make-w4a16-variant.py
./up.sh                   # under a supervisor, never under a shell timeout (see Gotchas)
./view.sh                 # health, what the engine consumed, RDMA vs TCP, memory on both ranks
./down.sh                 # stops both ranks and verifies it
```

`recipe.env.local` is an overlay: it is sourced after `recipe.env` and overrides it.

## Security posture

The engine container runs with **`--privileged`, `--network host`, `--ipc host`** on both boxes
(RDMA, shared-memory KV transfer, Ray wiring). Consequences:

- The OpenAI API is **unauthenticated** unless `VLLM_API_KEY` is set in the container
  environment. `recipe.env` binds it to `HOST=127.0.0.1`; put an authenticating proxy in front
  before changing it to `0.0.0.0`. The published figures were taken with `0.0.0.0` on a private
  network; the loopback default was booted from a fresh clone with identical results
  ([docs/HARDWARE.md](docs/HARDWARE.md#verified-from-a-fresh-clone)).
- `--network host` exposes on every NIC of the head, as observed with `ss -ltn`: the API
  (`PORT` 8000, also `/metrics`), `MASTER_PORT` 29501 (NCCL rendezvous), Ray worker ports
  10002-10005 and a handful of ephemeral Ray ports in the 3xxxx-4xxxx range. Ray's dashboard
  is disabled. Firewall the box or
  keep it on a private network.
- `--privileged` gives the container the host's devices, including `/dev/infiniband`.

**The default KV cache is FP8, via a patch of vLLM's attention backend** (`mods/qsa-kv-fp8`,
four files). Its safety argument is a dynamic-range measurement — `|K|max 89.5`, `|V|max 31.4`
against e4m3's 448 — taken on the **FP8 build of this model, not on the NVFP4 checkpoint served
here**. On unified memory a saturated value returns wrong tokens; it does not fault. Before
relying on the default, run `overlay/mods/qsa-kv-fp8/probe/` on the served checkpoint
([docs/MODS.md](docs/MODS.md#modsqsa-kv-fp8--kv-cache-in-fp8-on-the-qsa-attention-path) says
how), or set `KV_DTYPE=bfloat16` and `VLLM_QSA_KV_FP8=0` in `recipe.env.local`. That fallback
is documented but **was not booted**; it carries no number.

## Performance

Hardware and method: [docs/HARDWARE.md](docs/HARDWARE.md). Every figure in the **this recipe**
column names its source file.

| | FP8 baseline | **this recipe** | source |
|---|---|---|---|
| weights resident, per rank | 86.98 GiB | **63.70 GiB** | engine log, `Model loading took` |
| KV pool | 1.49 M tokens | **3.55 M tokens** | engine log, `GPU KV cache size` |
| KV bytes | 10.91 GiB | 26 GiB (pinned) | `KV_BYTES` in `recipe.env` |
| mean accepted length (MTP) | 2.956 | **2.987** | `results/decode-mal-FINAL-post-fadvise-*.json` |
| decode, median of 4 content types | 50.69 tok/s | **51.14 tok/s** | same file, `debit_mediane_globale` |
| — prose / code / structured / repetitive | — | 45.9 / 52.7 / 49.6 / 68.6 | same file, per content |
| aggregate at 4 concurrent | — | **99.4 tok/s** | `results/concurrence-FINAL-post-fadvise-*.json` |
| 5 × 296 k-token requests | **OOM-killed** | **5/5 at 200, 699 s, 0 preemptions** | `results/baseline-nvfp4-5x300k.json` |
| degeneration gate | clean 6/6 | **clean 6/6** | `results/degen-q38-P4-nvfp4-full.txt` |
| vision probe | — | 42/42 images correct at 8,216 prompt tokens | `results/multi-image-*.json` |
| cold prefill, 8/16/32/64 k | — | **2 787.8 tok/s** median (2 804.9 / 2 809.8 / 2 770.7 / 2 676.2) | cold proven: `prefix_cache_queries +360 294`, `hits 0` |

**Where the decode time goes**, from a torch profile of 30 pure-decode steps on both ranks:
decode runs at **73 % of this box's measured 235 GB/s bandwidth wall**, and the dominant kernel
family — 57.2 % of GPU time, 529 dense BF16 GEMM launches per step — is already at **87 %** of
it. GPU occupancy is 94.3 % (union of kernel intervals, not a sum of durations). So there is no
cheap kernel lever left here: everything sits uniformly near the wall, and the bytes that remain
are the `lm_head`, read **four times per step** — once to verify, once per MTP draft step, over
the full 248,320-row vocabulary. `REFUSED.md` §8-11 records the four levers measured against
that profile and refused.

On the long-context row: `running_max = 3` — at most three requests were in flight; KV peaked at
33.8 % of the pool ≈ 1.20 M tokens resident (above the 889 k of three live requests because
freed blocks linger in the prefix cache). 1.48 M tokens is what the run ingested, not what was
resident.

The FP8 baseline column is the FP8 build of the same model: decode and mean accepted length
from `results/decode-mal-PROD-2026-09-05-*.json`; weights and KV pool from that build's engine
log (not in `results/`).

`tok/s` is not portable between models: tokenizer fertility differs (here 3.258 chars/token),
and client-side streaming accounting differs from engine-internal accounting. Compare wall
seconds on identical text.

NVIDIA's NVFP4 scales are MSE-calibrated on text only (`cnn_dailymail` + `Nemotron-Post-Training-v2`,
per the model card); activations from image tokens are outside that distribution. Multimodal
quality beyond the probe above is not measured here.

## Flags that decide

None of these is a tuning knob.

**1. `MOE_BACKEND=marlin`.** Marlin is the only W4A16 path the sm_121 backend oracle offers; the
five it ranks ahead (`FLASHINFER_TRTLLM`, `FLASHINFER_CUTEDSL`, …, `VLLM_CUTLASS`) are W4A4.
Same weight bytes either way. Marlin is also a canary: with an unquantized MTP MoE, vLLM refuses
to boot (`moe_backend='marlin' is not supported for unquantized MoE`) instead of serving with
dead speculation. Throughput of the A4 paths was not measured here. Verify it in the engine's
`non-default args` echo. Do not pass it through `EXTRA_ARGS`, which carries exactly one flag:
the value is word-split inside upstream's `docker run` line, a second word is eaten by docker
(`unknown flag: --moe-backend`, boot dead in 23 s).

**2. `VLLM_QWEN4EXP_PLE_FP8=1` + `mods/ple-fp8-mixed`.** The 47.68 GiB n-gram table. Without
both, `ple_layer.py` returns `None` under the mixed-precision config (the sibling config is a
`ModelOptFp8Config`) and the table is allocated bf16: 47.68 → 95.43 GiB, OOM on both ranks.

**3. `mods/mtp-modelopt-mixed`.** Without it, speculation is **silently dead**: the draft layer
index is local in the checkpoint (`mtp.layers.0`) while vLLM builds `mtp.layers.48`, and
`FP8_BLOCK_SCALES` is not in vLLM's modelopt dispatch, so routed experts fall through to
unquantized with no error. Symptom: 200s, correct answers, zero accepted drafts — mean accepted
length 1.000 instead of 2.99, about 19 tok/s instead of 50. Ported from
[Tech2Wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark) (Apache-2.0).
After enabling speculation, check the mean accepted length, not the throughput.

**4. `KV_BYTES` (26 GiB).** `--kv-cache-memory-bytes` skips memory profiling. Without it, at
`gmu 0.85` vLLM fills its budget with KV, `MemAvailable` falls to 1.6 GiB, and the kernel
OOM-kills `ray::RayWorkerProc` under load (observed 2026-09-06 03:30:38). Context costs about
7.7 GiB per million tokens at 5 streams (derived from the long-context run). Do not raise it
without re-measuring `MemAvailable` on both ranks.

## Gotchas

- **`prefix-match-unit=128` is only legal while speculation is on.** KV block size is 1664 with
  MTP (13 × 128) and 1600 without; 1600 is not divisible by 128 and the engine refuses after
  loading the weights. Read `block sizes=[…]` in the error.
- **`gpu-memory-utilization` no longer sizes the pool once `KV_BYTES` is set, but is not
  inert.** `up.sh` derives its pre-boot abort threshold from `gmu × total`; lowering it relaxes
  the only guard against an OOM on a swapless box. Keep 0.85.
- **`docker inspect --format '{{.Config.Cmd}}'` returns `sleep infinity`.** vLLM is started by
  `docker exec`. The only witness for a flag is the engine's `non-default args` echo.
- **A variable on upstream's allow-list is not a variable anything consumes.** Check both ends:
  the launcher's allow-list and the profile that reads it.
- **Upstream keeps its NCCL and sm_121 correctness flags in a gitignored `.env`.** A fresh clone
  has none of them; one of them prevents int4 MoE from returning garbage silently.
  `recipe.env` is the single source here and `up.sh` renders that file from it.
- **Run `up.sh` under a supervisor, never under a shell timeout.** Upstream's launcher runs in
  the foreground with `trap cleanup EXIT INT TERM HUP`: a shell deadline kills the process group
  and tears the cluster down after "Application startup complete". `nohup` only ignores SIGHUP.
- **`up.sh` step 1 stops and removes the container on both ranks** before anything else (a stale
  worker container keeps the weights resident). Use `./up.sh --dry-run` to test.

## Rejected — [REFUSED.md](REFUSED.md)

Seven levers measured, five refused. `max-num-batched-tokens=8192`: TTFT +97 %, aggregate −20 %,
and 32,408 tokens of image input pass at 4096. Disabling the kernel page compactor: +1.07 %
against 2.00 % drift. MTP `K=4`: +8.6 % single-stream, −7 % at c=4.
`gpu-memory-utilization=0.70`: identical pool.

## Patch scripts

Four scripts rewrite the installed vLLM at container start by exact-text substitution with an
expected occurrence count; `install.sh --check` dry-runs them inside the container.
Reference: [docs/MODS.md](docs/MODS.md). Known limitation: the inline comments of those four
scripts are in French; `docs/MODS.md` covers their contract in English.

## Credits

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (MIT) — the two-node
  cluster launcher (`launch-cluster.sh`) this recipe drives.
- [Tech2Wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark) (Apache-2.0) —
  the two MTP fixes ported in `mods/mtp-modelopt-mixed`.
- vLLM PR #53899 (engine branch), PR #47665 (read-time cast pattern), issue #54426 and
  PR #54846 (independent fp8-KV work on the same path; see `PROVENANCE.md`).
- [bilikaz/qwen38-flash-next-cluster-recipe](https://github.com/bilikaz/qwen38-flash-next-cluster-recipe)
  — two of the levers tested in `REFUSED.md`.
- NVIDIA (checkpoint, NVIDIA Open Model License) and Qwen (base model, Qwen Community License 1.0).

## License

Apache-2.0 (`LICENSE`). No model weights and no third-party source are redistributed.
Attribution and model licences: [NOTICE](NOTICE). Pins: [PROVENANCE.md](PROVENANCE.md).
