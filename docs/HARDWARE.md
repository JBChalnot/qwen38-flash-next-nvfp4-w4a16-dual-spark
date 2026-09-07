# The box, and how the numbers were taken

Every number in `README.md` and `REFUSED.md` came off this configuration. Elsewhere they are
a hypothesis.

## Hardware

| | |
|---|---|
| boxes | 2 × MSI EdgeXpert MS-C931 (NVIDIA DGX Spark) |
| GPU | GB10 Grace Blackwell, **sm_121** |
| memory | **121.69 GiB unified per box** — CPU, GPU, OS and page cache share one pool |
| swap | **none** |
| interconnect | ConnectX-7, direct cable, RoCE. Measured: RDMA carries the collectives |
| driver / CUDA | 580-open / CUDA 13 |
| OS | DGX OS 7.5.0, Ubuntu 24.04 aarch64, kernel 6.17 |
| disk | ~124 GiB for the checkpoint, **on each box** |

**Unified memory is the constraint that shapes everything else.** Weights, KV cache, CUDA
graphs, activation peaks, the page cache and your own shell all draw on the same 121.69 GiB.
There is no swap: when it runs out, the kernel OOM-kills a Ray worker.

### The head rank is structurally ~7 GiB heavier

Measured, on an idle serve, and it is not a leak:

| process | head | worker |
|---|---|---|
| `ray::RayWorkerProc` (holds the weights) | 6.03 GiB | 6.06 GiB |
| `vllm serve` (HTTP server, tokenizer, request handling) | **3.02** | — |
| `VLLM::EngineCore` (the scheduler) | **1.29** | — |
| Ray control plane (`gcs_server`, `ray --head`, agents) | **~0.60** | ~0.11 |
| an OpenAI-compatible proxy in front (litellm measured) | **0.84** | — |

⇒ **Size every memory decision on the head.** The worker's spare memory is unusable: the KV
pool is symmetric and bounded by the tighter rank.

An interactive session on the head (editor, shell tooling) measured **1.47 GiB**, on a box whose
margin is single-digit GiB. On unified memory the instrument competes with the engine.

## How the measurements were taken

**Instruments** (all in `tools/`, all reporting a range and not a single shot):
- `decode-mal.py` — throughput **and** mean accepted length from the **same run**, across four
  content types (prose / code / structured / repetitive). The spread between content types is
  large (45.9 to 68.6 tok/s here); a single-content figure on a speculative decoder is not
    meaningful.
- `concurrency.py` — aggregate is **total output ÷ wall time to completion**, never
  `output ÷ (wall − mean TTFT)`: with concurrency the latter leaves other requests' prefill
  inside the window and understates decode.
- `multi-image.py` — multimodal, scored per image, **with a negative arm that sends no image**.
- `logprobs-mae.py` — logprob MAE against the engine's own noise. See the caveat below.
- `pagecache-residency.py` / `free-checkpoint-cache.py` — `mincore`-based, validated in both
  directions.

**Protocol:**
- the first sequence after a boot is warm-up and is discarded (2.32 s on the second shot
  against 0.64 s once warm);
- median **and** range, never a single shot: at temperature 0.7 the real spread of a single
  task reaches 185 %;
- inter-boot noise floor here is **±4 %** on decode, so no arm is concluded from one pair of
  boots. Intra-boot it is ±2.0 %;
- verify every flag in the engine's **`non-default args`** echo. `docker inspect
  --format '{{.Config.Cmd}}'` returns `sleep infinity` with this launcher, because vLLM is
  started via `docker exec`;
- a truncated output has contaminated *form* metrics: a JSON cut at `max_tokens` repeats its
  delimiters and can look like degeneration on a healthy serve. Set ceilings above the need.

⚠️ **The server is not deterministic at `temperature=0`** (exact logprob ties plus numeric
drift). Validate a kernel by logprob MAE **against the engine's own noise**, never by output
identity. And that noise depends on length: it is exactly 0.000000 on 11–89-token prompts and
0.19–0.40 at ~30,000 tokens, where the server disagrees with itself on the top-1 token for
identical input. Two channels, never aggregated.

## Verified from a fresh clone

On 2026-09-07 the recipe was cloned from GitHub at the same path on both boxes, `install.sh`
run (image reused, at the pinned commit), the W4A16 variant rebuilt into a new directory, and
`up.sh` booted with `HOST=127.0.0.1`. Result: identical pool (3,547,511), weights 63.7 GiB,
all four patches applied, RDMA on, mean accepted length **2.966**, decode median **52.3 tok/s**
(44.1 / 54.7 / 49.9 / 67.6 by content) — within the ±4 % inter-boot floor of the published
figures. `results/decode-mal-fresh-clone-20260907.json`.

Of the three build stages in `install.sh`, stages 1 (arm64 build base) and 3 (launcher layer,
on the existing stage-2 image) were rebuilt from the published Dockerfiles on 2026-09-07 and
produce the expected result (`BASE_ARM64_OK`; `/workspace`, `ray 2.56.1`, empty entrypoint).
Stage 2 — vLLM from source at `f561eca6c` — is the build that produced the measured image on
2026-08-26 and was **not re-run** from the published tree: it takes hours and cannot run with
the engine up on the same 121 GiB.

## What is NOT measured here

- **The bf16-KV fallback.** Documented in `recipe.env`, not booted. No number.
- **Anything at a different context length than stated.** The KV fp8 cost (−12 %) is at 700 k
  and essentially vanishes short.
- **Output quality.** The automated gates (degeneration, accepted length, logprob MAE, vision
  probe) discard a broken build; they do not measure quality.
- **The dynamic range of K/V on the served checkpoint.** The headroom figure quoted for fp8 KV
  (`|K| 89.500`, `|V| 31.375` against 448) was measured on the **FP8** build on 2026-08-28,
  not on the NVFP4 W4A16 one. Run `overlay/mods/qsa-kv-fp8/probe/` on your own checkpoint.
