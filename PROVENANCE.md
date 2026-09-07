# Provenance and pins

Everything that can move upstream, and what this recipe pins it to. **A patch script in
this repository is written against exactly one vLLM revision.** If you build a different
engine, the scripts refuse to apply — they never patch halfway — but you also lose the
measurements: they were taken on this engine and nothing else.

## The engine

The image is built in three stages by `install.sh`. None of them is `eugr/spark-vllm-docker`'s
own build path, whose `build-and-copy.sh` can download prebuilt wheels from the moving tags
`prebuilt-vllm-current` / `prebuilt-flashinfer-current`.

| stage | what | pinned to |
|---|---|---|
| 1 | `build/base-arm64/Dockerfile` — manylinux-like arm64 build base | `nvidia/cuda:13.0.3-devel-rockylinux9` |
| 2 | **vLLM's own `docker/Dockerfile`**, target `vllm-openai`, `BUILD_BASE_IMAGE` = stage 1, `torch_cuda_arch_list=12.0`, `CUDA_VERSION=13.0.3`, `PYTHON_VERSION=3.12` | **`f561eca6ca4f3f79808a696b1521cb76dc8aafa2`** |
| 3 | `build/workspace/Dockerfile` — `/workspace`, `ray[default]==2.56.1`, `ENTRYPOINT []` | on stage 2 |
| launcher | `eugr/spark-vllm-docker`, for `launch-cluster.sh` only | `30f72c8a36e3720d3c8785fa36884eab8196d45c` |

Stage 1 exists because vLLM's Dockerfile expects `pytorch/manylinux2_28-builder:cuda13.0`, which
has no arm64 manifest. `torch_cuda_arch_list=12.0` covers sm_121: since CUDA 12.9, `12.0` is a
family target. FlashInfer is pinned inside vLLM's Dockerfile at that commit (0.6.17 in the image).

Versions read from the measured image on 2026-09-07 (`importlib.metadata`; `install.sh --check`
asserts the vllm commit and warns if torch, ray or flashinfer-python differ). The pin fixes the
vLLM source, not the resolved Python dependencies: transformers and triton come from version
ranges and may resolve differently on a later build.

```
vllm               0.1.dev1+gf561eca6c
torch              2.13.0+cu130
triton             3.7.1
flashinfer-python  0.6.17
ray                2.56.1
transformers       5.16.1
python             3.12.3
cuda               13.0
```

`f561eca6c` is a commit of an open pull request against `vllm-project/vllm` (**#53899**,
PLE offload). It is not on any branch head, so `install.sh` fetches it by sha
(`git fetch --depth 1 origin <sha>`), which GitHub allows.

> **Pin the commit, not the pull request.** `refs/pull/53899/head` had moved to
> `357e054428cdb7dd697fa75024c548aced0efee8` by 2026-09-07. The PR head is a different engine.

> **`vllm-project/vllm` `main` does not work for this recipe.** The PLE-offload path this
> checkpoint needs is not on `main` (`grep VLLM_PLE_CPU_OFFLOAD` returns nothing there).

## What the launcher tree leaves floating

Only `launch-cluster.sh` is used from `eugr/spark-vllm-docker`. Its Dockerfile is not built.
`install.sh` records `eugr`, `vllm` and the base image tag in `.pins` at install time.

## The checkpoint

| what | pinned to |
|---|---|
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | revision `fab0aecb760cec45227f6656abcaafa11abca87a` |
| size verified | 25 files, **132 734 506 208 bytes** |

`tools/make-w4a16-variant.py` builds the served directory from it: 24 **relative**
symlinks plus one rewritten `config.json`. Relative, so the same directory resolves both
on the host and inside the container, where the tree is mounted under another prefix.

## Prior art

- `mods/qsa-kv-fp8` has two **later, independent** public counterparts: vLLM issue
  **#54426** (Nanetnounou, 2026-08-30, on a GB10) and PR **#54846** (andreasgru, 2026-09-01).
  This one dates from 2026-08-28; both were read on 2026-09-04 as corroboration (×1.79 and
  ×1.889 against ×1.84 here). Independent, not prior; see `NOTICE`. If #54846 merges, drop the mod.
- `mods/mtp-modelopt-mixed` is a **port** of two fixes by Tony DeAngelo (Tech2Wild), published
  2026-09-05, Apache-2.0; see `NOTICE` for the exact source lines.
- `mods/qsa-kv-fp8` patterns its read-time cast on **vLLM PR #47665**, as the script says inline.

## What the patch scripts anchor on

Each script under `overlay/mods/` locates its insertion point by
**exact source text** in the installed vLLM, counts the occurrences, and exits non-zero
unless the count is exactly what it expects.

No total anchor count is published: two ways of counting them disagreed, so no single number
would be reproducible. What is verifiable is the file list below and the behaviour:
`./install.sh --check` runs every script inside the container against a throwaway copy and
prints each one's own site counts (`2/2`, `3/3`, `refus neutralises : 6`, …), exiting
non-zero if any count is not what the script expects.

- **The anchors are a version guard for only two of the four scripts.** Run against a
  different engine (vLLM 0.26.1) inside its own container:

  | script | on vLLM 0.26.1 | why |
  |---|---|---|
  | `ple-fp8-mixed` | **refuses** | its target `models/qwen4_exp/nvidia/ple_layer.py` does not exist |
  | `qsa-kv-fp8` | **refuses** | same — its targets are model-specific files |
  | `mtp-modelopt-mixed` | ⚠️ **applies cleanly** | `quantization/modelopt.py` exists in both |
  | `opt-sm121-gemv` | ⚠️ **applies cleanly** | `model_executor/layers/utils.py` exists in both |

  ⇒ **The anchors are NOT a version guard for half the recipe.** What actually guards you is
  `./install.sh --check`, which verifies `vllm.__version__` against the pinned commit and
  **exits before it reaches the anchors**. Do not skip it, and do not read "all four applied"
  as "this is the right engine".

Files each script rewrites in the installed vLLM — extracted from the scripts themselves:

|script|files|
|---|---|
|`overlay/mods/opt-sm121-gemv`|`model_executor/layers/utils.py`, plus it **adds** `model_executor/layers/gemv_sm121.py`|
|`overlay/mods/ple-fp8-mixed`|`models/qwen4_exp/nvidia/ple_layer.py`|
|`overlay/mods/mtp-modelopt-mixed`|`model_executor/layers/quantization/modelopt.py`|
|`overlay/mods/qsa-kv-fp8`|`models/qwen4_exp/nvidia/ops/qsa.py`, `models/qwen4_exp/nvidia/qsa.py`, `models/qwen4_exp/common/qsa_cache.py`, `v1/attention/backends/flash_attn.py`|
|`overlay/mods/qsa-kv-fp8/probe/`|`v1/attention/backends/flash_attn.py` — instrumentation only, changes no arithmetic|

