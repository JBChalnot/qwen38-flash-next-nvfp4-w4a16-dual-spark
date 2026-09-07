# The four patch scripts

Each one rewrites an installed vLLM in place, at container start, by exact-text substitution
with an expected occurrence count. If a count differs, the script exits non-zero and patches
nothing — it cannot half-apply. `./install.sh --check` dry-runs all four **inside the
container**, against a throwaway copy of the built tree, so a drift is caught at install time.

**What that dry-run does not prove.** Run against a *different* engine
(vLLM 0.26.1), `ple-fp8-mixed` and `qsa-kv-fp8` refuse — their targets are model-specific files
that do not exist there — but **`mtp-modelopt-mixed` and `opt-sm121-gemv` apply cleanly**. So
"all four applied" is NOT evidence that you are on the right engine. The thing that is:
`install.sh --check` compares `vllm.__version__` to the pinned commit and stops **before** the
anchors. See [`../PROVENANCE.md`](../PROVENANCE.md).

> ⚠️ **The inline comments inside these scripts are in French.** This page is the English
> reference and is written so you never need to read them: it carries the contract, the
> mechanism, the failure mode and the check for each. The scripts' *behaviour* is what the
> dry-run verifies, not their prose.

Files each one touches: see [`../PROVENANCE.md`](../PROVENANCE.md).

---

## `mods/ple-fp8-mixed` — keep the n-gram table in FP8

**Without it, the serve OOM-kills both ranks at load.**

The checkpoint's PLE n-gram table is 47.68 GiB of FP8. NVIDIA declares it as FP8 in
`quantization_config.quantized_layers`. But `ple_layer.py` selects its FP8 embedding method
behind `isinstance(quant_config, Fp8Config)` — and under the mixed-precision path the sibling
config is a `ModelOptFp8Config`, a **different class**. The check returns `None`, vLLM falls
back to an unquantized embedding, and the table is allocated in bf16: **47.68 → 95.43 GiB**.

The patch adds a branch, gated on `VLLM_QWEN4EXP_PLE_FP8=1`, that returns the FP8 method when
the mixed config resolves the n-gram prefix to `FP8`. The method itself is self-contained: it
only needs the weight parameter and a per-tensor scale.

**Its guard checks the RESULT, not the bookkeeping.** After loading, the global scale must be
finite and strictly positive. Membership in the loader's `loaded` set is not a usable
observable: vLLM disables weight tracking for quantized models. The value is.
The predicate is exercised on five cases: healthy, the `finfo(float32).min` sentinel, zero,
NaN, infinity.

**Check:** the serve boots and `Model loading took` is ~63.7 GiB per rank, not ~87.

---

## `mods/mtp-modelopt-mixed` — make the speculative drafter load

🔴 **Without it, speculation is silently dead: the serve boots, returns 200, answers
correctly, and accepts zero drafts.** Mean accepted length 1.000 instead of ~2.99, roughly
19 tok/s where you expect 50, and nothing in the logs.

Two independent holes:

1. **The draft layer index is local to the checkpoint.** modelopt keys the entry as
   `mtp.layers.0.mlp.experts`; vLLM builds the drafter's modules at
   `mtp.layers.<num_hidden_layers>`, i.e. `mtp.layers.48`. The prefix never matches, so the
   entry is never found. The patch offers the draft-local index as an extra candidate.
2. **`FP8_BLOCK_SCALES` is not in vLLM's modelopt dispatch** — zero occurrences in the whole
   tree. And nothing raises: an unknown algorithm falls to `UnquantizedLinearMethod()` for a
   `LinearBase` and to `return None` for a `RoutedExperts`, so the drafter's experts are
   created **bf16 unquantized** while the checkpoint holds them as 128×128 block-scaled FP8.
   The patch routes them to vLLM's generic block-FP8 MoE method.

Ported from [tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark)
(Apache-2.0). Two deliberate changes: it is a patch script rather than a redistributed vLLM
file, and the index parsing uses plain string operations because the target file does not
import `re` and injecting an import would add a failure mode. An AST scope check refuses the
patch if any injected name is not assigned in its host function.

**Check — and this one is not optional:** read the **mean accepted length**, never the
throughput. `python3 tools/decode-mal.py` reports both from the same run. Low throughput reads
like a slow kernel when the cause is zero acceptance.

⚠️ `MOE_BACKEND=marlin` turns this failure from silent into loud: with it, vLLM refuses to
boot on an unquantized MoE (`moe_backend='marlin' is not supported for unquantized MoE`).
That is the second reason the flag is not optional.

---

## `mods/opt-sm121-gemv` — a Triton GEMV for the skinny decode matmuls

**Pure performance. Removing it costs throughput and nothing else.**

At batch 1 the decode path issues very tall, very thin matmuls. cuBLAS is not good at those on
this part: measured on `lm_head [248320, 2560]`, cuBLAS **172.8 GB/s** against Triton
**240.2** — ×1.39. The mod routes such shapes to a Triton kernel.

**The predicate has a launch floor of 6 MiB.** Below it Triton's launch overhead exceeds the
gain. The floor was measured with a micro-benchmark that rotates over a working set larger
than L2 (24 MiB): a benchmark that reuses one weight across iterations makes small weights
L2-resident from the second shot, which never happens in production where 1.18 GiB of weights
stream between reuses, and inverts the verdict for several shapes.

**Check.** Kernel-level: the rotation micro-benchmark, five shapes kept with gains of 23 to
38 %. The predicted end-to-end effect with the 6 MiB floor is **+1.78 %**, under the ±4 %
inter-boot noise floor, so it is not validated end to end. (An earlier 24 MiB predicate measured
+6.9 % end to end on two boots; that figure does not apply to this version.)

---

## `mods/qsa-kv-fp8` — KV cache in FP8 on the QSA attention path

**Read this whole section before enabling it.**

It buys **×1.84 the KV seats per byte** — this model's per-token K/V cost drops
14,144 → 7,696 bytes — so the same 26 GiB pin holds **3,547,511** tokens measured, against
~1.97 M derived for bf16.
It costs about **−12 % of decode throughput at 700 k of context** (essentially nothing at short
context), and it rewrites **four** files, one of them the attention backend, where the other
scripts rewrite one.

**What it has to defeat.** Two guards, and neither mentions sm_121:
- `flash_attn.py` raises `NotImplementedError: FlashAttention does not support fp8_e4m3
  kv-cache on this device` whenever a support predicate returns false — and that predicate
  only admits GPU families 90 and 100, while this part is family 120. The guard fires **even
  though the QSA path does not use FlashAttention to compute**: on this path the read is
  entirely Triton and FA only supplies metadata.
- vLLM allocates an fp8 cache as **`uint8`**. The generic FA path installs the
  `uint8 → float8_e4m3fn` view itself; a custom attention path does **not**. Without that
  view, `tl.load` reads integers and the arithmetic is **silently wrong**.

**What guards you, and what does not.**
- The exact-count anchors guard the **engine version**. An unexpected vLLM revision makes the
  script refuse.
- 🔴 **Nothing guards the CHECKPOINT.** The safety argument is that the attention K/V dynamic
  range stays far below e4m3's saturation point at 448. The measured figures — `|K|max
  89.500`, `|V|max 31.375`, i.e. 5.0× and 14.3× of headroom — were taken on the **FP8** build
  on 2026-08-28, **not** on the NVFP4 W4A16 checkpoint this recipe serves. A checkpoint with a
  wider range would saturate and return wrong tokens while every anchor matches cleanly.
  ⇒ **Run `mods/qsa-kv-fp8/probe/` on your own checkpoint and read the absmax it prints
  against 448.** The probe changes no arithmetic; it only instruments. Concretely —
  a boot in **bf16** with the probe and without the fp8 mod, then a few long requests:

  ```
  # recipe.env is sourced by up.sh and would overwrite variables set on the command line,
  # so put the override in recipe.env.local, which is sourced AFTER it:
  cat >> recipe.env.local <<'EOF'
  KV_DTYPE=bfloat16
  VLLM_QSA_KV_FP8=0
  MODS="mods/ple-fp8-mixed mods/mtp-modelopt-mixed mods/opt-sm121-gemv mods/qsa-kv-fp8/probe"
  EOF
  ./up.sh
  # ... send prompts covering your real content, long ones included, then:
  docker logs vllm_node 2>&1 | grep QSA-KV-RANGE | tail -20
  # remove those three lines from recipe.env.local before booting the real configuration
  ```

  `MODS` overrides `up.sh`'s default list (it is otherwise built from `KV_DTYPE`). Every
  200th write the probe logs the per-layer max of |K| and |V| and the global max (the interval
  is `QSA_KV_RANGE_EVERY` inside the container; `up.sh` does not forward it, so use the default). If the global max stays well under 448 — 89.5 / 31.4 on the FP8 build — the
  fp8 default is safe for that checkpoint; if it approaches it, set
  `KV_DTYPE=bfloat16` and stop there. ⚠️ `install.sh --check` dry-runs `overlay/mods/*/`
  only, so the probe is **not** covered by the anchor check. Its own guard is weaker than the
  others' — a presence check on `def do_kv_cache_update`, not an exact count — so on a
  drifted engine it could patch the wrong site. It only instruments, so the failure mode is
  a wrong or missing log line, never a wrong token.
- The precision change is measurable: **+0.2265 nat** of logprob MAE on the zero-noise
  short-prompt channel. ⚠️ `tools/logprobs-mae.py` uses a 0.05 nat threshold there and will
  report this as a structural deviation — **correctly**, because fp8 KV changes the arithmetic
  on purpose. Use the tool to see how much your configuration moves, not to be told it is fine.

**Also worth knowing:** e4m3 has 3 mantissa bits, so its relative error is ~6.25 % at worst
**independently of scale**. Adding a per-tensor scale buys precision for a *fixed-point* format
(int8 + scale steps down with the scale), and buys **nothing** for a floating one. Do not
expect a scale to fix this.

**Turn it off like this** — the recipe still works, with ~1.97 M tokens instead of 3.55 M:

```
KV_DTYPE=bfloat16
VLLM_QSA_KV_FP8=0
```

`up.sh` adds the mod only when `KV_DTYPE=fp8_e4m3`, and refuses if the two disagree — the flag
alone patches nothing, and fp8 without the mod is refused by the engine's own guard.
The bf16 fallback was not booted; it carries no number.
