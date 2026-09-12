# Rejected

Seven levers, measured on this exact stack on 2026-09-07. Five refused, one no-op, one
adopted. Raw JSON in [`results/`](results/).

**Where the reference column comes from.** Every "4096 / K=3" reference figure below is the
mean of two same-boot repetitions of the retained configuration,
`results/concurrence-CMP-c4-20-20260907-124947.json` and `-125357.json` (aggregate
35.0/59.7/85.6/100.4 and 36.9/60.2/83.0/97.2 for c=1..4), and for single-stream decode the
three windows of `results/decode-mal-CMP-*.json` (52.06 / 52.09 / 51.02 tok/s; mean accepted
length 3.005 / 2.974 / 2.949, mean **2.98**). The noise floor is `derive_pct` in
`results/compaction-ab.json`.

**The noise floor was established first, before testing anything.** Two repetitions of the
reference configuration, same boot:

| | measured drift |
|---|---|
| decode, single stream | ±2.0 % |
| aggregate at 1 concurrent | **±5.2 %** |
| aggregate at 2 / 3 / 4 | ±0.9 / ±3.1 / ±3.2 % |

Nothing below that floor is called an effect.

---

## 1. Disable the kernel page compactor — **no gain**

`vm.compaction_proactiveness=0`. From a
[neighbouring kit](https://github.com/bilikaz/qwen38-flash-next-cluster-recipe): on unified
memory the GPU's pages *are* host pages, so every page the kernel migrates for defragmentation
must first be unmapped from the GPU. That kit measures 4–5 s of stall every ~37 s and reports
~10 %.

The sysctl takes effect immediately, with no restart — the only lever here that can be
A/B/A'd **inside one boot**, which is why the drift is ±2.0 % instead of the ±4 % you pay
between boots.

| window | decode | pages scanned by the compactor, **over the window** |
|---|---|---|
| A1, compactor on | 52.06 | 111,211,008 in 269 s |
| B, **disabled** | **52.09** | **0** — the lever demonstrably applied |
| A2, on again | 51.02 | 108,533,373 in 273 s |

Effect **+1.07 %** against a drift of **2.00 %**. Under concurrency: +3.3 / +2.2 / **−2.0** /
+0.2 % at c=1..4 — the sign flips.

**Why it does nothing here.** From
`results/compaction-ab.json`, window A1: the compactor scans **24.8 M pages/min** —
**1.70 GB/s** — and moves essentially nothing. `compact_success = 0`, `compact_isolated`
313 MiB/min, and `pgmigrate_success` only **10 MiB/min** actually migrated. The stall
requires real migrations, because migration is what unmaps GPU pages. On this memory profile
there are none.


⇒ Zero gain, zero risk. If your memory profile differs, the number to watch is
`pgmigrate_success` in `/proc/vmstat`, **not** the scan counters.

## 2. RDMA — **nothing to do, but verify it**

Named by the same kit: **NCCL will run over TCP sockets on the same ConnectX cable and never
say so.** The environment looks right, the serve works, and
every step is roughly twice as slow.

Measured here during a 700-token generation: **RDMA +1751.8 MB vs TCP +0.2 MB**. The container
runs `privileged` with `IpcMode=host`, so it gets `/dev/infiniband` and the memory-lock
capability without the three flags that kit passes explicitly.

`ulimit -l` inside the container is **8 MiB**, not unlimited; RDMA works anyway. Relevant if a
future NCCL registers larger buffers.

`./view.sh` prints both counters. Run it once.

## 3. `max-num-batched-tokens` 4096 → 8192 — **refused**

| | 4096 | 8192 |
|---|---|---|
| KV pool | 3,547,511 | **3,547,511 — identical** |
| decode, median | 51.7 | 52.9 (inside noise) |
| aggregate c=2 | 59.9 | **54.0** (−9.9 %) |
| aggregate c=3 | 84.3 | **67.7** (−19.7 %) |
| aggregate c=4 | 98.8 | **86.3** (−12.7 %) |
| TTFT c=2 / c=3 / c=4 | 2.17 / 2.27 / 2.72 s | **3.58 / 4.47 / 4.20 s** |

One new fact: an older refusal of 8192 rested on "−20 % of KV pool". With
`--kv-cache-memory-bytes` set, profiling is skipped and **the pool is identical** — that cost
is gone. The refusal stands on TTFT (**+97 %**) and aggregate (−10 to −20 %), four to six
times the drift.

## 4. Does 4096 cap the image encoder? — **the claim is false**

The reason usually given for 8192 is that `max-num-batched-tokens` is also the image-input
encoder budget. Tested with batches of images, scoring the shape→colour mapping **per image**,
with a negative arm that sends no image:

| chunk | input | prompt tokens | result |
|---|---|---|---|
| 4096 | 14 images @768 px | **8,216** (2.0× the budget) | **42/42 correct** |
| 4096 | 3 images @1536 px | **6,992** (1.7×) | **9/9** |
| 8192 | 14 images @1536 px | **32,408** (4.0×) | **42/42** |
| — | *negative arm, no image* | — | 0–1/3 → the probe discriminates |

Chunked prefill absorbs multimodal input far beyond the budget without damage. **4096 does not
cap the image encoder.**

Reproduce: `python3 tools/multi-image.py --images 3,14 --taille 1536`.

## 5. MTP draft width `K=3` → `K=4` — **a trade, not a win**

| | K=3 | K=4 |
|---|---|---|
| mean accepted length | 2.98 | **3.31** (+11.1 %) |
| decode, repetitive content | 67.3 | **76.0** (+13.0 %) |
| decode, prose | 45.1 | **43.2** (−4.3 %) |
| aggregate c=1 | 36.0 | **39.1** (+8.6 %) |
| TTFT c=1 | 1.71 s | **0.96 s** (−44 %) |
| aggregate c=3 / c=4 | 84.3 / 98.8 | **79.1 / 91.6** (−6.2 / −7.3 %) |

K=4 buys single-stream and pays for it under concurrency. Mean accepted length rises 11 %
(2.98 → 3.31) while the per-draft acceptance rate falls (66.2 % → 57.7 %); throughput does
not follow.

⚠️ It also costs a flag: `--prefix-match-unit=128` has to be removed, because the KV block
size depends on the draft width and 128 stops being legal. Adopting K=4 means finding a new
legal unit.

**Kept K=3** for concurrent workloads. For sequential ones the −44 % TTFT may be worth it.

## 6. Lower `gpu-memory-utilization` to 0.70 — **no-op**

With `--kv-cache-memory-bytes` set, the engine says it itself:

```
reserved 26.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes
config and skipped memory profiling
```

Identical pool (3,547,511), identical initial free memory (108.2/112.5 vs 108.1/112.6). With
the pin set, `gmu` no longer sizes anything **in the engine** — but `up.sh` still derives its
pre-boot abort threshold from it, so lowering it relaxes the only memory guard (see README).

**Keep 0.85**: if the pin is ever removed, 0.85 yields the larger pool rather than a surprise.

## 7. Return the checkpoint's page cache after boot — **adopted**

The neighbouring kit patches the safetensors loader to drop each shard from the page cache as
it is consumed. Not ported: the default safetensors path is **mmap**, so the tensors *are* the
page cache. Dropping early is either a
no-op (the kernel will not evict a mapped page) or a silent re-read — in the one file where a
mistake never faults on unified memory. *After* the boot the question disappears: no
checkpoint page is useful any more.

Measured with `mincore` (`tools/pagecache-residency.py`, validated both ways: 0.00 GiB after a
cache drop, 8.00 GiB after reading 8 GiB):

| | head | worker |
|---|---|---|
| checkpoint in page cache, before boot | 0 | 0 |
| **after boot** | **10.29 GiB** | **16.71 GiB** |
| **MemFree after releasing it** | **1.13 → 11.54** | **1.77 → 18.59** |
| MemAvailable | +0.06 | +0.06 |

`MemAvailable` does not move — the cache already counted there. **`MemFree` goes up tenfold**,
and `MemFree` is what the CUDA driver allocates against. The failure this prevents looked like:
`memory allocation failed ... trying to allocate 186,608,640 bytes (free: 739,536,896)`.

No regression: 51.14 tok/s, acceptance 2.987, aggregate 36.5 / 58.7 / 85.0 / 99.4 — all inside
noise. Run `tools/free-checkpoint-cache.py` on **each** rank after boot (`up.sh` runs it *before*
boot, which is a different thing).

## 8. Enable the vendor's skinny GEMM (`low_latency_gemm.py`) — **refused, +3.8 %**

The model ships a CuTe-DSL skinny GEMM with *measured* configs for m=1,2,4,8 and PDL, and
`low_latency_gemm.py:151-155` switches it off on anything that is not sm_103:

```python
if dtype != torch.bfloat16 or not _is_sm103():
    return            # sm_121 leaves here; every dense projection falls back to F.linear
```

Nothing in the kernel forbids it (`cute_dsl/skinny_gemm.py:36-48` only tests that `cutlass.cute`
imports — it does, CuTe DSL 4.6.2 — and `_use_pdl()` is true from major ≥ 9). It looks like a free
win. It is not:

| regime | cuBLAS | skinny | gain |
|---|---|---|---|
| m=1 | 35.67 ms | 26.49 ms | **1.35×** |
| m=2 | 28.51 | 26.32 | 1.08× |
| **m=4 (what runs here)** | **28.29** | **26.44** | **1.07×** |
| m=8 | 28.40 | 26.38 | 1.08× |

Weighted by the real per-step call counts, and applied to the family that carries 57.2 % of decode
GPU time (32.86 ms/step), 1.07× is **−2.15 ms ⇒ +3.8 % end to end** — inside the ±4 % inter-boot
noise here. The gain lives at **m=1** and this deployment runs at m≥2: the draft LM-head call
measures 2 656.7 µs in production, which is the cuBLAS m≥2 time (2 687–2 709), not m=1 (3 661).

Two of ten shapes — **161 calls/step**, the K=320 hyper-connection and shared-down projections —
are *refused* outright (`K must be divisible by block_size * vector_width`), so the mod could not
cover the family even if it paid.

## 9. Restore `cooperative_topk` on sm_121 — **nothing to gain**

`nvidia/ops/qsa.py:788-792` excludes the fast path by name for this GPU family, with no comment
and no ticket:

```python
use_cooperative_topk = (... and not current_platform.is_device_capability_family(120))
```

Both ops are compiled and return the same top-k, so it reads like a bug worth fixing. A kernel
profile says otherwise: `persistent_topk` costs **1.059 ms out of 1 724 ms of decode GPU time —
0.06 %**. The whole QSA family (sparse+MQA+indexer+merge+topk+expand) is **1.1 %**; doubling all
of it would return +0.55 %.

## 10. Port the upstream fused GDN decode kernel — **it is already running**

Two independent reviews reported the fused `fused_gdn_decode_post_conv_mtp` path as missing from
this engine. It is not. In the live container:

```
GDN decode kernel: cuda                     # engine echo, both ranks
hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp") -> True
```

Every guard passes: v/k head ratio 48/16 = 3 ∈ (1,2,3,4,8), `MAX_FUSED_GDN_MTP_TOKENS = 8 > 4`,
BF16 recurrent state, `has_device_capability(80)`. It costs **0.69 ms/step (1.2 % of decode GPU
time)**. ⚠️ The fallback is **silent** — `logger.info_once("Falling back to the Triton GDN decode
path: …")`, buried in a 12-minute boot — so the check is to grep `GDN decode kernel:` in the
engine echo, not to trust a source reading.

## 11. MTP draft width `K=3` → `K=2` — **the ridge is flat**

The other side of §5. Same config, one variable plus its forced dependencies:

| | K=3 | K=2 |
|---|---|---|
| decode, median of 4 contents | 51.14 | **51.10** |
| mean accepted length | 2.987 | **2.490** |
| engine step | 58.41 ms | **48.73 ms** |

**+0.08 %.** One draft step costs **+9.68 ms of step** and returns **+0.497 of accepted length** —
16.6 % against 16.6 %, to the decimal. Bytes and acceptance trade 1:1, which is why K=2, K=3 and
K=4 all land within noise. Only a change that cuts draft bytes *at constant acceptance* can win.

⚠️ Two forced dependencies, not free variables. `cudagraph_capture_sizes` must be multiples of
`K+1`, and the KV block size moves with the draft width: **1664 at K=3, 1648 at K=2**, so
`--prefix-match-unit=128` becomes illegal (the largest power-of-two divisor of 1648 is **16**).
The first attempt died after 12 minutes on `ValueError: Invalid prefix_match_unit=128 … block
sizes=[1648, 8, 1648, …]`. Read the block size out of the error, do not assume it.
