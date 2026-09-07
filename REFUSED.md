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

