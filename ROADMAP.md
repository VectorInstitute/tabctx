# Roadmap

Nearly everything the original roadmap tracked has shipped and been
validated on real hardware -- see CHANGELOG.md v0.6.0-v0.9.0 for what
landed and the measured evidence. This file now holds only the
remainder (each item is also a GitHub issue labelled `roadmap`), and
gets deleted when it empties.

## Remaining

1. **Application catalog + persistent context store** ([#6](https://github.com/VectorInstitute/tabctx/issues/6)) (asked for by
   the first prospective users, 2026-09-01: "upload a cohort table per
   application, run TabICLv2 once, save the kv cache to disk, let end
   users just pick their application"). What exists and what's missing
   is spelled out in the README section "Serving many applications from
   pre-computed contexts"; the build order that falls out of it:
   1. A `ContextStore` protocol (persist / load / list / delete by id)
      with a local-directory implementation first and an object-storage
      one second; the existing `DiskSpillStore` becomes the eviction
      tier *on top of* it. Contexts gain a `pinned` flag (never evicted)
      and a `persisted` state; serialized files carry tabctx / backend /
      torch versions and device so a mismatch re-fits instead of
      loading garbage.
   2. `POST /v1/tabctx/datasets/{id}/persist`, `DELETE /v1/tabctx/datasets/{id}`
      (today nothing but LRU pressure removes a context), `GET /v1/tabctx/datasets`
      (per tenant: task, model, shape, feature names, residency), a
      warm-load manifest at replica start, and a batch `tabctx fit`
      entry point that writes to the store without a serving replica.
   3. Read-only tenant credentials at the gateway so "admin publishes,
      users predict" is enforceable; per-application aliases stay a
      gateway concern.
2. **Post-release: ephemeral GKE stress run + researcher deployment** ([#7](https://github.com/VectorInstitute/tabctx/issues/7))
   (deliberately sequenced after the release). Everything is staged in
   `VectorInstitute/inference-platform` branch `test/tabicl-gke-onboard`:
   the research overlay (`tabctx-research-values.yaml`: both models, one
   A100, spillover, 64Gi host RAM per the measured host-OOM boundary),
   probes (`probe_models.py`, `probe_stress_v09.py`, plus the existing
   battery), and a parameterized `onboard.sh`
   (`OVERLAY=... RELEASE=tabctx-research TABPFN_TOKEN=... KEEP_ALIVE=true`).
   The prod inference-platform deployment for Vector researchers follows.
3. **TabPFN memory calibration grid** ([#8](https://github.com/VectorInstitute/tabctx/issues/8)): the preloaded admission grid is
   TabICL data; TabPFN deployments fall back to the conservative formula
   plus runtime learning. Run `benchmarks/calibrate_memory.py`-style
   sweeps through `TabPFNBackend` on an A100 and add the grid.
4. **Docs site** ([#9](https://github.com/VectorInstitute/tabctx/issues/9)): README carries the full story today; a docs site
   matters as external adoption starts.

## Throughput and table size: multi-GPU and CUDA-level work

The LLM serving engines (vLLM, SGLang, TensorRT-LLM) got their gains first
from memory management and scheduling (paged KV, continuous batching,
prefix reuse) and then from CUDA kernels. tabctx already has the
scheduling layer (context cache, usage-aware admission, evict-ahead,
spill, same-context coalescing). This track adds multi-GPU replicas and
the kernel work. Everything is developed and tested on a GKE GPU node
first, then validated on prod's A40s (Ampere sm_86: no FlashAttention-3,
no FP8).

5. **Multi-GPU replica** ([#30](https://github.com/VectorInstitute/tabctx/issues/30), done -- PR #38/#40): one engine per device,
   fits placed where they fit without eviction on the least active
   device, predicts routed to the owning device. **One worker process per
   GPU** ([#39](https://github.com/VectorInstitute/tabctx/issues/39)): fits are host-bound (the driving thread is busy ~90% of
   wall time), so threads of one process contend for the GIL -- 4x A100,
   four concurrent fits: 2.1-3.1x aggregate from threads, 3.96-4.3x from
   processes. Under Ray Serve, 6 mixed users: 2.58 -> 5.0 predicts/s.
6. **Profiling** ([#31](https://github.com/VectorInstitute/tabctx/issues/31), first pass done): a 40k x 100 TabICL fit on an
   A100 is ~30% CPU preprocessing (per-feature Yeo-Johnson power
   transforms in scipy dominate) and ~70% GPU half, which is ~87% GPU-busy
   (flash attention 35%, layer norm 21% -- memory-bound over
   [4, 40, 40000, 128] activations -- copies 15%). Done from it: the CPU
   half is prepared before taking the device (overlaps another fit's GPU
   half: back-to-back fits 17-26% faster), and tabicl's per-call
   log(n) host-to-device syncs are cached (bit-identical; 10k x 50 fits
   16% faster), and the attention blocks are compiled (#36, below). Next
   from it: a GPU Yeo-Johnson fit for isolated-fit latency (~3 ms vs
   2.9 s at 40k x 100; not bit-identical to sklearn's lambdas, so behind
   an accuracy study). TabPFN-3.5's fit is the same shape: RMS norm 33% of
   its CUDA time, flash attention 26%, copies 16%.
7. **Per-device calibration grids** ([#37](https://github.com/VectorInstitute/tabctx/issues/37), TabPFN done): TabPFN-3.5 has its
   own requirement grid (A100, 20 shapes): it, too, sizes working memory
   from what is free, so admission reaches 100k rows at 100 features on an
   A40 (was 5k on an L4 under TabICL's formula). Still to do: L4/A40-
   measured grids (A100 grids scale by capacity today).
8. **Head-of-line scheduling** ([#39](https://github.com/VectorInstitute/tabctx/issues/39), done): light predicts (resident
   context, <= 2048 rows) run beside a fit on the same GPU instead of
   waiting for it -- ~0.1 s p50 during a 40k x 100 fit instead of its
   full 11 s, 0 wrong answers (thread-local backbone cache); in-use
   contexts are pinned against evict-ahead; restores and large predicts
   wait. Small-fit p95 under load 10.6 s -> 1.4 s.
9. **Tiled/fused column+row stage kernels** ([#32](https://github.com/VectorInstitute/tabctx/issues/32)): cut the fit peak and the
   memory-bound layer norms, raising the per-GPU table ceiling.
10. **Paged context KV + cross-context varlen predict** ([#33](https://github.com/VectorInstitute/tabctx/issues/33)): throughput
    with many users (PagedAttention + continuous batching analogue).
11. **INT8 context KV** ([#34](https://github.com/VectorInstitute/tabctx/issues/34), done for TabICL): ~0.52x resident, identical
    labels on the evaluation sets; TabPFN already defaults to int8.
12. **Context parallelism across GPUs** ([#35](https://github.com/VectorInstitute/tabctx/issues/35)): tables larger than one GPU
    (ring attention over the replica's devices). Disk offload already
    lets one A100 fit 130k x 500 (65M cells) in 26 min.
13. **torch.compile** ([#36](https://github.com/VectorInstitute/tabctx/issues/36), done for TabICL): each attention block's
    forward compiled, so Inductor fuses the layer norms, residual adds and
    GELU MLP around the attention kernel. A100: GPU half 1.7-2.2x faster
    (40k x 100: 7.65 -> 4.30 s), peak memory ~20% lower (33.1 -> 27.0
    GiB, so larger tables fit), accuracy unchanged on the evaluation sets;
    Ray Serve, 6 users on 4x A100: 5.78 -> 6.99 predicts/s. Dynamic
    shapes needed tabicl's ssmax log(n) computed symbolically (math.log on
    the row count recompiled every new table). ~50 graphs (fp16 path
    only) compile in warmup (~130 s per worker). CUDA graphs aren't worth it: the GPU half
    is GPU-bound, not launch-bound. TabPFN-3.5's own compile path
    (`enable_torch_compile`) measured 1.3-1.5x faster fits, ~30% faster
    predicts and ~22% lower peak, but it still recompiles for many new
    table sizes (9-82 s each, even with its chunked-SDPA loop kept out of
    the graph), so it stays off until those are traced down.
14. **Serving-framework-agnostic core** ([#39](https://github.com/VectorInstitute/tabctx/issues/39)): the router, per-GPU workers
    and scheduling are plain Python with no Ray dependency; a Dynamo
    worker hosts the same core. Next: residency events and pluggable
    placement so a platform scheduler can co-optimize LLMs and tabular
    models.

PyTorch-level first: tiling and chunking, `torch.compile` (generated Triton
kernels, CUDA graphs via `reduce-overhead`), FlexAttention for paged and
cross-context attention, SDPA, and `torch.distributed` context
parallelism. Hand-written Triton/CUDA only where a profile (#31) shows a
gap those can't close -- the models change too often (TabPFN v3 -> v3.5,
TabICL v1 -> v2 within months) for version-locked kernels to be the
default.

## Standing process note

Every significant fix in this project came from testing against real
systems -- most recently the shared-backbone kv-cache pin (a ~15GB
eviction leak) and the fragmentation OOM at 90% budget, both of which
only reproduce on a real GPU. Keep reading model internals AND deploying
before believing anything.
