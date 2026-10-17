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

5. **Multi-GPU replica** ([#30](https://github.com/VectorInstitute/tabctx/issues/30), in progress): one engine per device,
   fits placed where they fit without eviction on the least active
   device, predicts routed to the owning device.
6. **Profiling harness** ([#31](https://github.com/VectorInstitute/tabctx/issues/31)): per-stage time and peak-memory breakdown;
   picks and sizes everything below. Hypothesis to confirm: the fit
   peak is per-cell activations in the column/row stages, not attention.
7. **Per-device calibration grids** ([#37](https://github.com/VectorInstitute/tabctx/issues/37)): admission on non-A100 cards
   falls back to the conservative formula and is the first ceiling a
   large-table user hits.
8. **Tiled/fused column+row stage kernels** ([#32](https://github.com/VectorInstitute/tabctx/issues/32)): cut the fit peak, raising
   the per-GPU table ceiling (FlashAttention-style tiling; chunked
   prefill for the context KV).
9. **Paged context KV + cross-context varlen predict** ([#33](https://github.com/VectorInstitute/tabctx/issues/33)): throughput with
   many users (PagedAttention + continuous batching analogue).
10. **INT8 context KV** ([#34](https://github.com/VectorInstitute/tabctx/issues/34)): about 2x contexts per GPU.
11. **Context parallelism across GPUs** ([#35](https://github.com/VectorInstitute/tabctx/issues/35)): tables larger than one GPU
    (ring attention over the replica's devices).
12. **CUDA graphs for small predicts** ([#36](https://github.com/VectorInstitute/tabctx/issues/36)).

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
