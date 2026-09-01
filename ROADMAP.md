# Roadmap

Nearly everything the original roadmap tracked has shipped and been
validated on real hardware -- see CHANGELOG.md v0.6.0-v0.9.0 for what
landed and the measured evidence. This file now holds only the
remainder, and gets deleted when it empties.

## Remaining

1. **Application catalog + persistent context store** (asked for by
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
2. **Post-release: ephemeral GKE stress run + researcher deployment**
   (deliberately sequenced after the release). Everything is staged in
   `VectorInstitute/inference-platform` branch `test/tabicl-gke-onboard`:
   the research overlay (`tabctx-research-values.yaml`: both models, one
   A100, spillover, 64Gi host RAM per the measured host-OOM boundary),
   probes (`probe_models.py`, `probe_stress_v09.py`, plus the existing
   battery), and a parameterized `onboard.sh`
   (`OVERLAY=... RELEASE=tabctx-research TABPFN_TOKEN=... KEEP_ALIVE=true`).
   The prod inference-platform deployment for Vector researchers follows.
3. **TabPFN memory calibration grid**: the preloaded admission grid is
   TabICL data; TabPFN deployments fall back to the conservative formula
   plus runtime learning. Run `benchmarks/calibrate_memory.py`-style
   sweeps through `TabPFNBackend` on an A100 and add the grid.
4. **Docs site**: README carries the full story today; a docs site
   matters as external adoption starts.

## Standing process note

Every significant fix in this project came from testing against real
systems -- most recently the shared-backbone kv-cache pin (a ~15GB
eviction leak) and the fragmentation OOM at 90% budget, both of which
only reproduce on a real GPU. Keep reading model internals AND deploying
before believing anything.
