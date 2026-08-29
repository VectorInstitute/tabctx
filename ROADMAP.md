# Roadmap

Nearly everything the original roadmap tracked has shipped and been
validated on real hardware -- see CHANGELOG.md v0.6.0-v0.9.0 for what
landed and the measured evidence. This file now holds only the
remainder, and gets deleted when it empties.

## Remaining

1. **Post-release: ephemeral GKE stress run + researcher deployment**
   (deliberately sequenced after the release). Everything is staged in
   `VectorInstitute/inference-platform` branch `test/tabicl-gke-onboard`:
   the research overlay (`tabctx-research-values.yaml`: both models, one
   A100, spillover, 64Gi host RAM per the measured host-OOM boundary),
   probes (`probe_models.py`, `probe_stress_v09.py`, plus the existing
   battery), and a parameterized `onboard.sh`
   (`OVERLAY=... RELEASE=tabctx-research TABPFN_TOKEN=... KEEP_ALIVE=true`).
   The prod inference-platform deployment for Vector researchers follows.
2. **TabPFN memory calibration grid**: the preloaded admission grid is
   TabICL data; TabPFN deployments fall back to the conservative formula
   plus runtime learning. Run `benchmarks/calibrate_memory.py`-style
   sweeps through `TabPFNBackend` on an A100 and add the grid.
3. **Docs site**: README carries the full story today; a docs site
   matters as external adoption starts.

## Standing process note

Every significant fix in this project came from testing against real
systems -- most recently the shared-backbone kv-cache pin (a ~15GB
eviction leak) and the fragmentation OOM at 90% budget, both of which
only reproduce on a real GPU. Keep reading model internals AND deploying
before believing anything.
