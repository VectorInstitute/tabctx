"""What the SurvivalPFN and CausalPFN backends share.

Both models come from the same family (a TabDPT-style transformer, about
20M parameters, 75 MB of weights on Hugging Face) and are served the same
way:

- Weights are tensors only. Their packages load checkpoints with
  torch.load(weights_only=False), which runs whatever the file's pickle
  says; tabctx loads them itself with weights_only=True (both published
  checkpoints are a state dict and a plain config). A checkpoint that
  needs arbitrary unpickling fails to load.
- Revisions are pinned, so a served model id always means the same
  weights. hf_hub_download honours HF_HUB_CACHE and HF_HUB_OFFLINE, so
  weights staged on a node load without network access.
- One network per device, shared by every context (the context is the
  table; the network is the same for every tenant).
- The context is the training table, at most a fixed number of rows of
  it: a larger table is subsampled at fit, keeping each group's share
  (events and censored rows; treated and control rows), seeded. Test rows
  attend only to the context, so a row's answer never depends on the
  other rows in its request (the server merges and splits requests).
- A cached context lives on the host; the spill tier pickles it without
  any network, tagged with the model version.
"""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np


def load_checkpoint(repo_id: str, filename: str, revision: str) -> dict:
    """A pinned checkpoint from Hugging Face, loaded as tensors and plain
    containers only (weights_only=True)."""
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    if "model_state_dict" not in ckpt or "model_config" not in ckpt:
        raise ValueError(
            f"{repo_id}/{filename}@{revision} is not a model_state_dict + "
            "model_config checkpoint"
        )
    return ckpt


def subsample(groups: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Sorted indices of at most `size` rows, keeping each group's share
    of the rows (rounded, at least one row of every group present). All
    rows when there are no more than `size`."""
    n = len(groups)
    if n <= size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    values, counts = np.unique(groups, return_counts=True)
    quota = np.maximum(1, np.round(counts * size / n).astype(int))
    while quota.sum() > size:  # rounding up the small groups overshot
        quota[np.argmax(quota)] -= 1
    picked = [
        rng.choice(np.flatnonzero(groups == value), k, replace=False)
        for value, k in zip(values, quota, strict=True)
    ]
    return np.sort(np.concatenate(picked))


def tagged_dumps(obj: Any, tag: bytes) -> bytes:
    """obj pickled, prefixed with tag (the model version it was fit with)."""
    return tag + pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def tagged_loads(data: bytes, tag: bytes) -> Any:
    """The inverse of tagged_dumps. ValueError for data from another model
    version."""
    if not data.startswith(tag):
        raise ValueError(f"not a context from this model version ({tag!r})")
    return pickle.load(io.BytesIO(data[len(tag) :]))


def dumps_context(payload: Any, tag: bytes) -> bytes:
    """A context whose network is payload.estimator.icl_model, pickled
    without it (tagged_dumps)."""
    estimator = payload.estimator
    network = estimator.icl_model
    try:
        estimator.icl_model = None
        return tagged_dumps(payload, tag)
    finally:
        estimator.icl_model = network


def loads_context(data: bytes, tag: bytes, network: Any, device: str) -> Any:
    """The inverse of dumps_context, on this device's network."""
    payload = tagged_loads(data, tag)
    payload.estimator.icl_model = network
    payload.estimator.device = device
    return payload


def array_bytes(*arrays: np.ndarray | None) -> int:
    """Host bytes of a context's arrays (what the cache charges it)."""
    return sum(int(a.nbytes) for a in arrays if a is not None)


def float32_rows(X: Any) -> np.ndarray:
    """Feature rows as a contiguous float32 matrix (NaN = missing)."""
    return np.ascontiguousarray(np.asarray(X, dtype=np.float32))
