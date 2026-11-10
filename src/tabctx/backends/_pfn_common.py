"""What the SurvivalPFN and CausalPFN backends share.

Both models come from the same family (a TabDPT-style transformer, about
20M parameters, 75 MB of weights on Hugging Face) and are served the same
way:

- Weights are tensors only. Their packages load checkpoints with
  torch.load(weights_only=False), which runs whatever the file's pickle
  says; tabctx loads them itself with weights_only=True (both published
  checkpoints are a state dict and a plain config) and hands the network
  to each fit. A checkpoint that needs arbitrary unpickling fails to load.
- Revisions are pinned, so a served model id always means the same
  weights. hf_hub_download honours HF_HUB_CACHE and HF_HUB_OFFLINE, so
  weights staged on a node load without network access.
- One network per device, shared by every context (the context is the
  table; the network is the same for every tenant).
- A cached context is its table, on the host; the spill tier pickles it
  without the network and re-attaches this device's on load (the
  network is always payload.estimator.icl_model).
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


def dumps_context(payload: Any, tag: bytes) -> bytes:
    """A context (whose network is payload.estimator.icl_model) pickled
    without its network, prefixed with tag (the model version it was fit
    with)."""
    estimator = payload.estimator
    network = estimator.icl_model
    try:
        estimator.icl_model = None
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        estimator.icl_model = network
    return tag + data


def loads_context(data: bytes, tag: bytes, network: Any, device: str) -> Any:
    """The inverse of dumps_context, on this device's network. ValueError
    for a context from another model version."""
    if not data.startswith(tag):
        raise ValueError(f"not a context from this model version ({tag!r})")
    payload = pickle.load(io.BytesIO(data[len(tag) :]))
    payload.estimator.icl_model = network
    payload.estimator.device = device
    return payload


def array_bytes(*arrays: np.ndarray | None) -> int:
    """Host bytes of a context's arrays (what the cache charges it)."""
    return sum(int(a.nbytes) for a in arrays if a is not None)


def float32_rows(X: Any) -> np.ndarray:
    """Feature rows as a contiguous float32 matrix (NaN = missing)."""
    return np.ascontiguousarray(np.asarray(X, dtype=np.float32))
