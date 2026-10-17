#!/usr/bin/env python3
"""Per-stage GPU profile of fit and predict for one backend on one GPU (#31).

Answers "where do the time and the memory peak go?" for each table shape,
so optimizations are chosen from measurements:

- fit / predict wall time, transient peak and resident bytes (the same
  quantities tabctx's admission and cache accounting use);
- per-stage CUDA time and peak memory above the stage's entry level, via
  forward hooks on the model's top-level stages (TabICL: col_embedder,
  row_interactor, icl_predictor; TabPFN: the model's direct children).
  Stages called several times (TabICL's InferenceManager batches them)
  are summed for time and maxed for peak;
- optionally (--trace-shape), a torch.profiler run: the top CUDA kernels
  by time and the top ops by memory, plus a Chrome trace.

Runs the tabctx backends directly (no serving), on one device.

Usage (on a GPU host with tabicl/tabpfn and the weights available):
  profile_fit_predict.py --backend tabicl --device cuda:0 \
      --shapes 1000x10 10000x10 45000x10 5000x100 1000x500 500x1000 \
      --out profile-tabicl-a100.jsonl --trace-shape 10000x100 --trace-dir traces/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

MiB = 1024**2


def parse_shape(text: str) -> tuple[int, int]:
    rows, features = text.lower().split("x")
    return int(rows), int(features)


def make_table(rows: int, features: int, task: str, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, features)).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, min(1, features - 1)]
    y = (signal > 0).astype(int) if task == "classification" else signal
    return X, y


def build_backend(kind: str, device: str, cache_mode: str):
    if kind == "tabicl":
        from tabctx.backends.tabicl import TabICLBackend

        kv: bool | str = False if cache_mode == "off" else cache_mode
        return TabICLBackend(device=device, kv_cache=kv)
    from tabctx.backends.tabpfn import TabPFNBackend

    return TabPFNBackend(device=device, cache_mode=cache_mode)


def stage_modules(kind: str, backend, payload, task: str) -> dict[str, torch.nn.Module]:
    """The model's top-level stages, from a fitted payload."""
    if kind == "tabicl":
        model = backend._shared_backbones[task]["model_"]
        return {
            name: getattr(model, name)
            for name in ("col_embedder", "row_interactor", "icl_predictor")
            if hasattr(model, name)
        }
    model = None
    for attr in ("models_", "model_"):
        found = getattr(payload, attr, None)
        if isinstance(found, list | tuple) and found:
            model = found[0]
        elif isinstance(found, torch.nn.Module):
            model = found
        if model is not None:
            break
    if model is None:
        executor = getattr(payload, "executor_", None)
        models = getattr(executor, "models", None) or getattr(executor, "model", None)
        model = models[0] if isinstance(models, list | tuple) else models
    if not isinstance(model, torch.nn.Module):
        return {}
    return dict(model.named_children())


class StageRecorder:
    """CUDA-event timing and peak memory per stage.

    Wraps each stage's entry methods (forward, plus TabICL's cache-mode
    entry points forward_with_cache / prepare_repr_cache /
    forward_with_repr_cache, which bypass nn.Module hooks), recording only
    the outermost call so a method calling another isn't counted twice."""

    ENTRY_METHODS = (
        "forward",
        "forward_with_cache",
        "prepare_repr_cache",
        "forward_with_repr_cache",
    )

    def __init__(self, device: str, modules: dict[str, torch.nn.Module]) -> None:
        self.device = torch.device(device)
        self.calls: dict[str, int] = defaultdict(int)
        self.time_ms: dict[str, float] = defaultdict(float)
        self.peak_bytes: dict[str, int] = defaultdict(int)
        self._depth: dict[str, int] = defaultdict(int)
        self._patched: list[tuple[torch.nn.Module, str]] = []
        for name, module in modules.items():
            for method in self.ENTRY_METHODS:
                if callable(getattr(module, method, None)):
                    setattr(module, method, self._wrap(name, getattr(module, method)))
                    self._patched.append((module, method))

    def _wrap(self, name, fn):
        def wrapped(*args, **kwargs):
            if self._depth[name]:
                return fn(*args, **kwargs)
            self._depth[name] += 1
            torch.cuda.synchronize(self.device)
            base = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            try:
                return fn(*args, **kwargs)
            finally:
                end.record()
                torch.cuda.synchronize(self.device)
                self._depth[name] -= 1
                self.calls[name] += 1
                self.time_ms[name] += start.elapsed_time(end)
                peak = torch.cuda.max_memory_allocated(self.device) - base
                self.peak_bytes[name] = max(self.peak_bytes[name], peak)

        return wrapped

    def reset(self) -> None:
        self.calls.clear()
        self.time_ms.clear()
        self.peak_bytes.clear()

    def summary(self) -> dict:
        return {
            name: {
                "calls": self.calls[name],
                "cuda_ms": round(self.time_ms[name], 1),
                "peak_mib": round(self.peak_bytes[name] / MiB, 1),
            }
            for name in self.calls
        }

    def close(self) -> None:
        for module, method in self._patched:
            delattr(module, method)  # the instance attribute shadowing the method


def timed(device: torch.device, fn):
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize(device)
    return out, time.perf_counter() - t0


def profile_shape(args, backend, rows, features, recorder_box) -> dict:
    device = torch.device(args.device)
    X, y = make_table(rows + args.n_test, features, args.task, seed=rows * 7 + features)
    X_train, y_train, X_test = X[:rows], y[:rows], X[rows:]
    record: dict = {
        "backend": args.backend,
        "task": args.task,
        "cache_mode": args.cache_mode,
        "rows": rows,
        "features": features,
        "n_test": args.n_test,
        "gpu": torch.cuda.get_device_name(device),
    }
    try:
        recorder = recorder_box.get("recorder")
        if recorder is not None:
            recorder.reset()
        payload, fit_s = timed(device, lambda: backend.fit(X_train, y_train, args.task))
        record["fit_s"] = round(fit_s, 3)
        record["fit_peak_mib"] = round((backend.fit_peak_bytes_hint() or 0) / MiB, 1)
        record["resident_mib"] = round(
            (backend.context_bytes_hint(rows, features) or 0) / MiB, 1
        )
        if recorder is None and args.stages:
            recorder = StageRecorder(
                args.device, stage_modules(args.backend, backend, payload, args.task)
            )
            recorder_box["recorder"] = recorder
            # First fit loaded the weights; re-fit so stages are measured.
            payload = None
            torch.cuda.empty_cache()
            payload, fit_s = timed(
                device, lambda: backend.fit(X_train, y_train, args.task)
            )
            record["fit_s"] = round(fit_s, 3)
            record["fit_peak_mib"] = round(
                (backend.fit_peak_bytes_hint() or 0) / MiB, 1
            )
        if recorder is not None:
            record["fit_stages"] = recorder.summary()
            recorder.reset()
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        _, predict_s = timed(device, lambda: backend.predict(payload, X_test, False))
        record["predict_s"] = round(predict_s, 3)
        record["predict_peak_mib"] = round(
            (torch.cuda.max_memory_allocated(device) - base) / MiB, 1
        )
        _, warm_s = timed(device, lambda: backend.predict(payload, X_test[:10], False))
        record["predict10_s"] = round(warm_s, 4)
        if recorder is not None:
            record["predict_stages"] = recorder.summary()
            recorder.reset()
        record["status"] = "ok"
        payload = None
    except Exception as e:  # noqa: BLE001 -- OOM and model limits are results
        record["status"] = type(e).__name__
        record["error"] = str(e)[:300]
    torch.cuda.empty_cache()
    return record


def trace_shape(args, backend, rows, features) -> None:
    from torch.profiler import ProfilerActivity, profile

    out_dir = Path(args.trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = make_table(rows + args.n_test, features, args.task, seed=1)
    backend.fit(X[:rows], y[:rows], args.task)  # warm
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        payload = backend.fit(X[:rows], y[:rows], args.task)
        backend.predict(payload, X[rows:], False)
    stem = f"{args.backend}-{args.cache_mode}-{rows}x{features}"
    prof.export_chrome_trace(str(out_dir / f"{stem}.trace.json"))
    averages = prof.key_averages()
    (out_dir / f"{stem}.top-cuda.txt").write_text(
        averages.table(sort_by="cuda_time_total", row_limit=30)
    )
    (out_dir / f"{stem}.top-mem.txt").write_text(
        averages.table(sort_by="self_cuda_memory_usage", row_limit=30)
    )
    print(f"trace written: {out_dir / stem}.*", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["tabicl", "tabpfn"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--task", choices=["classification", "regression"], default="classification"
    )
    p.add_argument("--cache-mode", choices=["kv", "repr", "off"], default="kv")
    p.add_argument("--shapes", nargs="+", type=parse_shape, required=True)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--no-stages", dest="stages", action="store_false")
    p.add_argument("--out", default=None, help="append JSON lines here")
    p.add_argument("--trace-shape", type=parse_shape, default=None)
    p.add_argument("--trace-dir", default="traces")
    args = p.parse_args()

    backend = build_backend(args.backend, args.device, args.cache_mode)
    warm_X, warm_y = make_table(200, 8, args.task, seed=0)
    backend.warmup()  # load weights outside measurements
    recorder_box: dict = {}
    out = open(args.out, "a") if args.out else None  # noqa: SIM115
    for rows, features in args.shapes:
        record = profile_shape(args, backend, rows, features, recorder_box)
        line = json.dumps(record)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()
    if args.trace_shape:
        trace_shape(args, backend, *args.trace_shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
