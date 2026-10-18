"""Where a fit goes among a replica's devices: a pluggable policy (#39).

The router (multidevice.py) owns the mechanics -- snapshots, per-id
locks, re-fit ordering -- and asks a ``PlacementPolicy`` only the
decision: given each device's view, which one takes this fit. Keeping
the decision separate lets a platform that schedules LLMs and tabular
models together (Ray Serve today, Dynamo tomorrow) supply its own policy,
e.g. one that avoids a GPU it is about to give to an LLM, without
touching the router.

A view is a lock-free snapshot, so a policy never waits on a device's
work, and it must be cheap: it runs on every fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FitRequest:
    n_train: int
    n_features: int
    backend: str
    dataset_id: str | None = None


@dataclass(frozen=True)
class DeviceView:
    """One device as a policy sees it."""

    index: int
    device: str
    headroom_bytes: int  # what its admission gate offers now
    need_bytes: int  # this fit's estimated peak there
    in_flight: int  # calls running on it
    recent_calls: int  # calls started within the activity window
    alive: bool = True

    @property
    def fits_without_eviction(self) -> bool:
        return self.alive and self.headroom_bytes >= self.need_bytes


class PlacementPolicy(Protocol):
    def choose(self, request: FitRequest, devices: Sequence[DeviceView]) -> int:
        """The index of the device that takes the fit."""
        ...


class LeastActiveThatFits:
    """The default, aimed at a few concurrent users fitting small and large
    tables and predicting against them:

    1. Devices where the fit's peak fits WITHOUT evicting anything win;
       among them the least active (fewest calls in flight, then fewest
       started recently), ties to the most headroom. Each device runs one
       fit at a time, so spreading active users is what lets them run
       concurrently.
    2. Otherwise (a large table, or a full replica) the device with the
       most headroom left after the fit, i.e. the fewest evictions.

    Dead devices are never chosen while a live one exists."""

    def choose(self, request: FitRequest, devices: Sequence[DeviceView]) -> int:
        del request
        live = [d for d in devices if d.alive] or list(devices)
        fits = [d for d in live if d.fits_without_eviction]
        if fits:
            best = min(
                fits,
                key=lambda d: (d.in_flight, d.recent_calls, -d.headroom_bytes, d.index),
            )
        else:
            best = max(live, key=lambda d: (d.headroom_bytes - d.need_bytes, -d.index))
        return best.index
