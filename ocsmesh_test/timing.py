"""Non-invasive per-phase timers for OCSMesh's HfunCollector.

Wraps selected private methods with perf_counter timers so a run can split
``meshdata`` time into detection, dispatch and application without changing
OCSMesh. Times are inclusive and nest, so do not add them up.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Dict

_logger = logging.getLogger(__name__)
_TIMES: Dict[str, Dict[str, float]] = {}
_INSTALLED = False

# (class in ocsmesh.hfun.collector, attribute, label)
_TARGETS = [
    ("HfunCollector", "_apply_features", "apply_features"),
    ("HfunCollector", "_apply_features_fast", "apply_features_fast"),
    ("HfunCollector", "_apply_contours", "apply_contours"),
    ("HfunCollector", "_apply_flow_limiters", "apply_flow_limiters"),
    ("HfunCollector", "_apply_const_val", "apply_const_val"),
    ("HfunCollector", "_apply_linefeatures", "apply_linefeatures"),
    ("HfunCollector", "_apply_patch", "apply_patch"),
    ("HfunCollector", "_apply_channels", "apply_channels"),
    ("HfunCollector", "_apply_constraints", "apply_constraints"),
    ("HfunCollector", "_calculate_and_write_hfun_to_disk", "write_hfun_to_disk"),
    ("HfunCollector", "_get_hfun_composite", "composite"),
    ("HfunCollector", "_dispatch_refinement_tasks", "dispatch_refinement"),
    ("_RefinementContourCollector", "calculate", "contour_detect"),
    ("_ChannelRefineCollector", "calculate", "channel_detect"),
]


def _record(label: str, seconds: float) -> None:
    rec = _TIMES.setdefault(label, {"seconds": 0.0, "calls": 0})
    rec["seconds"] += seconds
    rec["calls"] += 1


def _wrap(fn, label):
    @functools.wraps(fn)
    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _record(label, time.perf_counter() - t0)
    return timed


def _wrap_dispatch(fn):
    @functools.wraps(fn)
    def timed(self, kind, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(self, kind, *args, **kwargs)
        finally:
            _record(f"dispatch_{kind}", time.perf_counter() - t0)
    return timed


def install() -> None:
    """Patch the collector classes once (idempotent). Missing targets are skipped."""
    global _INSTALLED  # noqa: PLW0603
    if _INSTALLED:
        return
    from ocsmesh.hfun import collector as mod
    for cls_name, attr, label in _TARGETS:
        cls = getattr(mod, cls_name, None)
        fn = getattr(cls, attr, None) if cls is not None else None
        if fn is None:
            _logger.debug("phase timing: %s.%s not found", cls_name, attr)
            continue
        setattr(cls, attr, _wrap_dispatch(fn) if label == "dispatch_refinement" else _wrap(fn, label))
    _INSTALLED = True


def snapshot() -> Dict[str, Dict[str, float]]:
    return {k: {"seconds": round(v["seconds"], 2), "calls": int(v["calls"])}
            for k, v in sorted(_TIMES.items())}


def reset() -> None:
    _TIMES.clear()
