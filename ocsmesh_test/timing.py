"""Non-invasive timers for OCSMesh.

Wraps selected OCSMesh functions and methods with perf_counter timers so a
run can split ``meshdata`` time into stages (collector level: detection,
dispatch, application) and steps (what each refinement does inside a tile:
building the water polygon, buffering, KD-tree queries, masks, ...), without
changing OCSMesh. Times are inclusive and nest, so do not add them up.

Stage labels are plain (``apply_channels``); step labels start with
``step:`` (``step:get_multipolygon``).

The timers only see the process they are installed in: in ``parallel``
mode the Pool workers are not timed, and under MPI each rank times its own
work but only rank 0's numbers reach the summary. Profile with ``serial``
runs: the code inside each step is the same in every mode.
"""

from __future__ import annotations

import functools
import importlib
import logging
import time
from typing import Dict

_logger = logging.getLogger(__name__)
_TIMES: Dict[str, Dict[str, float]] = {}
_INSTALLED = False

STEP_PREFIX = "step:"

# (module, attribute path inside the module, label)
_STAGES = [
    ("ocsmesh.hfun.collector", "HfunCollector._apply_features", "apply_features"),
    ("ocsmesh.hfun.collector", "HfunCollector._apply_features_fast", "apply_features_fast"),
    ("ocsmesh.hfun.collector", "HfunCollector._apply_contours", "apply_contours"),
    ("ocsmesh.hfun.collector", "HfunCollector._apply_flow_limiters", "apply_flow_limiters"),
    ("ocsmesh.hfun.collector", "HfunCollector._apply_const_val", "apply_const_val"),
    ("ocsmesh.hfun.collector", "HfunCollector._apply_linefeatures", "apply_linefeatures"),
    ("ocsmesh.hfun.collector", "HfunCollector._apply_patch", "apply_patch"),
    ("ocsmesh.hfun.collector", "HfunCollector._apply_channels", "apply_channels"),
    ("ocsmesh.hfun.collector", "HfunCollector._apply_constraints", "apply_constraints"),
    ("ocsmesh.hfun.collector", "HfunCollector._calculate_and_write_hfun_to_disk", "write_hfun_to_disk"),
    ("ocsmesh.hfun.collector", "HfunCollector._get_hfun_composite", "composite"),
    ("ocsmesh.hfun.collector", "HfunCollector._dispatch_refinement_tasks", "dispatch_refinement"),
    ("ocsmesh.hfun.collector", "_RefinementContourCollector.calculate", "contour_detect"),
    ("ocsmesh.hfun.collector", "_ChannelRefineCollector.calculate", "channel_detect"),
]

_S = STEP_PREFIX
_STEPS = [
    # detection: contours and channels
    ("ocsmesh.raster", "Raster.get_contour", _S + "get_contour"),
    ("ocsmesh.raster", "Raster.get_channels", _S + "get_channels"),
    ("ocsmesh.raster", "Raster.get_multipolygon", _S + "get_multipolygon"),
    ("ocsmesh.utils", "get_polygon_channels", _S + "polygon_channels"),
    ("geopandas.base", "GeoPandasBase.buffer", _S + "geoseries_buffer"),
    ("geopandas.base", "GeoPandasBase.simplify", _S + "geoseries_simplify"),
    ("geopandas", "overlay", _S + "geopandas_overlay"),
    ("shapely.ops", "transform", _S + "shapely_transform"),
    # application: distance-based (contours, channels, features, patches)
    ("ocsmesh.hfun.raster", "HfunRaster.add_feature", _S + "add_feature"),
    ("ocsmesh.hfun.raster", "HfunRaster.add_patch", _S + "add_patch"),
    ("ocsmesh.hfun.raster", "HfunRaster.get_xy_memcache", _S + "get_xy_memcache"),
    ("ocsmesh.utils", "repartition_features", _S + "repartition_features"),
    ("ocsmesh.utils", "transform_linestring", _S + "transform_linestring"),
    ("rasterio.mask", "raster_geometry_mask", _S + "raster_geometry_mask"),
    # application: elevation-based
    ("ocsmesh.hfun.raster", "HfunRaster.add_constant_value", _S + "add_constant_value"),
    ("ocsmesh.hfun.raster", "HfunRaster.add_subtidal_flow_limiter", _S + "add_flow_limiter"),
    ("ocsmesh.hfun.raster", "HfunRaster.apply_added_constraints", _S + "apply_added_constraints"),
    ("ocsmesh.features.constraint", "TopoConstConstraint.apply", _S + "topo_bound_apply"),
    ("ocsmesh.features.constraint", "TopoFuncConstraint.apply", _S + "topo_func_apply"),
    ("ocsmesh.features.constraint", "CourantNumConstraint.apply", _S + "courant_apply"),
    ("ocsmesh.features.constraint", "RegionConstraint.apply", _S + "region_apply"),
    ("ocsmesh.features.constraint", "Constraint._apply_rate", _S + "constraint_rate"),
    # raster I/O
    ("ocsmesh.raster", "Raster.get_values", _S + "get_values"),
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
    timed.__ocsmesh_test_timed__ = True
    return timed


def _wrap_dispatch(fn):
    @functools.wraps(fn)
    def timed(self, kind, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(self, kind, *args, **kwargs)
        finally:
            _record(f"dispatch_{kind}", time.perf_counter() - t0)
    timed.__ocsmesh_test_timed__ = True
    return timed


class _TimedTree:
    """Proxy for a scipy cKDTree that times ``query``."""

    def __init__(self, tree):
        self._tree = tree

    def query(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self._tree.query(*args, **kwargs)
        finally:
            _record(_S + "kdtree_query", time.perf_counter() - t0)

    def __getattr__(self, name):
        return getattr(self._tree, name)


def _timed_kdtree_factory(orig):
    def build(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            tree = orig(*args, **kwargs)
        finally:
            _record(_S + "kdtree_build", time.perf_counter() - t0)
        return _TimedTree(tree)
    build.__ocsmesh_test_timed__ = True
    return build


def _patch(module_name: str, path: str, wrapper) -> bool:
    try:
        owner = importlib.import_module(module_name)
    except ImportError:
        return False
    *parents, attr = path.split(".")
    for name in parents:
        owner = getattr(owner, name, None)
        if owner is None:
            return False
    # the attribute as defined on this class (not inherited), so a method
    # defined on a base class is patched once, on the base class
    fn = owner.__dict__.get(attr) if isinstance(owner, type) else getattr(owner, attr, None)
    if fn is None or getattr(fn, "__ocsmesh_test_timed__", False):
        return False
    setattr(owner, attr, wrapper(fn))
    return True


def install(steps: bool = True) -> None:
    """Install the timers once (idempotent). Missing targets are skipped."""
    global _INSTALLED  # noqa: PLW0603
    if _INSTALLED:
        return
    targets = _STAGES + (_STEPS if steps else [])
    for module_name, path, label in targets:
        wrapper = (_wrap_dispatch if label == "dispatch_refinement"
                   else functools.partial(lambda fn, lab: _wrap(fn, lab), lab=label))
        if not _patch(module_name, path, wrapper):
            _logger.debug("timing: %s.%s not found", module_name, path)
    if steps:
        # the KD-tree used to measure distances to features
        _patch("ocsmesh.hfun.raster", "cKDTree", _timed_kdtree_factory)
    _INSTALLED = True


def snapshot() -> Dict[str, Dict[str, float]]:
    return {k: {"seconds": round(v["seconds"], 3), "calls": int(v["calls"])}
            for k, v in sorted(_TIMES.items())}


def reset() -> None:
    _TIMES.clear()
