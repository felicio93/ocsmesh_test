"""Mesh recipe: build Geom and Hfun from the config for one test case.

Refinement blocks live under ``refinements`` in the config; a test lists
which blocks it applies. This keeps the NJ recipe and any future domain in
data, not code.
"""

from __future__ import annotations

import functools
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import Config

_logger = logging.getLogger(__name__)


def build_geom(cfg: Config, raster_paths: List[str], domain, domain_crs: str, nprocs: int):
    """Build the domain geometry once and return a static Geom.

    GeomCollector is lazy and recomputes on every call, so the multipolygon
    is evaluated here (timed), repaired if invalid, and wrapped as a plain
    polygon Geom for MeshDriver.
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.validation import make_valid
    from ocsmesh import Geom

    t0 = time.perf_counter()
    geom = Geom(
        raster_paths,
        base_shape=domain,
        base_shape_crs=domain_crs,
        zmin=float(cfg.get("mesh.geom_zmin", -10000.0)),
        zmax=float(cfg.get("mesh.geom_zmax", 10.0)),
        nprocs=nprocs,
    )
    mp = geom.get_multipolygon()
    if not mp.is_valid:
        _logger.warning("geometry invalid; applying make_valid")
        fixed = make_valid(mp)
        parts = []
        for g in getattr(fixed, "geoms", [fixed]):
            if isinstance(g, Polygon):
                parts.append(g)
            elif isinstance(g, MultiPolygon):
                parts.extend(g.geoms)
        mp = MultiPolygon(parts)
    out = Geom(mp, crs=geom.crs)
    _logger.info("Geom built in %.1f s", time.perf_counter() - t0)
    return out


def build_hfun(cfg: Config, raster_paths: List[str], domain, domain_crs: str,
               nprocs: int, execution_mode: str, method: str, test: Dict[str, Any]):
    from ocsmesh import Hfun

    hfun = Hfun(
        raster_paths,
        hmin=float(cfg["mesh.hmin"]),
        hmax=float(cfg["mesh.hmax"]),
        nprocs=nprocs,
        base_shape=domain,
        base_shape_crs=domain_crs,
        method=method,
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        hfun.execution_mode = execution_mode

    refinements = cfg.get("refinements", {}) or {}
    for block in test.get("refinements", []):
        spec = refinements.get(block)
        if spec is None:
            raise KeyError(f"test {test['id']} uses unknown refinement block {block!r}")
        _apply_block(hfun, block, spec, cfg)
    return hfun


def depth_linear(z, a: float = 0.5, b: float = 0.0):
    """Size from depth for ``topo_func`` blocks: ``a * |z| + b``.

    Module level (with ``functools.partial``) so it pickles for the
    parallel and MPI modes."""
    return a * np.abs(z) + b


_TOPO_FUNCS = {"depth_linear": depth_linear}


def _bounds(spec: Dict[str, Any]) -> Dict[str, float]:
    kw = {}
    if spec.get("lower") is not None:
        kw["lower_bound"] = float(spec["lower"])
    if spec.get("upper") is not None:
        kw["upper_bound"] = float(spec["upper"])
    return kw


def _sources(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {"source_index": spec["source_index"]} if spec.get("source_index") is not None else {}


def _shape(cfg: Config, spec: Dict[str, Any], areal: bool):
    """Read ``spec['shapefile']`` (any vector format; relative to base_dir)."""
    import geopandas as gpd
    from shapely.ops import unary_union

    path = spec["shapefile"]
    gdf = gpd.read_file(path if Path(path).is_absolute() else cfg.base_dir / path)
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    shape = unary_union(geoms)
    kind = "Polygon" if areal else "LineString"
    if kind not in shape.geom_type:
        raise ValueError(f"{path}: expected {kind} geometries, got {shape.geom_type}")
    return shape, gdf.crs


def _rate(spec: Dict[str, Any], key: str = "rate"):
    value = spec.get(key)
    return None if value is None else float(value)


def _apply_block(hfun, block: str, spec: Any, cfg: Config) -> None:
    kind = block.split(":")[0]
    if kind == "depth_bands":
        for band in spec:
            kw = {"value": float(band["value"])}
            if band.get("lower") is not None:
                kw["lower_bound"] = float(band["lower"])
            if band.get("upper") is not None:
                kw["upper_bound"] = float(band["upper"])
            if band.get("source_index") is not None:
                kw["source_index"] = band["source_index"]
            hfun.add_constant_value(**kw)
    elif kind == "contour":
        for c in (spec if isinstance(spec, list) else [spec]):
            hfun.add_contour(level=float(c["level"]), expansion_rate=float(c["expansion_rate"]),
                             target_size=float(c["target_size"]))
    elif kind == "flow_limiters":
        for fl in spec:
            hfun.add_subtidal_flow_limiter(
                hmin=float(fl["hmin"]), hmax=float(fl["hmax"]),
                lower_bound=float(fl["lower"]), upper_bound=float(fl["upper"]),
                **({"source_index": fl["source_index"]} if fl.get("source_index") is not None else {}),
                # only passed when set: older OCSMesh has no flat_areas argument
                **({"flat_areas": fl["flat_areas"]} if fl.get("flat_areas") is not None else {}))
    elif kind == "channels":
        for ch in spec:
            hfun.add_channel(level=float(ch.get("level", 0.0)), width=float(ch["width"]),
                             target_size=float(ch["target_size"]),
                             expansion_rate=ch.get("expansion_rate"),
                             tolerance=ch.get("tolerance"))
    elif kind == "region_constraints":
        for rc in spec:
            shape, crs = _shape(cfg, rc, areal=True)
            hfun.add_region_constraint(value=float(rc["value"]), shape=shape, crs=crs,
                                       value_type=rc.get("value_type", "min"),
                                       rate=_rate(rc), **_sources(rc))
    elif kind == "topo_bound":          # add_topo_bound_constraint
        for tb in spec:
            hfun.add_topo_bound_constraint(value=float(tb["value"]), **_bounds(tb),
                                           value_type=tb.get("value_type", "min"),
                                           rate=_rate(tb), **_sources(tb))
    elif kind == "topo_func":           # add_topo_func_constraint
        for tf in spec:
            fn_spec = dict(tf["func"])
            fn = functools.partial(_TOPO_FUNCS[fn_spec.pop("name")],
                                   **{k: float(v) for k, v in fn_spec.items()})
            hfun.add_topo_func_constraint(func=fn, **_bounds(tf),
                                          value_type=tf.get("value_type", "min"),
                                          rate=_rate(tf), **_sources(tf))
    elif kind == "courant":             # add_courant_num_constraint
        for cn in spec:
            kw = {"upper_bound": float(cn.get("upper", 0.9)),
                  "timestep": float(cn.get("timestep", 150)),
                  "wave_amplitude": float(cn.get("wave_amplitude", 2))}
            if cn.get("lower") is not None:
                kw["lower_bound"] = float(cn["lower"])
            hfun.add_courant_num_constraint(**kw, **_sources(cn))
    elif kind == "patches":             # add_patch
        for pa in spec:
            shape, _ = _shape(cfg, pa, areal=True)
            hfun.add_patch(shape=shape, target_size=float(pa["target_size"]),
                           expansion_rate=_rate(pa, "expansion_rate"))
    elif kind == "features":            # add_feature (user lines)
        for fe in spec:
            shape, crs = _shape(cfg, fe, areal=False)
            hfun.add_feature(shape=shape, target_size=float(fe["target_size"]),
                             expansion_rate=float(fe.get("expansion_rate", 0.01)), crs=crs)
    else:
        raise KeyError(f"unknown refinement kind {kind!r} in block {block!r}")
