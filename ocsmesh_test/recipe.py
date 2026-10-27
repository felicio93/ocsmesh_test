"""Mesh recipe: build Geom and Hfun from the config for one test case.

Refinement blocks live under ``refinements`` in the config; a test lists
which blocks it applies. This keeps the NJ recipe and any future domain in
data, not code.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List

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
        _apply_block(hfun, block, spec)
    return hfun


def _apply_block(hfun, block: str, spec: Any) -> None:
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
                **({"source_index": fl["source_index"]} if fl.get("source_index") is not None else {}))
    elif kind == "channels":
        for ch in spec:
            hfun.add_channel(level=float(ch.get("level", 0.0)), width=float(ch["width"]),
                             target_size=float(ch["target_size"]),
                             expansion_rate=ch.get("expansion_rate"),
                             tolerance=ch.get("tolerance"))
    elif kind == "region_constraints":
        import geopandas as gpd
        for rc in spec:
            gdf = gpd.read_file(rc["shapefile"])
            hfun.add_region_constraint(value=float(rc["value"]), shape=gdf.union_all(),
                                       crs=gdf.crs, value_type=rc.get("value_type", "min"),
                                       rate=rc.get("rate", 0.01))
    else:
        raise KeyError(f"unknown refinement kind {kind!r} in block {block!r}")
