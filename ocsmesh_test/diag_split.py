"""Where do split runs lose channels? One tile, serial, step by step.

    ocsmesh-test diag-split -c CONFIG --variant VARIANT --test 4 --lon LON --lat LAT

Runs the channel detection of one test on the DEM tile(s) holding
(lon, lat), once unsplit and once with the variant's ``split_size``, in
serial mode, and prints the channel area within ``--radius`` degrees of
the point after each step:

* ``get_channels``: what detection returns (per tile, or per window with
  its halo), and whether ``get_polygon_channels`` took its "all is
  channel" shortcut;
* ``clip_to_core``: the window's channels kept in its core;
* ``drop_small_channels``: after joining the pieces and the size cutoff.

The unsplit ``get_channels`` area is the target; the first step where
the split area falls short is the one that loses the channels.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Dict, List
from unittest import mock

_logger = logging.getLogger(__name__)


def _km2(geom, lat: float) -> float:
    """Area of a geometry in degrees, in km2 (good enough near ``lat``)."""
    if geom is None or geom.is_empty:
        return 0.0
    return geom.area * (111.32 ** 2) * math.cos(math.radians(lat))


def _near(geom, box):
    import shapely
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    return geom.intersection(box)


def _tiles_at(paths: List[str], lon: float, lat: float) -> List[str]:
    import rasterio
    from pyproj import Transformer
    out = []
    for p in paths:
        with rasterio.open(p) as src:
            x, y = lon, lat
            if src.crs is not None and not src.crs.is_geographic:
                x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
            b = src.bounds
            if min(b.left, b.right) <= x <= max(b.left, b.right) and \
                    min(b.bottom, b.top) <= y <= max(b.bottom, b.top):
                out.append(p)
    return out


def run(cfg, test_id: int, lon: float, lat: float, radius: float) -> List[Dict[str, Any]]:
    from shapely.geometry import box

    import ocsmesh
    from ocsmesh import utils
    from ocsmesh.hfun import split as hsplit

    from .data import manifest as manifest_mod
    from .recipe import build_hfun

    entries = manifest_mod.load(cfg)
    domain, domain_crs = manifest_mod.load_domain(cfg)
    paths = manifest_mod.raster_paths(
        entries, domain, require_polygon=bool(cfg.get("dems.require_polygon", True)))
    tiles = _tiles_at(paths, lon, lat)
    if not tiles:
        raise SystemExit(f"no DEM tile holds ({lon}, {lat})")
    print(f"tiles at ({lon}, {lat}): {tiles}")
    near_box = box(lon - radius, lat - radius, lon + radius, lat + radius)
    test = cfg.test(test_id)
    rows: List[Dict[str, Any]] = []

    orig_get_channels = ocsmesh.Raster.get_channels
    orig_poly = utils.get_polygon_channels
    orig_clip = hsplit.clip_to_core
    orig_drop = hsplit.drop_small_channels

    def get_channels(self, *a, **k):
        out = orig_get_channels(self, *a, **k)
        b = self.get_bbox(crs="EPSG:4326").bounds
        rows.append({"run": state["run"], "step": "get_channels",
                     "drop_small": k.get("drop_small", True),
                     "window": f"{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}",
                     "km2": _km2(out, lat), "near_km2": _km2(_near(out, near_box), lat),
                     "type": type(out).__name__, "all_channel": state.pop("all", False)})
        return out

    def poly(polygon, *a, **k):
        out = orig_poly(polygon, *a, **k)
        state["all"] = out is polygon
        return out

    def clip(shape, crs, core):
        out = orig_clip(shape, crs, core)
        rows.append({"run": state["run"], "step": "clip_to_core",
                     "window": "core " + ",".join(f"{v:.3f}" for v in core.bounds()),
                     "valid_in": bool(shape.is_valid), "km2": _km2(out, lat),
                     "near_km2": _km2(_near(out, near_box), lat),
                     "near_in_core_before_km2": _km2(_near(_near(shape, near_box),
                                                         box(*core.bounds())), lat)})
        return out

    def drop(items, width):
        out = orig_drop(items, width)
        for (shape, _, core), kept in zip(items, out):
            rows.append({"run": state["run"], "step": "drop_small_channels",
                         "window": "core " + ",".join(f"{v:.3f}" for v in core.bounds())
                         if core is not None else "unsplit",
                         "km2": _km2(kept, lat), "near_km2": _km2(_near(kept, near_box), lat),
                         "near_in_km2": _km2(_near(shape, near_box), lat)})
        return out

    state: Dict[str, Any] = {}
    hfun_opts = dict(cfg.get("hfun_options", {}) or {})
    with mock.patch.object(ocsmesh.Raster, "get_channels", get_channels), \
            mock.patch.object(utils, "get_polygon_channels", poly), \
            mock.patch.object(hsplit, "clip_to_core", clip), \
            mock.patch.object(hsplit, "drop_small_channels", drop):
        for run_name, split_size in (("unsplit", None), ("split", hfun_opts.get("split_size"))):
            state["run"] = run_name
            cfg.raw.setdefault("hfun_options", {})
            cfg.raw["hfun_options"] = {**hfun_opts, "split_size": split_size}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hfun = build_hfun(cfg, tiles, domain, domain_crs, nprocs=1,
                                  execution_mode="serial", method="exact", test=test)
                hfun._apply_features()  # pylint: disable=protected-access
    cfg.raw["hfun_options"] = hfun_opts
    return rows


def render(rows: List[Dict[str, Any]]) -> str:
    keys = ["run", "step", "window", "drop_small", "all_channel", "valid_in", "type",
            "km2", "near_km2", "near_in_km2", "near_in_core_before_km2"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
