"""Nodata handling in the size function (handoff N1 and the exact-method seam artifact)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box

from . import CheckResult, check, is_manager
from ._common import HMAX, HMIN, add_depth_bands, hfun_values
from ..data import synthetic


@check("hfun.empty_feature_keeps_values",
       "add_feature/add_patch with an empty shape leaves the size function unchanged "
       "(an unwritten destination file used to zero it, which clamps to hmin)", issue="N1")
def empty_feature_keeps_values(tmp: Path) -> CheckResult:
    from shapely.geometry import MultiPolygon
    from ocsmesh import Raster
    from ocsmesh.hfun.raster import HfunRaster

    tile = synthetic.write_tile(tmp / "A.tif", 0.002, (0.0, 0.0, 0.1, 0.1))
    h = HfunRaster(Raster(str(tile)), hmin=HMIN, hmax=HMAX)
    h.add_constant_value(1000.0)
    before = h.get_values().copy()
    h.add_feature(feature=MultiLineString([]), expansion_rate=0.01, target_size=500.0, nprocs=1)
    after_feature = h.get_values().copy()
    problems = []
    if not np.array_equal(before, after_feature):
        problems.append(f"add_feature(empty) changed values to {np.unique(after_feature)[:5]}")
    try:
        h.add_patch(MultiPolygon([]), expansion_rate=0.01, target_size=500.0, nprocs=1)
        if not np.array_equal(before, h.get_values()):
            problems.append(f"add_patch(empty) changed values to {np.unique(h.get_values())[:5]}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"add_patch(empty) raised {type(exc).__name__}")
    if problems:
        return CheckResult("fail", "; ".join(problems))
    return CheckResult("pass", "values unchanged by empty feature and empty patch")


@check("hfun.all_nodata_tile_with_channels",
       "a clipped tile that is 100% nodata must not flatten the size function "
       "(NJ t4/t5 came out uniform 50 m)", issue="N1")
def all_nodata_tile_with_channels(tmp: Path) -> CheckResult:
    from ocsmesh import Hfun

    a = synthetic.write_tile(tmp / "A.tif", 0.002, (0.0, 0.0, 0.2, 0.2),
                             depth_fn=lambda x, y: synthetic.default_bathymetry(x, y, coast_x=0.15, span=0.1))
    b = synthetic.write_tile(tmp / "B.tif", 0.002, (0.2, 0.0, 0.4, 0.2), all_nodata=True)
    # AOI covers A and a sliver of B's bbox: B survives Raster.clip as all-nodata
    aoi = box(0.0, 0.0, 0.2, 0.2).union(box(0.19, 0.05, 0.21, 0.06))

    def run(paths):
        h = Hfun(paths, hmin=HMIN, hmax=HMAX, nprocs=1, base_shape=aoi,
                 base_shape_crs="EPSG:4326", method="exact")
        h.execution_mode = "serial"
        h.add_channel(level=0.0, width=2000.0, target_size=500.0, expansion_rate=0.01, tolerance=50)
        md = h.meshdata()
        if md is None:
            return None
        return np.asarray(md.values).ravel()

    ref = run([str(a)])
    try:
        v = run([str(a), str(b)])
    except Exception as exc:  # noqa: BLE001
        return CheckResult("fail", f"all-nodata tile crashed the run: {type(exc).__name__}: {str(exc)[:80]}")
    if not is_manager():
        return CheckResult("skip", "worker rank")
    m = {"min_with_tile": float(v.min()), "min_without_tile": float(ref.min()),
         "mean_with_tile": float(v.mean()), "mean_without_tile": float(ref.mean())}
    if v.min() < 0.99 * ref.min():
        return CheckResult("fail", f"hfun min dropped to {v.min():.0f} m (reference {ref.min():.0f} m)", m)
    return CheckResult("pass", f"hfun min {v.min():.0f} m with the empty tile, {ref.min():.0f} m without", m)


@check("hfun.nodata_margin_seam_exact",
       "exact method: a higher-priority tile's nodata margin must not override valid "
       "lower-priority data (tile-edge strips in the NJ exact size function)", issue="seam")
def nodata_margin_seam_exact(tmp: Path) -> CheckResult:
    from pyproj import CRS
    from ocsmesh import Hfun, utils

    entries = synthetic.nj_like_stack(tmp / "dems")
    aoi = synthetic.nj_like_domain()
    paths = [e["path"] for e in entries]
    out = {}
    for method in ("exact", "fast"):
        h = Hfun(paths, hmin=HMIN, hmax=HMAX, nprocs=1, base_shape=aoi,
                 base_shape_crs="EPSG:4326", method=method)
        h.execution_mode = "serial"
        add_depth_bands(h)
        md = h.meshdata()
        if md is None:
            continue
        utils.reproject(md, CRS.from_epsg(4326))
        c, v = md.coords, np.asarray(md.values).ravel()
        # Inside the fine tiles (x > 0.30 the depth is shallower than -15 m -> 1000 m band),
        # away from the AOI edge; nothing there should differ from 1000 m.
        region = box(0.30, 0.20, 0.70, 0.45).intersection(aoi.buffer(-0.02))
        inside = np.array([region.contains(p) for p in map(lambda xy: __import__("shapely").geometry.Point(*xy), c)])
        wrong = int(np.sum(v[inside] != 1000.0))
        out[method] = {"nodes_in_region": int(inside.sum()), "wrong_nodes": wrong}
    if not is_manager():
        return CheckResult("skip", "worker rank")
    ex = out["exact"]
    detail = (f"exact: {ex['wrong_nodes']} of {ex['nodes_in_region']} nodes off the 1000 m band; "
              f"fast: {out['fast']['wrong_nodes']} of {out['fast']['nodes_in_region']}")
    return CheckResult("fail" if ex["wrong_nodes"] else "pass", detail, out)
