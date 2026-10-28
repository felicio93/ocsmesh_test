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
    """Same tiles with and without nodata margins must give the same size function.

    The region checked (x >= 0.32, inside the AOI) is entirely in the 1000 m
    band and covered by valid lower-priority data under every margin, so any
    node off 1000 m there is caused by the margins. The no-margin run is the
    control: if it is not clean, the check reports an error, not a failure.
    """
    from pyproj import CRS
    from shapely import contains_xy
    from ocsmesh import Hfun, utils

    aoi = synthetic.nj_like_domain()
    region = box(0.32, 0.06, 0.70, 0.54).intersection(aoi.buffer(-0.02))
    wrong = {}
    for margins in (False, True):
        entries = synthetic.nj_like_stack(tmp / f"dems_{margins}", margins=margins)
        h = Hfun([e["path"] for e in entries], hmin=HMIN, hmax=HMAX, nprocs=1, base_shape=aoi,
                 base_shape_crs="EPSG:4326", method="exact")
        h.execution_mode = "serial"
        add_depth_bands(h)
        md = h.meshdata()
        if md is None:
            continue
        utils.reproject(md, CRS.from_epsg(4326))
        c, v = md.coords, np.asarray(md.values).ravel()
        inside = contains_xy(region, c[:, 0], c[:, 1])
        wrong[margins] = (int(np.sum(np.abs(v[inside] - 1000.0) > 1.0)), int(inside.sum()))
    if not is_manager():
        return CheckResult("skip", "worker rank")
    m = {"control_off_band": wrong[False][0], "margins_off_band": wrong[True][0],
         "nodes_in_region": wrong[True][1]}
    if wrong[False][0]:
        return CheckResult("error", f"control (no margins) not clean: {wrong[False][0]} nodes off band", m)
    detail = f"{wrong[True][0]} of {wrong[True][1]} nodes off the 1000 m band with nodata margins (0 without)"
    return CheckResult("fail" if wrong[True][0] else "pass", detail, m)
