"""Antimeridian / 0-360 longitude handling (handoff fixes 2, 3, 4).

These encode the Alaska convention (inputs in 0-360). They stay red until
the antimeridian design decision is implemented, which is intended.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import box

from . import CheckResult, check, is_manager
from ..data import synthetic


@check("utils.estimate_bounds_utm_east_of_180",
       "estimate_bounds_utm must return a metric CRS for geographic bounds with lon > 180 "
       "(it returned None, so sizes were computed in degrees)", issue="fix 4")
def estimate_bounds_utm_east_of_180(tmp: Path) -> CheckResult:
    from ocsmesh import utils
    crs = utils.estimate_bounds_utm((190.0, 55.0, 200.0, 60.0), "EPSG:4326")
    if crs is None:
        return CheckResult("fail", "returned None for bounds (190, 55, 200, 60)")
    if getattr(crs, "is_geographic", False):
        return CheckResult("fail", f"returned a geographic CRS: {crs}")
    return CheckResult("pass", f"returned {crs.to_string()[:60]}")


@check("constraint.region_constraint_0_360",
       "RegionConstraint defined in 0-360 longitudes applies east of 180 "
       "(points reprojected by PROJ come back in -180..180)", issue="fix 2")
def region_constraint_0_360(tmp: Path) -> CheckResult:
    from pyproj import CRS, Transformer
    from ocsmesh.features.constraint import RegionConstraint

    region = box(185.0, 55.0, 195.0, 60.0)
    lons = np.linspace(-174.0, -166.0, 9)      # 186..194 E in the 0-360 convention
    lats = np.full_like(lons, 57.0)
    tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3338), always_xy=True)
    x, y = tr.transform(lons, lats)
    locations = np.column_stack([x, y])
    old = np.full((len(lons), 1), 3000.0)
    ref = np.full((len(lons), 1), -50.0)
    c = RegionConstraint(value=1500.0, shape=region, crs="EPSG:4326", value_type="max", rate=None)
    new = c.apply(ref, old, locations=locations, crs=CRS.from_epsg(3338)).ravel()
    n_applied = int(np.sum(new == 1500.0))
    if n_applied == len(lons):
        return CheckResult("pass", f"constraint applied at all {len(lons)} points east of 180")
    return CheckResult("fail", f"constraint applied at {n_applied} of {len(lons)} points east of 180")


@check("hfun.dem_east_of_180_metric_crs",
       "a DEM stored with 0-360 longitudes (east of 180) yields a size function in a metric "
       "CRS covering the DEM (fast method)", issue="fix 3/4")
def dem_east_of_180_metric_crs(tmp: Path) -> CheckResult:
    from pyproj import CRS
    from ocsmesh import Hfun

    tile = synthetic.write_tile(tmp / "ak.tif", 0.01, (190.0, 55.0, 192.0, 56.5),
                                depth_fn=lambda x, y: np.full(x.shape, -100.0, dtype="float32"))
    h = Hfun([str(tile)], hmin=2000.0, hmax=10000.0, nprocs=1, method="fast")
    h.execution_mode = "serial"
    h.add_constant_value(4000.0)
    try:
        md = h.meshdata()
    except AssertionError:
        return CheckResult("fail", "fast method asserts because estimate_bounds_utm returned None east of 180")
    if not is_manager():
        return CheckResult("skip", "worker rank")
    n = len(md.coords)
    crs = CRS.from_user_input(md.crs) if md.crs is not None else None
    if n == 0:
        return CheckResult("fail", "empty size function")
    if crs is None or crs.is_geographic:
        return CheckResult("fail", f"{n} nodes but CRS is {crs} (sizes in degrees)", {"nodes": n})
    return CheckResult("pass", f"{n} nodes in {crs.to_string()[:50]}", {"nodes": n})
