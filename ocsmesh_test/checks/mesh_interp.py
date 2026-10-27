"""Mesh.interpolate nodata handling (handoff fix 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from . import CheckResult, check, is_manager


def _write(path: Path, arr: np.ndarray, nodata, res=0.01):
    import rasterio
    from rasterio.transform import from_origin
    h, w = arr.shape
    kw = {} if nodata is None else {"nodata": nodata}
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(0.0, 1.0, res, res), **kw) as d:
        d.write(arr.astype("float32"), 1)
    return path


@check("mesh.interpolate_spline_nodata",
       "Mesh.interpolate(method='spline') must treat raster nodata as missing so nodes in a "
       "hole fall back to the next raster, like linear/nearest do", issue="fix 5")
def interpolate_spline_nodata(tmp: Path) -> CheckResult:
    from pyproj import CRS
    from ocsmesh import Mesh, Raster
    from ocsmesh.internal import MeshData

    res = 0.01
    xs = (np.arange(100) + 0.5) * res
    bg_val = -5.0 - 195.0 * xs                     # background varies with x
    bg = _write(tmp / "bg.tif", np.tile(bg_val, (100, 1)), -99999.0)
    hi_arr = np.full((100, 100), -50.0)
    hi_arr[40:60, 40:60] = -99999.0                # hole = declared nodata
    hi = _write(tmp / "hi.tif", hi_arr, -99999.0)

    g = np.linspace(0.05, 0.95, 19)
    xx, yy = np.meshgrid(g, g)
    pts = np.c_[xx.ravel(), yy.ravel()]
    hole = (pts[:, 0] > 0.42) & (pts[:, 0] < 0.58) & (pts[:, 1] > 0.42) & (pts[:, 1] < 0.58)
    expected = -5.0 - 195.0 * pts[hole, 0]

    results, bad = {}, []
    for method in ("spline", "linear", "nearest"):
        md = MeshData(coords=pts.copy(), tria=Delaunay(pts).simplices,
                      values=np.zeros((len(pts), 1)), crs=CRS.from_epsg(4326))
        mesh = Mesh(md)
        mesh.interpolate([Raster(str(bg)), Raster(str(hi))], method=method, nprocs=1)
        v = mesh.meshdata.values.ravel()[hole]
        ok = bool(np.allclose(v, expected, atol=2.0))
        results[method] = {"ok": ok, "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
        if not ok:
            bad.append(f"{method}: hole nodes {np.nanmin(v):.0f}..{np.nanmax(v):.0f} (expected {expected.min():.0f}..{expected.max():.0f})")
    if not is_manager():
        return CheckResult("skip", "worker rank")
    if bad:
        return CheckResult("fail", "; ".join(bad), results)
    return CheckResult("pass", "hole nodes take the background value for spline, linear and nearest", results)
