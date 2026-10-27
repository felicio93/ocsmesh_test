"""Small utilities requested alongside the fixes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from . import CheckResult, check


@check("utils.clip_elevation",
       "utils.clip_elevation(mesh, min_elevation, max_elevation) clamps node elevations "
       "(e.g. max=10 turns every value above 10 into 10, min=-6000 caps depth at 6000 m)",
       issue="feature")
def clip_elevation(tmp: Path) -> CheckResult:
    from pyproj import CRS
    from ocsmesh import Mesh, utils
    from ocsmesh.internal import MeshData

    fn = getattr(utils, "clip_elevation", None)
    if fn is None:
        return CheckResult("fail", "ocsmesh.utils.clip_elevation does not exist")
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
    vals = np.array([-7000.0, -100.0, 5.0, 20.0, 10.0]).reshape(-1, 1)
    mesh = Mesh(MeshData(coords=pts, tria=Delaunay(pts).simplices, values=vals.copy(), crs=CRS.from_epsg(4326)))
    out = fn(mesh, -6000.0, 10.0)
    got = np.asarray((out if out is not None else mesh).meshdata.values).ravel()
    want = np.array([-6000.0, -100.0, 5.0, 10.0, 10.0])
    if np.allclose(got, want):
        return CheckResult("pass", "values clamped to [-6000, 10]")
    return CheckResult("fail", f"got {got.tolist()}, expected {want.tolist()}")
