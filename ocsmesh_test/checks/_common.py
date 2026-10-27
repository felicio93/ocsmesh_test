"""Helpers shared by the checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import box

from ..data import synthetic

HMIN, HMAX = 50.0, 3000.0
BANDS = [(3000.0, None, -30.0), (2500.0, -30.0, -25.0), (2000.0, -25.0, -20.0),
         (1500.0, -20.0, -15.0), (1000.0, -15.0, None)]


def add_depth_bands(hfun) -> None:
    for value, lo, hi in BANDS:
        kw = {"value": value}
        if lo is not None:
            kw["lower_bound"] = lo
        if hi is not None:
            kw["upper_bound"] = hi
        hfun.add_constant_value(**kw)


def expected_band(depth: np.ndarray) -> np.ndarray:
    """Size the five bands assign to a depth (nodata treated as 'no information')."""
    out = np.full(depth.shape, HMAX, dtype=float)
    for value, lo, hi in BANDS:
        m = np.ones(depth.shape, dtype=bool)
        if lo is not None:
            m &= depth > lo
        if hi is not None:
            m &= depth < hi
        out[m] = np.minimum(out[m], value)
    return out


def half_clipped_tile(tmp: Path):
    """One 100x100 tile and an AOI covering only its eastern half (clip shifts the origin)."""
    path = synthetic.write_tile(tmp / "A.tif", 0.002, (0.0, 0.0, 0.2, 0.2),
                                depth_fn=lambda x, y: synthetic.default_bathymetry(x, y, coast_x=0.2, span=0.2))
    return [str(path)], box(0.1, 0.0, 0.2, 0.2)


def coast_with_inlet(x, y, coast_x: float = 0.5, inlet_y: float = 0.3, half_width: float = 0.001):
    """Bathymetry with a ~220 m wide inlet cutting into the land, for channel detection."""
    z = synthetic.default_bathymetry(x, y, coast_x=coast_x, span=0.1)
    z[(x > coast_x - 0.02) & (np.abs(y - inlet_y) < half_width)] = -3.0
    return z


def hfun_values(h):
    """Raw hfun raster array of the first raster size function in a collector."""
    return h._hfun_list[0].get_values().copy()  # noqa: SLF001
