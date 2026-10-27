"""Synthetic DEM tiles.

Used by the local smoke configuration (run the whole pipeline on a laptop)
and by the regression checks. The tiles mimic the NJ stack: a coarse
background, two medium tiles that overlap each other, and two fine tiles
with nodata margins, all describing the same idealized bathymetry so that
differences between code paths are attributable to the code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

NODATA = -99999.0


def default_bathymetry(x: np.ndarray, y: np.ndarray, coast_x: float = 0.5,
                       span: float = 0.5) -> np.ndarray:
    """Depth -40 m far west rising to +5 m east of ``coast_x`` (N-S coast)."""
    return (-40.0 + 45.0 * np.clip((x - (coast_x - span)) / span, 0.0, 1.2)).astype("float32")


def write_tile(path: Path, res: float, bounds: Sequence[float],
               depth_fn: Callable = default_bathymetry, nodata_border_px: int = 0,
               all_nodata: bool = False, crs: str = "EPSG:4326",
               nodata: float = NODATA) -> Path:
    """Write one GeoTIFF tile covering ``bounds`` (west, south, east, north)."""
    x0, y0, x1, y1 = bounds
    w = max(1, int(round((x1 - x0) / res)))
    h = max(1, int(round((y1 - y0) / res)))
    yy, xx = np.mgrid[0:h, 0:w]
    x = x0 + (xx + 0.5) * res
    y = y1 - (yy + 0.5) * res
    z = np.full((h, w), nodata, dtype="float32") if all_nodata else depth_fn(x, y).astype("float32")
    if nodata_border_px and not all_nodata:
        b = nodata_border_px
        z[:b, :] = nodata
        z[-b:, :] = nodata
        z[:, :b] = nodata
        z[:, -b:] = nodata
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1,
                       dtype="float32", crs=crs, transform=from_origin(x0, y1, res, res),
                       nodata=nodata) as dst:
        dst.write(z, 1)
    return path


def nj_like_stack(dem_dir: Path, scale: float = 1.0) -> List[Dict]:
    """The standard 5-tile synthetic stack; returns manifest entries in priority order.

    ``scale`` multiplies pixel sizes (use >1 for faster runs).
    """
    dem_dir = Path(dem_dir)
    spec = [
        # name,       res,            bounds,                   nodata border px
        ("gebco/bg",     0.004 * scale, (0.00, 0.00, 0.80, 0.60), 0),
        ("third/a",      0.001 * scale, (0.05, 0.05, 0.42, 0.55), 3),
        ("third/b",      0.001 * scale, (0.40, 0.05, 0.75, 0.55), 3),
        ("ninth/a",      0.0004 * scale, (0.30, 0.20, 0.52, 0.45), 5),
        ("ninth/b",      0.0004 * scale, (0.50, 0.20, 0.70, 0.45), 5),
    ]
    entries = []
    for i, (name, res, bounds, border) in enumerate(spec):
        path = dem_dir / f"{name}.tif"
        write_tile(path, res, bounds, nodata_border_px=border)
        entries.append({
            "name": name, "source": name.split("/")[0], "path": str(path),
            "url": "synthetic", "available": True, "bounds": list(bounds),
            "priority": i,
        })
    return entries


def nj_like_domain():
    """Circular AOI that cuts through every tile of :func:`nj_like_stack`."""
    return Point(0.42, 0.30).buffer(0.27)


def write_domain_shapefile(path: Path, polygon, crs: str = "EPSG:4326") -> Path:
    import geopandas as gpd  # local import: keeps module import light
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs=crs).to_file(path)
    return path
