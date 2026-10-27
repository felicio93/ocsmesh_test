"""GEBCO background: crop a locally available GEBCO grid to the domain.

GEBCO has no stable programmatic subset service (the NCEI WCS endpoint
rejects it), so the user downloads the global or regional GeoTIFF / NetCDF
once from https://download.gebco.net/ and points ``dems.gebco.file`` at
it. This module crops it to the domain bounding box plus a buffer and
writes a GeoTIFF next to the other DEMs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.windows import from_bounds

_logger = logging.getLogger(__name__)


def crop_gebco(src_file: Path, dest: Path, bounds, buffer_deg: float = 0.5,
               nodata: Optional[float] = None) -> Dict:
    """Crop ``src_file`` to ``bounds`` (+buffer) and write ``dest``.

    Returns a manifest entry. ``bounds`` is (west, south, east, north) in
    EPSG:4326. Domains crossing the antimeridian are not handled.
    """
    west, south, east, north = bounds
    west, south, east, north = west - buffer_deg, south - buffer_deg, east + buffer_deg, north + buffer_deg
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_file) as src:
        if src.crs is None:
            raise ValueError(f"{src_file} has no CRS; GEBCO files should be EPSG:4326")
        b = src.bounds
        west, east = max(west, b.left), min(east, b.right)
        south, north = max(south, b.bottom), min(north, b.top)
        if west >= east or south >= north:
            raise ValueError(f"domain {bounds} does not intersect {src_file} ({b})")
        win = from_bounds(west, south, east, north, src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win)
        transform = src.window_transform(win)
        nd = src.nodata if nodata is None else nodata
        if nd is None:
            nd = -99999.0
        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "width": data.shape[1], "height": data.shape[0],
            "crs": src.crs, "transform": transform, "nodata": nd,
            "compress": "deflate",
        }
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)
        out_bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1], transform)
    _logger.info("GEBCO cropped to %s (%dx%d px)", dest.name, data.shape[1], data.shape[0])
    return {
        "name": f"gebco/{dest.stem}",
        "source": "gebco",
        "path": str(dest),
        "url": str(src_file),
        "available": True,
        "bounds": list(out_bounds),
    }
