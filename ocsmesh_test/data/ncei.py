"""NCEI / CUDEM topobathy tile listing and download.

Tile names are scraped from the dataset index pages at runtime (nothing is
hard-coded), filtered by the domain *polygon*, and streamed from the NOAA
coastal lidar S3 bucket.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from shapely.geometry import box

_logger = logging.getLogger(__name__)

S3_BASE = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem"
INDEX_BASE = "https://coast.noaa.gov/htdata/raster2/elevation"

# NCEI/CUDEM tiles are named by their NORTH-WEST (upper-left) corner:
# ncei19_n39x50_w074x75_2018v2.tif covers 39.25-39.50 N, 74.75-74.50 W
# (checked against the GeoTIFF header of that tile).
_TIF_RE = re.compile(
    r"ncei\d+_[nN](\d+)[xX](\d+)_([wWeE])(\d+)[xX](\d+)_\d{4}v\d+\.tif",
    re.IGNORECASE,
)
_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
_INDEX_CACHE: Dict[str, List[str]] = {}


def parse_nw_corner(filename: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) of the tile's north-west corner, or None."""
    m = _TIF_RE.search(filename)
    if m is None:
        return None
    lat_deg, lat_frac, hemi, lon_deg, lon_frac = m.groups()
    lat = int(lat_deg) + int(lat_frac) / 100.0
    lon = int(lon_deg) + int(lon_frac) / 100.0
    if hemi.lower() == "w":
        lon = -lon
    return lat, lon


def tile_box(filename: str, tile_size_deg: float):
    """Shapely box of the tile footprint from its name, or None."""
    corner = parse_nw_corner(filename)
    if corner is None:
        return None
    lat, lon = corner
    return box(lon, lat - tile_size_deg, lon + tile_size_deg, lat)


def list_tiles(dataset: str, subfolder: str, timeout: int = 120) -> List[str]:
    """Scrape the bare ``*.tif`` names of one dataset subfolder."""
    url = f"{INDEX_BASE}/{dataset}/{subfolder}/index.html"
    if url in _INDEX_CACHE:
        return _INDEX_CACHE[url]
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _logger.warning("could not fetch %s: %s", url, exc)
        _INDEX_CACHE[url] = []
        return []
    names = sorted({
        href.split("/")[-1]
        for href in _HREF_RE.findall(resp.text)
        if href.endswith(".tif") and _TIF_RE.search(href.split("/")[-1])
    })
    _INDEX_CACHE[url] = names
    return names


def select_tiles(names: Iterable[str], domain, tile_size_deg: float,
                 selection: str = "polygon", buffer_deg: float = 0.0) -> List[str]:
    """Keep tiles whose footprint intersects the domain.

    ``selection='polygon'`` tests against the AOI polygon (what the mesh
    needs). ``selection='bbox'`` tests against the AOI bounding box, which
    is the old behaviour that let all-nodata tiles into the size function.
    Name-derived footprints are only used to decide what to download; the
    run-time filter (:func:`manifest.raster_paths`) uses the real file
    bounds and the valid-data fraction.
    """
    target = domain.buffer(buffer_deg) if buffer_deg else domain
    if selection == "bbox":
        target = box(*target.bounds)
    kept = []
    for name in names:
        fp = tile_box(name, tile_size_deg)
        if fp is None:
            _logger.warning("cannot parse tile name %s; keeping it", name)
            kept.append(name)
        elif fp.intersects(target):
            kept.append(name)
    return kept


def download_file(url: str, dest: Path, retries: int = 4, timeout: int = 600) -> bool:
    """Stream ``url`` to ``dest`` through a .part file. Returns success."""
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=4 << 20):
                        fh.write(chunk)
            part.rename(dest)
            return True
        except Exception as exc:  # noqa: BLE001
            _logger.warning("attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(5 * attempt)
    return False


def fetch_dataset(source: Dict, domain, dem_dir: Path, selection: str,
                  buffer_deg: float, dry_run: bool = False) -> List[Dict]:
    """Download every tile of one NCEI source that intersects the domain.

    ``source`` is one entry of ``dems.ncei`` in the config:
    ``{name, dataset, subfolders, tile_size_deg}``.
    Returns manifest entries (dicts).
    """
    entries: List[Dict] = []
    dataset = source["dataset"]
    size = float(source.get("tile_size_deg", 0.25))
    for sub in source["subfolders"]:
        names = list_tiles(dataset, sub)
        if not names:
            _logger.warning("%s/%s: no tiles listed", dataset, sub)
            continue
        chosen = select_tiles(names, domain, size, selection, buffer_deg)
        _logger.info("%s/%s: %d of %d tiles intersect the domain (%s)",
                     dataset, sub, len(chosen), len(names), selection)
        out_dir = dem_dir / source["name"] / sub
        for fname in chosen:
            url = f"{S3_BASE}/{dataset}/{sub}/{fname}"
            dest = out_dir / fname
            if dry_run:
                ok = True
            else:
                ok = download_file(url, dest)
                if not ok:
                    _logger.error("download failed: %s", url)
            fp = tile_box(fname, size)
            entries.append({
                "name": f"{source['name']}/{sub}/{Path(fname).stem}",
                "source": source["name"],
                "path": str(dest),
                "url": url,
                "available": ok and (dry_run or dest.exists()),
                "bounds": list(fp.bounds) if fp is not None else None,
            })
    return entries
