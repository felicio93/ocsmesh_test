"""Build and read the DEM manifest.

The manifest is a JSON list of entries ordered by priority (lowest first,
which is the order OCSMesh's collectors expect: the last raster wins where
tiles overlap). Each entry: name, source, path, url, available, bounds,
priority, and after ``--validate``: valid_fraction (share of non-nodata
pixels inside the domain) and intersects_polygon.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import box

from ..config import Config
from . import gebco, ncei, synthetic

_logger = logging.getLogger(__name__)


def load_domain(cfg: Config):
    """Return (polygon in EPSG:4326, crs string)."""
    shp = cfg.resolve("domain.shapefile")
    gdf = gpd.read_file(shp)
    if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.union_all(), "EPSG:4326"


def manifest_path(cfg: Config) -> Path:
    return cfg.resolve("dems.manifest", f"{cfg.get('dems.dir', 'dems')}/manifest_{cfg.name}.json")


def dem_dir(cfg: Config) -> Path:
    return cfg.resolve("dems.dir", "dems")


def build(cfg: Config, dry_run: bool = False, validate: bool = False) -> List[Dict]:
    """Acquire all DEM sources listed in the config and write the manifest."""
    out_dir = dem_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict] = []

    syn = cfg.get("dems.synthetic")
    if syn:
        shp = cfg.resolve("domain.shapefile")
        if not shp.exists():
            synthetic.write_domain_shapefile(shp, synthetic.nj_like_domain())
        entries.extend(synthetic.nj_like_stack(out_dir, scale=float(syn.get("scale", 1.0))))
        domain, _ = load_domain(cfg)
    else:
        domain, _ = load_domain(cfg)
        buffer_deg = float(cfg.get("domain.buffer_deg", 0.0))
        selection = cfg.get("dems.tile_selection", "polygon")

        gfile = None
        if isinstance(cfg.get("dems.gebco"), dict):
            gfile = cfg.require("dems.gebco.file")     # names the missing env var
            if not dry_run and not Path(gfile).exists():
                raise FileNotFoundError(f"GEBCO file not found: {gfile}")
        if gfile:
            dest = out_dir / "gebco" / f"gebco_{cfg.name}.tif"
            if dry_run:
                entries.append({"name": f"gebco/{dest.stem}", "source": "gebco", "path": str(dest),
                                "url": gfile, "available": dest.exists(), "bounds": None})
            else:
                entries.append(gebco.crop_gebco(Path(gfile), dest, domain.bounds,
                                                buffer_deg=float(cfg.get("dems.gebco.buffer_deg", 0.5))))
        for src in cfg.get("dems.ncei", []) or []:
            entries.extend(ncei.fetch_dataset(src, domain, out_dir, selection, buffer_deg, dry_run))

    for i, e in enumerate(entries):
        e["priority"] = i

    if validate and not dry_run:
        annotate_validity(entries, domain)

    if not dry_run:
        mp = manifest_path(cfg)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(entries, indent=2))
        _logger.info("manifest written: %s (%d entries)", mp, len(entries))
    return entries


def annotate_validity(entries: List[Dict], domain) -> None:
    """Add ``intersects_polygon`` and ``valid_fraction`` (inside the AOI) to each entry.

    ``valid_fraction`` is the share of non-nodata pixels after clipping to
    the AOI, exactly what OCSMesh sees after ``Raster.clip``. 0.0 marks the
    tiles that caused the uniform-hmin bug (N1).
    """
    for e in entries:
        p = Path(e["path"])
        if not p.exists():
            continue
        with rasterio.open(p) as src:
            fp = box(*src.bounds)
            e["intersects_polygon"] = bool(fp.intersects(domain))
            try:
                data, _ = rasterio.mask.mask(src, [domain], crop=True, filled=False)
                e["valid_fraction"] = float(1.0 - np.mean(np.ma.getmaskarray(data)))
            except ValueError:
                e["valid_fraction"] = 0.0


def load(cfg: Config) -> List[Dict]:
    mp = manifest_path(cfg)
    if not mp.exists():
        raise FileNotFoundError(f"manifest not found: {mp} (run `ocsmesh-test data` first)")
    data = json.loads(mp.read_text())
    if isinstance(data, dict):  # tolerate the old name-keyed layout
        data = sorted(data.values(), key=lambda e: e["priority"])
    return data


def raster_paths(entries: List[Dict], domain=None, require_polygon: bool = True) -> List[str]:
    """Paths in priority order.

    Drops unavailable tiles, tiles that ``--validate`` found to have no
    valid data inside the AOI, and (``require_polygon``) tiles whose real
    file bounds miss the AOI polygon.
    """
    paths = []
    for e in sorted(entries, key=lambda e: e["priority"]):
        if not e.get("available"):
            _logger.warning("skipping unavailable %s", e["name"])
            continue
        p = Path(e["path"])
        if not p.exists():
            _logger.warning("missing file %s", p)
            continue
        if e.get("valid_fraction") == 0.0:
            _logger.info("skipping tile with no valid data inside the AOI: %s", e["name"])
            continue
        if require_polygon and domain is not None:
            with rasterio.open(p) as src:
                if not box(*src.bounds).intersects(domain):
                    _logger.info("skipping tile outside the AOI polygon: %s", e["name"])
                    continue
        paths.append(str(p))
    return paths


def count_usable(cfg: Config) -> int:
    """Number of rasters a run will pass to OCSMesh (after all filters)."""
    domain, _ = load_domain(cfg)
    return len(raster_paths(load(cfg), domain,
                            require_polygon=bool(cfg.get("dems.require_polygon", True))))


def summary_table(entries: List[Dict]) -> str:
    """Markdown table of the manifest."""
    rows = ["| # | name | source | available | intersects AOI | valid % |", "|---|---|---|---|---|---|"]
    for e in entries:
        vf = e.get("valid_fraction")
        rows.append(
            f"| {e['priority']} | {e['name']} | {e['source']} | {'yes' if e.get('available') else 'no'} | "
            f"{e.get('intersects_polygon', '?')} | {'' if vf is None else f'{100*vf:.1f}'} |")
    return "\n".join(rows)
