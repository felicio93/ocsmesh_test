"""DEM variants: resampled and/or split copies of a benchmark's DEMs.

A variant is defined in the config under ``dems.variants``::

    dems:
      variants:
        res_12m:  {resample_m: 12.5, resampling: average}
        split_2x2: {split: 2}
        split_4x4_ov2km: {split: 4, overlap_m: 2000}

``resample_m``
    Target pixel size in metres (converted to degrees at each tile's
    latitude for geographic DEMs). Tiles already at that size or coarser
    are kept as they are; DEMs are never upsampled.
``resampling``
    How pixels are combined: any ``rasterio`` resampling name, e.g.
    ``average`` (default), ``min`` (deepest value: keeps narrow channels
    open), ``max``, ``bilinear``, ``nearest``.
``split``
    Split each tile into ``split x split`` sub-tiles of equal pixel count.
    Sub-tiles without valid data inside the domain are dropped.
``overlap_m``
    Extra margin around each sub-tile, in metres. OCSMesh composites
    overlapping tiles by priority, so this is a naive overlap (the
    sub-tile that wins an overlap keeps its own edge effects), not the
    padded sub-tiles of a future OCSMesh option.
``sources``
    Manifest sources to transform (default: every source except gebco).
    Other tiles are used as they are.

Resampling is applied before splitting. A variant gets its own DEM folder,
manifest and results folder (see ``config.load_config(variant=...)``), so
its runs can be scored against the original ones with ``compare``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

from ..config import Config

_logger = logging.getLogger(__name__)

_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON_EQ = 111_320.0


def _target_res(src, res_m: float):
    """Pixel size (x, y) in the tile's units for ``res_m`` metres."""
    if src.crs is not None and src.crs.is_geographic:
        lat = 0.5 * (src.bounds.bottom + src.bounds.top)
        return (res_m / (_M_PER_DEG_LON_EQ * math.cos(math.radians(lat))),
                res_m / _M_PER_DEG_LAT)
    return res_m, res_m


def _profile(src, **updates):
    profile = src.profile.copy()
    profile.update(driver="GTiff", compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    profile.update(updates)
    return profile


def resample_tile(path: Path, out: Path, res_m: float, method: str) -> Path:
    """Write ``path`` resampled to ``res_m`` metres; returns the file to use."""
    with rasterio.open(path) as src:
        rx, ry = _target_res(src, res_m)
        if rx <= abs(src.transform.a) * 1.0001 and ry <= abs(src.transform.e) * 1.0001:
            _logger.info("%s already at or coarser than %.1f m; kept", path.name, res_m)
            return path
        width = max(1, round((src.bounds.right - src.bounds.left) / rx))
        height = max(1, round((src.bounds.top - src.bounds.bottom) / ry))
        transform = rasterio.transform.from_bounds(*src.bounds, width, height)
        nodata = src.nodata if src.nodata is not None else -99999.0
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **_profile(src, width=width, height=height,
                                                 transform=transform, nodata=nodata)) as dst:
            for band in range(1, src.count + 1):
                data = np.full((height, width), nodata, dtype=src.dtypes[band - 1])
                reproject(rasterio.band(src, band), data,
                          src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                          dst_transform=transform, dst_crs=src.crs, dst_nodata=nodata,
                          resampling=Resampling[method])
                dst.write(data, band)
    return out


def split_tile(path: Path, out_dir: Path, n: int, overlap_m: float) -> List[Path]:
    """Write ``n x n`` sub-tiles of ``path`` (each grown by ``overlap_m``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    with rasterio.open(path) as src:
        ox_m, oy_m = _target_res(src, overlap_m) if overlap_m else (0.0, 0.0)
        pad_x = math.ceil(ox_m / abs(src.transform.a)) if overlap_m else 0
        pad_y = math.ceil(oy_m / abs(src.transform.e)) if overlap_m else 0
        cols = np.linspace(0, src.width, n + 1).round().astype(int)
        rows = np.linspace(0, src.height, n + 1).round().astype(int)
        for i in range(n):
            for j in range(n):
                c0, c1 = max(cols[j] - pad_x, 0), min(cols[j + 1] + pad_x, src.width)
                r0, r1 = max(rows[i] - pad_y, 0), min(rows[i + 1] + pad_y, src.height)
                win = Window(c0, r0, c1 - c0, r1 - r0)
                data = src.read(window=win)
                if src.nodata is not None and np.all(data == src.nodata):
                    continue
                out = out_dir / f"{path.stem}_r{i}c{j}.tif"
                with rasterio.open(out, "w", **_profile(
                        src, width=win.width, height=win.height,
                        transform=src.window_transform(win))) as dst:
                    dst.write(data)
                outs.append(out)
    return outs


def variant_spec(cfg: Config, name: str) -> Dict[str, Any]:
    variants = cfg.get("dems.variants", {}) or {}
    if name not in variants:
        raise KeyError(f"unknown DEM variant {name!r}; defined: {sorted(variants)}")
    return dict(variants[name] or {})


def build(cfg: Config, base_cfg: Config, name: str) -> List[Dict]:
    """Write the variant's DEMs and manifest from the base config's manifest.

    ``cfg`` is the config loaded with the variant (its manifest path and DEM
    folder); ``base_cfg`` the same config without it (the original tiles).
    """
    from . import manifest as manifest_mod

    spec = variant_spec(base_cfg, name)
    res_m = spec.get("resample_m")
    method = spec.get("resampling", "average")
    n_split = int(spec.get("split", 1) or 1)
    overlap_m = float(spec.get("overlap_m", 0.0) or 0.0)
    sources = spec.get("sources")
    if method not in Resampling.__members__:
        raise ValueError(f"unknown resampling {method!r}; use one of {sorted(Resampling.__members__)}")

    out_root = manifest_mod.manifest_path(cfg).parent
    out_root.mkdir(parents=True, exist_ok=True)
    domain, _ = manifest_mod.load_domain(base_cfg)

    entries: List[Dict] = []
    for e in sorted(manifest_mod.load(base_cfg), key=lambda e: e["priority"]):
        p = Path(e["path"])
        transform_it = (e["source"] in sources) if sources else (e["source"] != "gebco")
        if not transform_it or not e.get("available") or not p.exists() \
                or e.get("valid_fraction") == 0.0:
            entries.append(dict(e))
            continue
        src_dir = out_root / e["source"]
        if res_m:
            p = resample_tile(p, src_dir / "resampled" / p.name, float(res_m), method)
        parts = split_tile(p, src_dir / "split", n_split, overlap_m) if n_split > 1 else [p]
        for k, part in enumerate(parts):
            with rasterio.open(part) as src:
                bounds = list(src.bounds)
            suffix = f"/{part.stem}" if n_split > 1 else ""
            entries.append({**e, "name": f"{e['name']}{suffix}", "path": str(part),
                            "bounds": bounds, "variant": name, "parent": e["name"]})
        _logger.info("%s -> %d tile(s)", e["name"], len(parts))

    for i, e in enumerate(entries):   # same priority order, sub-tiles in place of their tile
        e["priority"] = i
    manifest_mod.annotate_validity(entries, domain)
    manifest_mod.manifest_path(cfg).write_text(json.dumps(entries, indent=2))
    (out_root / "variant.json").write_text(json.dumps({"name": name, **spec}, indent=2))
    return entries
