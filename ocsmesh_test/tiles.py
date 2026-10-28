"""Per-tile channel detection timing (handoff N2).

Channel detection runs serially on the coordinator, one raster at a time,
and took ~24,500 s of NJ t4. Whether distributing it per raster can scale
depends on how that time is split between tiles. This module reproduces
what ``HfunCollector`` does for each tile (clip to the AOI, then
``Raster.get_channels``) and times it, one tile per process, for one or
more simplification tolerances.
"""

from __future__ import annotations

import csv
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import Config

_logger = logging.getLogger(__name__)


def _time_tile(path: str, label: str, aoi_wkt: str, level: float, width: float,
               tolerance: Optional[float]) -> Dict:
    """Worker: clip one tile to the AOI and time channel detection."""
    import ocsmesh  # noqa: F401  (configures the environment before anything else)
    from pyproj import Transformer
    from shapely import ops, wkt
    from ocsmesh import Raster

    row = {"tile": label, "tolerance": tolerance}
    try:
        raster = Raster(path)
        aoi = wkt.loads(aoi_wkt)
        if not raster.crs.equals("EPSG:4326"):
            tr = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
            aoi = ops.transform(tr.transform, aoi)
        t0 = time.perf_counter()
        raster.clip(aoi)
        row["clip_s"] = round(time.perf_counter() - t0, 2)
        row["pixels"] = int(raster.src.width * raster.src.height)
        t0 = time.perf_counter()
        channels = raster.get_channels(level=level, width=width, tolerance=tolerance)
        row["detect_s"] = round(time.perf_counter() - t0, 2)
        if channels is None or channels.is_empty:
            row["n_channels"], row["empty"] = 0, True
        else:
            row["n_channels"] = len(getattr(channels, "geoms", [channels]))
            row["empty"] = False
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["status"] = f"error: {type(exc).__name__}: {exc}"[:200]
    return row


def channel_timing(cfg: Config, tolerances: Sequence[Optional[float]], workers: int,
                   level: Optional[float] = None, width: Optional[float] = None) -> List[Dict]:
    from .data import manifest

    ch = (cfg.get("refinements.channels") or [{}])[0]
    level = float(ch.get("level", 0.0)) if level is None else level
    width = float(ch.get("width", 1000.0)) if width is None else width
    domain, _ = manifest.load_domain(cfg)
    entries = manifest.load(cfg)
    paths = manifest.raster_paths(entries, domain,
                                  require_polygon=bool(cfg.get("dems.require_polygon", True)))
    labels = {str(Path(e["path"])): e["name"] for e in entries}
    jobs = [(p, labels.get(str(Path(p)), Path(p).name), domain.wkt, level, width, tol)
            for tol in tolerances for p in paths]
    _logger.info("timing channel detection: %d tiles x %d tolerance(s), %d workers, width %.0f m",
                 len(paths), len(tolerances), workers, width)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_time_tile, *j) for j in jobs]
        for fut in as_completed(futures):
            row = fut.result()
            row.update({"level": level, "width": width})
            _logger.info("%-45s tol=%-5s %s s %s", row["tile"], row["tolerance"],
                         row.get("detect_s"), row["status"])
            rows.append(row)
    rows.sort(key=lambda r: (str(r["tolerance"]), -(r.get("detect_s") or 0)))
    return rows


def render(rows: List[Dict], title: str) -> str:
    out = [f"# {title}", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}. One tile per process; "
           "`detect_s` is `Raster.get_channels` after clipping to the AOI (what the collector does "
           "serially on rank 0).", ""]
    for tol in sorted({str(r["tolerance"]) for r in rows}):
        sub = [r for r in rows if str(r["tolerance"]) == tol]
        ok = [r for r in sub if r["status"] == "ok"]
        total = sum(r["detect_s"] for r in ok)
        biggest = max((r["detect_s"] for r in ok), default=0.0)
        out += [f"## tolerance = {tol}", "",
                f"Serial total (what one coordinator spends): **{total:,.1f} s**. "
                f"Slowest single tile (lower bound with one tile per worker): **{biggest:,.1f} s**. "
                f"Best possible speedup from per-tile distribution: "
                f"**{(total / biggest if biggest else 0):.1f}x**.", "",
                "| tile | pixels | clip (s) | detect (s) | share of total | channels | status |",
                "|---|---|---|---|---|---|---|"]
        for r in sub:
            share = f"{100 * r['detect_s'] / total:.1f}%" if r["status"] == "ok" and total else ""
            out.append(f"| {r['tile']} | {r.get('pixels', ''):,} | {r.get('clip_s', '')} | "
                       f"{r.get('detect_s', '')} | {share} | {r.get('n_channels', '')} | {r['status']} |"
                       if r["status"] == "ok" else
                       f"| {r['tile']} |  |  |  |  |  | {r['status']} |")
        out.append("")
    return "\n".join(out)


def write_csv(rows: List[Dict], path: Path) -> None:
    keys = ["tile", "tolerance", "level", "width", "pixels", "clip_s", "detect_s",
            "n_channels", "empty", "status"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
