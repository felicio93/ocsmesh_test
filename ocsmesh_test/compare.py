"""Score runs against a reference run: size function, mesh, detected shapes, time.

Used to judge DEM resampling and splitting: the reference is the same test
on the original DEMs (``exact``, unsplit), the other runs are variants.
Each run is a case directory holding ``summary.json`` (written by ``run``).

Scores
------
size function
    Both size functions are interpolated (linearly, on their own
    triangulations) on a regular grid over the reference's extent. The
    relative difference ``|h - h_ref| / h_ref`` is summarized over the
    *refined area*, where either size is below ``hmax``: median, 95th
    percentile, maximum and the share of that area differing by more than
    10%. Over the whole domain most of the area sits at ``hmax`` in both,
    which would hide the differences. Where the size jumps between levels
    (bands, channels, limiters) an edge that moves by one triangle flips a
    whole strip, so a *shift-tolerant* share is also given: a point counts
    only if its size is more than 10% outside the range the other size
    function takes within ``SHIFT_M`` (rounded up to whole grid cells).
mesh
    Node and element counts against the reference.
channels
    Reference and variant channel polygons in UTM: intersection over union,
    and the reference area missed and the area added (km^2).
contours
    Distances between the reference and variant lines, both ways, from
    points every ``contour_step`` metres along them: median, 95th
    percentile and maximum (m).
time
    ``meshdata`` time and its speedup against the reference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_logger = logging.getLogger(__name__)

REL_TOL = 0.10          # "differs" threshold for the size function
SHIFT_M = 250.0         # edge shift tolerated by the shift-tolerant score (m)


def load_case(case_dir: Path) -> Dict[str, Any]:
    case_dir = Path(case_dir)
    summary = json.loads((case_dir / "summary.json").read_text())
    summary["_dir"] = str(case_dir)
    return summary


def _meshdata(path: str, crs: Optional[str]):
    from ocsmesh import Mesh
    return Mesh.open(path, crs=crs or "EPSG:4326").meshdata   # .2dm carries no CRS


def _interpolator(md):
    """Linear interpolator of a size function on its own triangles.

    Falls back to a Delaunay triangulation of the nodes when the mesh's
    triangles overlap (matplotlib: "Triangulation is invalid"), as some
    ``fast`` size-function meshes do. Returns masked arrays either way.
    """
    import matplotlib.tri as mtri
    xy = np.asarray(md.coords, dtype=float)
    vals = np.asarray(md.values, dtype=float).ravel()
    try:
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1], np.asarray(md.tria))
        interp = mtri.LinearTriInterpolator(tri, vals)
        interp(xy[:1, 0], xy[:1, 1])     # builds the trifinder: raises on invalid meshes
        return interp
    except RuntimeError as exc:
        _logger.warning("size-function mesh not usable as is (%s); using a Delaunay "
                        "triangulation of its nodes", exc)
    from scipy.interpolate import LinearNDInterpolator
    uniq, idx = np.unique(xy, axis=0, return_index=True)
    lin = LinearNDInterpolator(uniq, vals[idx])

    def interp(gx, gy):
        out = lin(gx, gy)
        return np.ma.masked_invalid(out)
    return interp


def _shift_tolerant_error(h_a, h_b, k):
    """Relative distance of ``h_b`` from the range of ``h_a`` within ``k`` cells.

    Zero where ``h_b`` matches some value of ``h_a`` close by, so an edge
    between two size levels that moved by up to ``k`` cells does not count.
    NaN cells (outside either mesh) are ignored.
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    size = 2 * k + 1
    lo = minimum_filter(np.where(np.isnan(h_a), np.inf, h_a), size=size, mode="nearest")
    hi = maximum_filter(np.where(np.isnan(h_a), -np.inf, h_a), size=size, mode="nearest")
    with np.errstate(invalid="ignore"):
        err = np.maximum.reduce([np.zeros_like(h_b), h_b - hi, lo - h_b]) / h_a
    return err


def compare_hfun(ref: Dict[str, Any], other: Dict[str, Any], grid: int = 1000,
                 shift_m: float = SHIFT_M) -> Dict[str, Any]:
    md_r = _meshdata(ref["hfun_path"], ref.get("hfun_crs"))
    md_o = _meshdata(other["hfun_path"], other.get("hfun_crs"))
    xmin, ymin = md_r.coords.min(axis=0)
    xmax, ymax = md_r.coords.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, grid), np.linspace(ymin, ymax, grid))
    H_r = np.ma.filled(np.ma.masked_invalid(_interpolator(md_r)(gx, gy)).astype(float), np.nan)
    H_o = np.ma.filled(np.ma.masked_invalid(_interpolator(md_o)(gx, gy)).astype(float), np.nan)
    valid = ~(np.isnan(H_r) | np.isnan(H_o))
    h_r, h_o = H_r[valid], H_o[valid]
    hmax = float(np.max(np.asarray(md_r.values)))
    refined = np.minimum(h_r, h_o) < 0.999 * hmax
    # grid cell size in metres (geographic size functions: degrees at mid latitude)
    dx, dy = (xmax - xmin) / (grid - 1), (ymax - ymin) / (grid - 1)
    if abs(xmax) <= 360 and abs(ymax) <= 90:
        lat = np.radians(0.5 * (ymin + ymax))
        dx, dy = dx * 111_320.0 * np.cos(lat), dy * 110_540.0
    cell_m = float(max(dx, dy))
    k = max(1, int(np.ceil(shift_m / cell_m)))
    out: Dict[str, Any] = {"grid_points": int(valid.sum()),
                           "refined_share": float(refined.mean()) if valid.any() else 0.0,
                           "cell_m": cell_m, "shift_cells": k}
    if refined.any():
        rel = np.abs(h_o[refined] - h_r[refined]) / h_r[refined]
        # both ways: variant vs reference neighbourhood, reference vs variant neighbourhood
        e1 = _shift_tolerant_error(H_r, H_o, k)[valid][refined]
        e2 = (_shift_tolerant_error(H_o, H_r, k)[valid][refined] * h_o[refined] / h_r[refined])
        shift = np.nan_to_num(np.maximum(e1, e2), nan=0.0, posinf=0.0)
        out.update({"rel_median": float(np.median(rel)),
                    "rel_p95": float(np.percentile(rel, 95)),
                    "rel_max": float(rel.max()),
                    "share_over_tol": float((rel > REL_TOL).mean()),
                    "shift_p95": float(np.percentile(shift, 95)),
                    "shift_share_over_tol": float((shift > REL_TOL).mean()),
                    "mean_ref": float(h_r[refined].mean()),
                    "mean_other": float(h_o[refined].mean())})
    return out


def _utm(gdf):
    return gdf.estimate_utm_crs()


def _read_shapes(case: Dict[str, Any], kind: str):
    import geopandas as gpd
    path = (case.get("shape_paths") or {}).get(kind)
    if not path or not Path(path).exists():
        return None
    return gpd.read_file(path)


def compare_channels(ref: Dict[str, Any], other: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    a = _read_shapes(ref, "channels")
    b = _read_shapes(other, "channels")
    if a is None:
        return None
    crs = _utm(a)
    ua = _clean_union(a.to_crs(crs))
    ub = _clean_union(b.to_crs(crs)) if b is not None and len(b) else None
    area_a = ua.area
    if ub is None or ub.is_empty:
        return {"ref_km2": area_a / 1e6, "iou": 0.0, "missed_km2": area_a / 1e6, "added_km2": 0.0}
    try:
        inter = ua.intersection(ub).area
        union = ua.union(ub).area
    except Exception:  # noqa: BLE001 - GEOS topology errors on near-coincident edges
        import shapely
        ua, ub = shapely.set_precision(ua, 0.01), shapely.set_precision(ub, 0.01)
        inter = ua.intersection(ub).area
        union = ua.union(ub).area
    return {"ref_km2": area_a / 1e6,
            "iou": inter / union if union else 1.0,
            "missed_km2": max(area_a - inter, 0.0) / 1e6,
            "added_km2": max(ub.area - inter, 0.0) / 1e6}


def _clean_union(gdf):
    """Union of the polygons after repairing invalid ones (1 cm grid if needed)."""
    import shapely
    geoms = shapely.make_valid(gdf.geometry.values)
    polys = [g for g in shapely.get_parts(geoms) if g.geom_type in ("Polygon", "MultiPolygon")]
    if not polys:
        return shapely.Polygon()
    try:
        return shapely.union_all(polys)
    except Exception:  # noqa: BLE001
        return shapely.union_all(shapely.set_precision(polys, 0.01))


def _line_points(gdf, crs, step: float) -> np.ndarray:
    import shapely
    lines = gdf.to_crs(crs).explode(index_parts=False)
    lines = lines[lines.geom_type.isin(["LineString", "LinearRing"]) & ~lines.is_empty]
    if not len(lines):
        return np.empty((0, 2))
    dense = shapely.segmentize(lines.geometry.values, step)
    return shapely.get_coordinates(dense)


def compare_contours(ref: Dict[str, Any], other: Dict[str, Any],
                     step: float = 25.0) -> Optional[Dict[str, Any]]:
    from scipy.spatial import cKDTree
    a = _read_shapes(ref, "contours")
    b = _read_shapes(other, "contours")
    if a is None:
        return None
    crs = _utm(a)
    pa = _line_points(a, crs, step)
    pb = _line_points(b, crs, step) if b is not None else np.empty((0, 2))
    if not len(pa) or not len(pb):
        return {"ref_points": int(len(pa)), "other_points": int(len(pb))}
    d_ab, _ = cKDTree(pb).query(pa)     # reference -> variant (missed detail)
    d_ba, _ = cKDTree(pa).query(pb)     # variant -> reference (spurious lines)
    d = np.concatenate([d_ab, d_ba])
    return {"ref_points": int(len(pa)), "other_points": int(len(pb)),
            "dist_median_m": float(np.median(d)), "dist_p95_m": float(np.percentile(d, 95)),
            "dist_max_m": float(d.max())}


def compare_case(ref: Dict[str, Any], other: Dict[str, Any], grid: int = 1000,
                 time_ref: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Score ``other`` against ``ref``; the speedup is against ``time_ref`` if given."""
    out: Dict[str, Any] = {"case": other["_dir"], "status": other.get("status")}
    t_r = (time_ref or ref).get("meshdata_time_s")
    t_o = other.get("meshdata_time_s")
    out["meshdata_s"] = t_o
    out["speedup"] = (t_r / t_o) if t_r and t_o else None
    m_r, m_o = ref.get("mesh") or {}, other.get("mesh") or {}
    if m_r.get("nodes") and m_o.get("nodes"):
        out["mesh_nodes"] = m_o["nodes"]
        out["mesh_nodes_diff"] = (m_o["nodes"] - m_r["nodes"]) / m_r["nodes"]
    if other.get("status") != "success":
        return out
    try:
        out["hfun"] = compare_hfun(ref, other, grid=grid)
    except Exception as exc:  # noqa: BLE001
        out["hfun_error"] = repr(exc)
    for kind, fn in (("channels", compare_channels), ("contours", compare_contours)):
        try:
            res = fn(ref, other)
            if res is not None:
                out[kind] = res
        except Exception as exc:  # noqa: BLE001
            out[f"{kind}_error"] = repr(exc)
    return out


def _f(x, fmt="{:.1f}"):
    return "" if x is None else fmt.format(x)


def render(ref: Dict[str, Any], results: List[Dict[str, Any]], labels: List[str],
           time_ref: Optional[Dict[str, Any]] = None) -> str:
    lines = [f"# Comparison against `{ref['_dir']}`", "",
             f"Reference: test {ref.get('test_id')} {ref.get('test_name')}, {ref.get('mode')} "
             f"{ref.get('method')}, meshdata {_f(ref.get('meshdata_time_s'))} s, "
             f"mesh nodes {(ref.get('mesh') or {}).get('nodes', '')}.", ""]
    if time_ref is not None:
        lines += [f"Speedup is against `{time_ref['_dir']}` ({time_ref.get('mode')}, meshdata "
                  f"{_f(time_ref.get('meshdata_time_s'))} s); the scores are against the reference.", ""]
    lines += [
             "Size function: relative difference over the refined area (either size below hmax); "
             f"`>{100*REL_TOL:.0f}%` is the share of that area differing by more.", "",
             f"`shifted >{100*REL_TOL:.0f}%` is the same share once an edge may move by about "
             f"{SHIFT_M:.0f} m (rounded up to grid cells; the size is compared with the range of the other size function "
             "around each point), so it counts real changes of size, not edges that moved slightly.", "",
             "| case | status | meshdata (s) | speedup | mesh nodes (diff) | hfun median | hfun p95 | hfun max | "
             f"hfun >{100*REL_TOL:.0f}% | shifted >{100*REL_TOL:.0f}% | channels IoU | missed / added (km²) | "
             "contour p50 / p95 / max (m) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, r in zip(labels, results):
        h = r.get("hfun") or {}
        ch = r.get("channels") or {}
        co = r.get("contours") or {}
        nodes = (f"{r['mesh_nodes']:,} ({100*r['mesh_nodes_diff']:+.1f}%)"
                 if r.get("mesh_nodes") else "")
        lines.append(
            f"| {label} | {r.get('status')} | {_f(r.get('meshdata_s'))} | {_f(r.get('speedup'), '{:.2f}x')} | "
            f"{nodes} | {_f(h.get('rel_median'), '{:.1%}')} | {_f(h.get('rel_p95'), '{:.1%}')} | "
            f"{_f(h.get('rel_max'), '{:.0%}')} | {_f(h.get('share_over_tol'), '{:.1%}')} | "
            f"{_f(h.get('shift_share_over_tol'), '{:.1%}')} | "
            f"{_f(ch.get('iou'), '{:.3f}')} | "
            f"{_f(ch.get('missed_km2'), '{:.2f}')}{' / ' if ch else ''}{_f(ch.get('added_km2'), '{:.2f}')} | "
            f"{_f(co.get('dist_median_m'))}{' / ' if 'dist_median_m' in co else ''}"
            f"{_f(co.get('dist_p95_m'))}{' / ' if 'dist_median_m' in co else ''}{_f(co.get('dist_max_m'))} |")
        errs = [f"{k}: {v}" for k, v in r.items() if k.endswith("_error")]
        if errs:
            lines.append(f"| ↳ {label} errors | " + "; ".join(errs) + " |" + " |" * 11)
    lines.append("")
    return "\n".join(lines)
