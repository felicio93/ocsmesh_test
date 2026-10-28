"""Collect summary.json files and render a markdown report for the ticket."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config

_logger = logging.getLogger(__name__)

EQUIV_TOL = 0.005  # relative tolerance for serial-vs-distributed agreement


def collect(cfg: Config) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(cfg.results_root.glob("test*/*/*/summary.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            _logger.warning("unreadable %s", p)
    return rows


def _fmt(x, nd=1):
    if x is None:
        return ""
    return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"


def _speedup(base, cur):
    if not base or not cur:
        return ""
    return f"{base / cur:.2f}x"


def _index(rows):
    return {(r["test_id"], r["mode"], r["method"]): r for r in rows}


def render(cfg: Config, rows: List[Dict[str, Any]]) -> str:
    # size-function-only runs are their own "method" so they never mix with full runs
    rows = [dict(r, method=f"{r['method']}-hfun") if r.get("hfun_only") else r for r in rows]
    idx = _index(rows)
    tests = [t for t in cfg.tests if any(r["test_id"] == t["id"] for r in rows)]
    modes = [m for m in cfg.modes if any(r["mode"] == m for r in rows)]
    candidates = [m for base in cfg.methods for m in (base, f"{base}-hfun")]
    methods = [m for m in candidates if any(r["method"] == m for r in rows)]
    env = next((r.get("env", {}) for r in rows if r.get("env")), {})
    out: List[str] = []
    w = out.append

    w(f"# {cfg.name} benchmark report")
    w("")
    w(f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {len(rows)} run(s) in `{cfg.results_root}`.")
    w("")
    w("## Setup")
    w("")
    w("| item | value |")
    w("|---|---|")
    w(f"| OCSMesh | {env.get('ocsmesh_branch', '?')} @ {env.get('ocsmesh_commit', '?')} |")
    w(f"| host | {env.get('hostname', '?')} (python {env.get('python', '?')}) |")
    w(f"| hmin / hmax | {cfg.get('mesh.hmin')} / {cfg.get('mesh.hmax')} m |")
    w(f"| engine | {cfg.get('mesh.engine', 'gmsh')} ({cfg.get('mesh.bnd_representation', 'adapt')}) |")
    n_r = next((r.get("n_rasters") for r in rows if r.get("n_rasters")), "?")
    w(f"| rasters | {n_r} |")
    w(f"| modes | {', '.join(f'{m} ({_mode_desc(cfg, m, rows)})' for m in modes)} |")
    if cfg.get("report_note"):
        w("")
        w(f"> {cfg.get('report_note').strip()}")
    w("")
    w("| test | name | refinements |")
    w("|---|---|---|")
    for t in tests:
        w(f"| {t['id']} | {t['name']} | {', '.join(t.get('refinements', [])) or 'none'} |")
    w("")

    w("## Results")
    w("")
    w("| test | mode | method | status | total (s) | geom (s) | meshdata (s) | driver (s) | hfun nodes | hfun min/mean/max | mesh nodes | elements |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tests:
        for mode in modes:
            for method in methods:
                r = idx.get((t["id"], mode, method))
                if r is None:
                    continue
                h = r.get("hfun", {})
                m = r.get("mesh", {})
                w(f"| {t['id']} | {mode} | {method} | {r.get('status')} | {_fmt(r.get('total_time_s'))} | "
                  f"{_fmt(r.get('geom_time_s'))} | {_fmt(r.get('meshdata_time_s'))} | "
                  f"{_fmt(r.get('meshdriver_time_s'))} | {_fmt(h.get('nodes'))} | "
                  f"{_fmt(h.get('min'), 0)}/{_fmt(h.get('mean'), 0)}/{_fmt(h.get('max'), 0)} | "
                  f"{_fmt(m.get('nodes'))} | {_fmt(m.get('elements'))} |")
    w("")

    if "serial" in modes:
        for method in methods:
            w(f"## Speedup vs serial, {method} (meshdata time; total in parentheses)")
            w("")
            w("| test | serial (s) | " + " | ".join(m for m in modes if m != "serial") + " |")
            w("|---|---|" + "---|" * (len(modes) - 1))
            for t in tests:
                base = idx.get((t["id"], "serial", method))
                if base is None:
                    continue
                cells = []
                for mode in modes:
                    if mode == "serial":
                        continue
                    r = idx.get((t["id"], mode, method))
                    if r is None:
                        cells.append("")
                        continue
                    cells.append(f"{_fmt(r.get('meshdata_time_s'))} s "
                                 f"{_speedup(base.get('meshdata_time_s'), r.get('meshdata_time_s'))} "
                                 f"({_speedup(base.get('total_time_s'), r.get('total_time_s'))})")
                w(f"| {t['id']} {t['name']} | {_fmt(base.get('meshdata_time_s'))} | " + " | ".join(cells) + " |")
            w("")

    for exact_m, fast_m in (("exact", "fast"), ("exact-hfun", "fast-hfun")):
        if exact_m not in methods or fast_m not in methods:
            continue
        w(f"## {fast_m} vs {exact_m} (serial if available)")
        w("")
        what = "hfun" if exact_m.endswith("-hfun") else "mesh"
        w(f"| test | exact meshdata (s) | fast meshdata (s) | fast speedup | exact {what} nodes | fast {what} nodes | diff |")
        w("|---|---|---|---|---|---|---|")
        ref_mode = "serial" if "serial" in modes else modes[0]
        for t in tests:
            e = idx.get((t["id"], ref_mode, exact_m))
            f = idx.get((t["id"], ref_mode, fast_m))
            if not e or not f:
                continue
            en = e.get(what, {}).get("nodes")
            fn = f.get(what, {}).get("nodes")
            diff = f"{100 * (fn - en) / en:+.1f}%" if en and fn else ""
            w(f"| {t['id']} {t['name']} | {_fmt(e.get('meshdata_time_s'))} | {_fmt(f.get('meshdata_time_s'))} | "
              f"{_speedup(e.get('meshdata_time_s'), f.get('meshdata_time_s'))} | {_fmt(en)} | {_fmt(fn)} | {diff} |")
        w("")

    phases = sorted({k for r in rows for k in (r.get("phase_times_s") or {})})
    if phases:
        w("## Phase breakdown (s, inclusive and nested; do not add up)")
        w("")
        w("| test | mode | method | " + " | ".join(phases) + " |")
        w("|---|---|---|" + "---|" * len(phases))
        for t in tests:
            for mode in modes:
                for method in methods:
                    r = idx.get((t["id"], mode, method))
                    if r is None or not r.get("phase_times_s"):
                        continue
                    pt = r["phase_times_s"]
                    w(f"| {t['id']} | {mode} | {method} | " +
                      " | ".join(_fmt(pt[p]["seconds"]) if p in pt else "" for p in phases) + " |")
        w("")

    if "serial" in modes:
        w("## Equivalence with serial (same test and method)")
        w("")
        w(f"A row is flagged when hfun node count, mesh node count or hfun mean differ from serial by more than {100*EQUIV_TOL:.1f}%.")
        w("")
        w("| test | method | mode | hfun nodes (serial / mode) | hfun mean (serial / mode) | mesh nodes (serial / mode) | verdict |")
        w("|---|---|---|---|---|---|---|")
        for t in tests:
            for method in methods:
                base = idx.get((t["id"], "serial", method))
                if base is None:
                    continue
                for mode in modes:
                    if mode == "serial":
                        continue
                    r = idx.get((t["id"], mode, method))
                    if r is None:
                        continue
                    flags = []
                    bh, rh = base.get("hfun", {}), r.get("hfun", {})
                    bm, rm = base.get("mesh", {}), r.get("mesh", {})
                    for label, a, b in (("hfun nodes", bh.get("nodes"), rh.get("nodes")),
                                        ("hfun mean", bh.get("mean"), rh.get("mean")),
                                        ("mesh nodes", bm.get("nodes"), rm.get("nodes"))):
                        if a and b and abs(a - b) / abs(a) > EQUIV_TOL:
                            flags.append(label)
                    verdict = "OK" if not flags else "DIFFERS: " + ", ".join(flags)
                    w(f"| {t['id']} | {method} | {mode} | {_fmt(bh.get('nodes'))} / {_fmt(rh.get('nodes'))} | "
                      f"{_fmt(bh.get('mean'))} / {_fmt(rh.get('mean'))} | {_fmt(bm.get('nodes'))} / {_fmt(rm.get('nodes'))} | {verdict} |")
        w("")

    failed = [r for r in rows if r.get("status") != "success"]
    if failed:
        w("## Failures")
        w("")
        for r in failed:
            w(f"- test {r['test_id']} {r['mode']} {r['method']}: {r.get('status')} {r.get('error', '')}")
        w("")
    return "\n".join(out)


def _mode_desc(cfg: Config, mode: str, rows=()) -> str:
    m = cfg.mode(mode)
    if cfg.is_mpi_mode(mode):
        # actual rank count from the runs (``ntasks: auto`` is resolved at submit time)
        ranks = next((r.get("env", {}).get("mpi_size") for r in rows
                      if r.get("mode") == mode and r.get("env", {}).get("mpi_size")), m.get("ntasks"))
        return f"{ranks} ranks x {m.get('cpus_per_task', 1)} cpu, {m.get('nodes', 1)} node(s)"
    return f"{m.get('nprocs', 1)} proc(s)"


def write_figures(cfg: Config, rows: List[Dict[str, Any]], out_dir: Path,
                  modes: Optional[List[str]] = None) -> List[Path]:
    """Scatter plots of hfun size values for each (test, method, mode). Requires ocsmesh."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from pyproj import CRS
    from ocsmesh import Mesh, utils

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in rows:
        if modes and r["mode"] not in modes:
            continue
        hp = r.get("hfun_path")
        if not hp or not Path(hp).exists():
            continue
        md = Mesh.open(hp, crs=r.get("hfun_crs") or "EPSG:4326").meshdata  # .2dm carries no CRS
        if md.crs is not None and not CRS.from_user_input(md.crs).is_geographic:
            utils.reproject(md, CRS.from_epsg(4326))
        c, v = md.coords, np.asarray(md.values).ravel()
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(c[:, 0], c[:, 1], c=v, s=1, cmap="plasma_r",
                        vmin=cfg.get("mesh.hmin"), vmax=cfg.get("mesh.hmax"))
        plt.colorbar(sc, ax=ax, label="size (m)")
        ax.set_aspect("equal")
        ax.set_title(f"t{r['test_id']} {r['test_name']} | {r['mode']} | {r['method']}")
        p = out_dir / f"hfun_t{r['test_id']}_{r['mode']}_{r['method']}{'-hfun' if r.get('hfun_only') else ''}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths
