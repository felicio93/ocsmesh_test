"""Run one benchmark case (test, mode, method) and write summary.json.

Under MPI every rank calls this; only rank 0 writes files. A failure on
the manager writes a failed summary and aborts all ranks so a job never
hangs until walltime.
"""

from __future__ import annotations

import json
import logging
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import ocsmesh  # noqa: F401  must precede any mpi4py import
from ocsmesh import Mesh, MeshDriver
from ocsmesh.mpi import _is_mpi_active, _is_mpi_env_detected

from . import timing
from .config import Config
from .data import manifest as manifest_mod
from .recipe import build_geom, build_hfun

_logger = logging.getLogger(__name__)

_MPI_ACTIVE = _is_mpi_env_detected() and _is_mpi_active()
_RANK, _SIZE = 0, 1
_MPI = None
if _MPI_ACTIVE:
    try:
        from mpi4py import MPI as _MPI
        _RANK = _MPI.COMM_WORLD.Get_rank()
        _SIZE = _MPI.COMM_WORLD.Get_size()
    except ImportError:
        _MPI_ACTIVE = False
IS_MANAGER = _RANK == 0


def abort(code: int = 1) -> None:
    """Terminate every rank (plain exit when not under MPI)."""
    logging.shutdown()
    if _MPI_ACTIVE and _SIZE > 1 and _MPI is not None:
        _MPI.COMM_WORLD.Abort(code)
    sys.exit(code)


def environment_info() -> Dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "ocsmesh_file": str(Path(ocsmesh.__file__).resolve()),
        "ocsmesh_version": getattr(ocsmesh, "__version__", None),
        "mpi_size": _SIZE,
    }
    try:
        repo = Path(ocsmesh.__file__).resolve().parents[1]
        info["ocsmesh_commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        info["ocsmesh_branch"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        pass
    return info


def _stats(meshdata) -> Dict[str, Any]:
    vals = np.asarray(meshdata.values, dtype=float).ravel()
    return {
        "nodes": int(len(meshdata.coords)),
        "elements": int(0 if meshdata.tria is None else len(meshdata.tria)),
        "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)),
        "mean": float(np.nanmean(vals)),
    }


def run_case(cfg: Config, test_id: int, mode: str, method: str,
             out_dir: Optional[Path] = None, hfun_only: bool = False) -> Dict[str, Any]:
    test = cfg.test(test_id)
    out_dir = Path(out_dir) if out_dir else cfg.case_dir(test_id, mode, method)
    if IS_MANAGER:
        out_dir.mkdir(parents=True, exist_ok=True)
    nprocs = cfg.nprocs(mode)
    ocs_mode = cfg.ocsmesh_mode(mode)
    timing.install()
    timing.reset()

    summary: Dict[str, Any] = {
        "benchmark": cfg.name, "test_id": int(test["id"]), "test_name": test["name"],
        "mode": mode, "method": method, "ocsmesh_mode": ocs_mode, "nprocs": nprocs,
        "hfun_only": hfun_only, "out_dir": str(out_dir), "env": environment_info(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _logger.info("=== %s test %s (%s) mode=%s method=%s nprocs=%d ranks=%d",
                 cfg.name, test["id"], test["name"], mode, method, nprocs, _SIZE)
    t_total = time.perf_counter()
    try:
        entries = manifest_mod.load(cfg)
        domain, domain_crs = manifest_mod.load_domain(cfg)
        paths = manifest_mod.raster_paths(
            entries, domain, require_polygon=bool(cfg.get("dems.require_polygon", True)))
        summary["n_rasters"] = len(paths)
        summary["rasters"] = [Path(p).name for p in paths]

        geom = None
        if IS_MANAGER and not hfun_only:
            t0 = time.perf_counter()
            geom = build_geom(cfg, paths, domain, domain_crs, nprocs)
            summary["geom_time_s"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        hfun = build_hfun(cfg, paths, domain, domain_crs, nprocs, ocs_mode, method, test)
        summary["hfun_build_time_s"] = round(time.perf_counter() - t0, 2)
        summary["execution_mode_effective"] = hfun.execution_mode

        t0 = time.perf_counter()
        meshdata = hfun.meshdata()
        summary["meshdata_time_s"] = round(time.perf_counter() - t0, 2)

        if meshdata is None:  # MPI worker rank
            summary["status"] = "worker_rank"
            summary["total_time_s"] = round(time.perf_counter() - t_total, 2)
            return summary

        summary["hfun"] = _stats(meshdata)
        summary["hfun_crs"] = meshdata.crs.to_string() if meshdata.crs is not None else None
        hfun_path = out_dir / f"hfun_t{test['id']}_{test['name']}_{mode}_{method}.2dm"
        Mesh(meshdata).write(str(hfun_path), overwrite=True, format="2dm")
        summary["hfun_path"] = str(hfun_path)

        if not hfun_only:
            t0 = time.perf_counter()
            driver = MeshDriver(geom, hfun, engine_name=cfg.get("mesh.engine", "gmsh"),
                                bnd_representation=cfg.get("mesh.bnd_representation", "adapt"))
            final_mesh = driver.run()
            summary["meshdriver_time_s"] = round(time.perf_counter() - t0, 2)
            mesh_path = out_dir / f"mesh_t{test['id']}_{test['name']}_{mode}_{method}.2dm"
            final_mesh.write(str(mesh_path), format="2dm", overwrite=True)
            summary["mesh_path"] = str(mesh_path)
            md = final_mesh.meshdata
            summary["mesh"] = {"nodes": int(len(md.coords)),
                               "elements": int(0 if md.tria is None else len(md.tria))}
        summary["status"] = "success"
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        summary["failed_rank"] = _RANK
        _logger.error("FAILED: %s\n%s", exc, summary["traceback"])

    summary["total_time_s"] = round(time.perf_counter() - t_total, 2)
    summary["phase_times_s"] = timing.snapshot()
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if IS_MANAGER:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        _logger.info("total %.1f s, status=%s", summary["total_time_s"], summary["status"])
    if summary["status"] == "failed":
        abort(1)
    return summary
