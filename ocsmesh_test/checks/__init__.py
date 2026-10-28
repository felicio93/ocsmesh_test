"""Regression checks for OCSMesh fixes, on synthetic data.

Each check is a function returning a :class:`CheckResult`. Checks run
against whatever OCSMesh is installed, so on an unfixed checkout they fail
and on the fixed branch they pass; the markdown table is what goes in the
ticket.

MPI note: under an MPI launcher *every* rank runs *every* check in
lockstep, because constructing an ``HfunCollector`` is collective (it
broadcasts the work directory). Checks make all OCSMesh calls on all
ranks and only compare results on the manager (rank 0); worker ranks
report ``skip``.
"""

from __future__ import annotations

import logging
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

STATUS_ORDER = {"fail": 0, "error": 1, "pass": 2, "info": 3, "skip": 4}


@dataclass
class CheckResult:
    status: str                      # pass | fail | error | info | skip
    detail: str = ""
    metrics: Dict[str, object] = field(default_factory=dict)


@dataclass
class Check:
    name: str
    description: str
    fn: Callable[[Path], CheckResult]
    issue: str = ""                  # handoff id: N1, N3, fix 5, ...
    requires_mpi: bool = False
    status: str = ""
    detail: str = ""
    seconds: float = 0.0
    metrics: Dict[str, object] = field(default_factory=dict)


REGISTRY: List[Check] = []


def check(name: str, description: str, issue: str = "", requires_mpi: bool = False):
    def deco(fn):
        REGISTRY.append(Check(name=name, description=description, fn=fn, issue=issue,
                              requires_mpi=requires_mpi))
        return fn
    return deco


def mpi_state():
    """(active, rank, size) without importing mpi4py before ocsmesh."""
    import ocsmesh  # noqa: F401
    from ocsmesh.mpi import _is_mpi_active, _is_mpi_env_detected
    if _is_mpi_env_detected() and _is_mpi_active():
        from mpi4py import MPI
        return True, MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    return False, 0, 1


def is_manager() -> bool:
    return mpi_state()[1] == 0


def _load_modules() -> None:
    from . import hfun_nodata, hfun_distributed, mesh_interp, geometry, dateline, utils_misc  # noqa: F401


def _start_watchdog(name: str):
    """Abort every rank if a check runs longer than OCSMESH_CHECK_TIMEOUT seconds.

    A check that raises on one rank before a collective call leaves the
    other ranks blocked in that collective; without this the job would sit
    until walltime.
    """
    import os
    import sys
    import threading

    timeout = float(os.environ.get("OCSMESH_CHECK_TIMEOUT", "1800"))

    def _abort():
        from mpi4py import MPI
        sys.stderr.write(f"[ocsmesh-test] check {name} exceeded {timeout:.0f} s; aborting all ranks\n")
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(2)

    timer = threading.Timer(timeout, _abort)
    timer.daemon = True
    timer.start()
    return timer


def run(only: Optional[List[str]] = None, skip_mpi: bool = False) -> List[Check]:
    _load_modules()
    active, rank, size = mpi_state()
    results = []
    for c in REGISTRY:
        if only and not any(o in c.name for o in only):
            continue
        if c.requires_mpi and (skip_mpi or not active):
            c.status, c.detail = "skip", "needs an MPI launcher with >1 rank"
            results.append(c)
            continue
        t0 = time.perf_counter()
        watchdog = _start_watchdog(c.name) if active else None
        with tempfile.TemporaryDirectory(prefix=f"chk_{c.name.replace('.', '_')}_") as tmp:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = c.fn(Path(tmp))
                c.status, c.detail, c.metrics = res.status, res.detail, res.metrics
            except Exception as exc:  # noqa: BLE001
                c.status = "error"
                c.detail = f"{type(exc).__name__}: {exc}"
                _logger.debug(traceback.format_exc())
        if watchdog is not None:
            watchdog.cancel()
        c.seconds = time.perf_counter() - t0
        if rank == 0:
            _logger.info("%-45s %-5s %.1fs  %s", c.name, c.status.upper(), c.seconds, c.detail[:100])
        results.append(c)
    return results


def render(results: List[Check]) -> str:
    import ocsmesh
    out = ["# OCSMesh regression checks", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} against `{Path(ocsmesh.__file__).resolve().parent}`.",
           "", "| status | check | issue | what it verifies | detail | time (s) |", "|---|---|---|---|---|---|"]
    for c in sorted(results, key=lambda c: (STATUS_ORDER.get(c.status, 9), c.name)):
        out.append(f"| **{c.status.upper()}** | `{c.name}` | {c.issue} | {c.description} | "
                   f"{c.detail.replace('|', '/')} | {c.seconds:.1f} |")
    counts = {}
    for c in results:
        counts[c.status] = counts.get(c.status, 0) + 1
    out += ["", "Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + "."]
    return "\n".join(out)
