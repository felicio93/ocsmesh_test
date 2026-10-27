"""Serial vs distributed equivalence of the size function (handoff N3, N2) and MPI checks."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import CheckResult, check, is_manager
from ._common import HMAX, HMIN, add_depth_bands, coast_with_inlet, half_clipped_tile, hfun_values
from ..data import synthetic


def _depth_band_hfun(paths, aoi, mode, nprocs):
    from ocsmesh import Hfun
    h = Hfun(paths, hmin=HMIN, hmax=HMAX, nprocs=nprocs, base_shape=aoi,
             base_shape_crs="EPSG:4326", method="exact")
    h.execution_mode = mode
    h.add_constant_value(1000.0, lower_bound=-30.0, upper_bound=-10.0)
    h._apply_features()  # noqa: SLF001  refinements only, no meshing
    return h


@check("hfun.clipped_tile_serial_vs_parallel",
       "with base_shape clipping, parallel workers must apply depth-based refinements at "
       "the same pixels as serial (workers used the unclipped DEM as topo)", issue="N3")
def clipped_tile_serial_vs_parallel(tmp: Path) -> CheckResult:
    paths, aoi = half_clipped_tile(tmp)
    hs = _depth_band_hfun(paths, aoi, "serial", 1)
    hp = _depth_band_hfun(paths, aoi, "parallel", 2)
    if not is_manager():
        return CheckResult("skip", "worker rank")
    s, p = hfun_values(hs), hfun_values(hp)
    m = {"cells_differ": int(np.sum(s != p)), "cells": int(s.size),
         "band_cells_serial": int(np.sum(s == 1000)), "band_cells_parallel": int(np.sum(p == 1000))}
    if m["cells_differ"]:
        return CheckResult("fail", f"{m['cells_differ']} of {m['cells']} cells differ "
                           f"(band cells serial {m['band_cells_serial']} vs parallel {m['band_cells_parallel']})", m)
    return CheckResult("pass", f"identical hfun raster ({m['cells']} cells)", m)


@check("hfun.clipped_tile_serial_vs_mpi",
       "same as the parallel check but through MPIExecutor", issue="N3", requires_mpi=True)
def clipped_tile_serial_vs_mpi(tmp: Path) -> CheckResult:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    tmp = comm.bcast(str(tmp), root=0)
    tmp = Path(tmp)
    if comm.Get_rank() == 0:
        paths, aoi = half_clipped_tile(tmp)
    else:
        paths, aoi = None, None
    paths, aoi = comm.bcast((paths, aoi), root=0)
    hs = _depth_band_hfun(paths, aoi, "serial", 1)
    hm = _depth_band_hfun(paths, aoi, "mpi", 1)
    if not is_manager():
        return CheckResult("skip", "worker rank")
    s, p = hfun_values(hs), hfun_values(hm)
    n = int(np.sum(s != p))
    return CheckResult("fail" if n else "pass", f"{n} of {s.size} cells differ", {"cells_differ": n})


@check("hfun.channel_detection_parallel",
       "channel detection runs per raster in the workers when nprocs>1 (it was serial on the "
       "coordinator and took 95% of NJ t4); results must stay identical", issue="N2")
def channel_detection_parallel(tmp: Path) -> CheckResult:
    from ocsmesh import Hfun
    from .. import timing

    paths = []
    for i, (x0, y0) in enumerate([(0.3, 0.2), (0.5, 0.2), (0.3, 0.35), (0.5, 0.35)]):
        # every tile holds an inlet so no tile yields an empty channel polygon (that is N1, not N2)
        depth_fn = (lambda yc: lambda x, y: coast_with_inlet(x, y, coast_x=x0 + 0.1, inlet_y=yc))(y0 + 0.075)
        paths.append(str(synthetic.write_tile(tmp / f"t{i}.tif", 0.0005, (x0, y0, x0 + 0.2, y0 + 0.15),
                                              depth_fn=depth_fn)))
    timing.install()
    out = {}
    for mode, nprocs in (("serial", 1), ("parallel", 4)):
        timing.reset()
        h = Hfun(paths, hmin=HMIN, hmax=HMAX, nprocs=nprocs, method="exact")
        h.execution_mode = mode
        h.add_channel(level=0.0, width=600.0, target_size=100.0, expansion_rate=0.01, tolerance=20)
        t0 = time.perf_counter()
        h._apply_features()  # noqa: SLF001
        snap = timing.snapshot()
        out[mode] = {
            "apply_s": round(time.perf_counter() - t0, 2),
            "detect_s": snap.get("channel_detect", {}).get("seconds"),
            "values": [hf.get_values() for hf in h._hfun_list],  # noqa: SLF001
        }
    if not is_manager():
        return CheckResult("skip", "worker rank")
    same = all(np.array_equal(a, b) for a, b in zip(out["serial"]["values"], out["parallel"]["values"]))
    m = {k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in out.items()}
    ds, dp = out["serial"]["detect_s"], out["parallel"]["detect_s"]
    detail = f"detection serial {ds} s vs parallel {dp} s; refined rasters {'identical' if same else 'DIFFER'}"
    if not same:
        return CheckResult("fail", detail, m)
    # Timing on small synthetic tiles is too noisy to grade; the NJ phase table
    # (channel_detect per mode) is the real measure. Report it as information.
    return CheckResult("info", detail, m)


@check("hfun.synthetic_recipe_serial_vs_mpi",
       "full synthetic stack with AOI, depth bands and a contour: MPI meshdata must match serial "
       "within 0.5% (hfun nodes and mean)", issue="N3", requires_mpi=True)
def synthetic_recipe_serial_vs_mpi(tmp: Path) -> CheckResult:
    from mpi4py import MPI
    from ocsmesh import Hfun
    comm = MPI.COMM_WORLD
    tmp = Path(comm.bcast(str(tmp), root=0))
    if comm.Get_rank() == 0:
        entries = synthetic.nj_like_stack(tmp / "dems", scale=2.0)
        paths = [e["path"] for e in entries]
    else:
        paths = None
    paths = comm.bcast(paths, root=0)
    aoi = synthetic.nj_like_domain()
    stats = {}
    for mode in ("serial", "mpi"):
        h = Hfun(paths, hmin=HMIN, hmax=HMAX, nprocs=1, base_shape=aoi, base_shape_crs="EPSG:4326", method="exact")
        h.execution_mode = mode
        add_depth_bands(h)
        h.add_contour(level=0.0, expansion_rate=0.005, target_size=300.0)
        md = h.meshdata()
        if md is not None:
            v = np.asarray(md.values).ravel()
            stats[mode] = {"nodes": int(len(md.coords)), "mean": float(v.mean()), "min": float(v.min())}
    if not is_manager():
        return CheckResult("skip", "worker rank")
    s, m = stats["serial"], stats["mpi"]
    dn = abs(s["nodes"] - m["nodes"]) / s["nodes"]
    dm = abs(s["mean"] - m["mean"]) / s["mean"]
    detail = f"nodes {s['nodes']} vs {m['nodes']} ({100*dn:.2f}%), mean {s['mean']:.1f} vs {m['mean']:.1f} ({100*dm:.2f}%)"
    return CheckResult("pass" if max(dn, dm) <= 0.005 else "fail", detail, stats)
