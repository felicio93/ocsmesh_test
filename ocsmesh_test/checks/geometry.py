"""Geometry robustness of the Gmsh engine (handoff fix 1)."""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import Polygon
from shapely.validation import make_valid

from . import CheckResult, check, is_manager


@check("engine.gmsh_geometry_collection",
       "a boundary whose make_valid/union yields a GeometryCollection (polygon + degenerate "
       "line) must still mesh: only the areal parts go to Gmsh", issue="fix 1")
def gmsh_geometry_collection(tmp: Path) -> CheckResult:
    from ocsmesh import Geom, MeshDriver

    # square with a zero-width spike: invalid; make_valid -> GeometryCollection(Polygon, LineString)
    spike = Polygon([(0, 0), (1000, 0), (1000, 1000), (500, 1000), (500, 1500), (500, 1000), (0, 1000)])
    fixed = make_valid(spike)
    if fixed.geom_type != "GeometryCollection":
        return CheckResult("error", f"test geometry did not produce a GeometryCollection ({fixed.geom_type})")
    geom = Geom(spike, crs="EPSG:32618")
    try:
        mesh = MeshDriver(geom, None, engine_name="gmsh", bnd_representation="fixed").run()
    except Exception as exc:  # noqa: BLE001
        return CheckResult("fail", f"MeshDriver raised {type(exc).__name__}: {str(exc)[:80]}")
    if not is_manager():
        return CheckResult("skip", "worker rank")
    n = len(mesh.meshdata.coords)
    return CheckResult("pass" if n > 0 else "fail", f"meshed {n} nodes", {"nodes": n})
