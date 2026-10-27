"""Fast tests of the harness itself (no meshing)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from shapely.geometry import Point, box

from ocsmesh_test import report, slurm
from ocsmesh_test.config import Config, load_config
from ocsmesh_test.data import manifest, ncei, synthetic

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_config_expands_env_and_resolves_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_DIR", str(tmp_path))
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("name: demo\nresults_root: ${DEMO_DIR}/res\ndems: {dir: dems}\n"
                        "tests: [{id: 0, name: a}]\nmodes: {serial: {nprocs: 1}, mpi_x: {ntasks: 3, cpus_per_task: 2}}\n")
    cfg = load_config(cfg_path)
    assert cfg.results_root == tmp_path / "res"
    assert cfg.resolve("dems.dir") == tmp_path / "dems"
    assert cfg.ocsmesh_mode("mpi_x") == "mpi" and cfg.nprocs("mpi_x") == 2
    assert cfg.ocsmesh_mode("serial") == "serial"
    assert cfg.case_dir(0, "serial", "exact") == tmp_path / "res" / "test0_a" / "serial" / "exact"


def test_shipped_configs_load():
    for p in CONFIGS.glob("*.yaml"):
        cfg = load_config(p)
        assert cfg.tests and cfg.methods and cfg.modes


def test_ncei_tile_name_parsing_and_selection():
    assert ncei.parse_sw_corner("ncei19_n39x50_w074x75_2018v2.tif") == (39.5, -74.75)
    assert ncei.parse_sw_corner("ncei13_n40x25_e010x00_2014v1.tif") == (40.25, 10.0)
    assert ncei.parse_sw_corner("readme.txt") is None
    names = ["ncei19_n39x50_w074x75_2018v2.tif",   # touches the domain
             "ncei19_n40x25_w073x75_2018v2.tif"]   # inside the bbox but outside the circle
    domain = Point(-74.5, 39.6).buffer(0.2)
    assert ncei.select_tiles(names, domain, 0.25, "polygon") == names[:1]
    assert ncei.select_tiles(names, box(-75, 39, -73, 40.5), 0.25, "bbox") == names


def test_synthetic_manifest_validation_flags_empty_tiles(tmp_path):
    entries = synthetic.nj_like_stack(tmp_path / "dems", scale=4.0)
    empty = synthetic.write_tile(tmp_path / "dems" / "empty.tif", 0.016, (0.70, 0.0, 0.9, 0.3), all_nodata=True)
    entries.append({"name": "empty", "source": "x", "path": str(empty), "url": "", "available": True,
                    "bounds": [0.7, 0.0, 0.9, 0.3], "priority": 5})
    domain = synthetic.nj_like_domain()
    manifest.annotate_validity(entries, domain)
    by = {e["name"]: e for e in entries}
    assert by["gebco/bg"]["valid_fraction"] > 0.5      # circle inside its bbox: ~pi/4 of the clip is valid
    assert by["empty"]["valid_fraction"] == 0.0
    assert by["empty"]["intersects_polygon"] is False
    paths = manifest.raster_paths(entries, domain)
    assert str(empty) not in paths and len(paths) == 5


def test_slurm_render_has_resources(tmp_path):
    cfg = load_config(CONFIGS / "nj_gaea.yaml")
    txt = slurm.render(cfg, 2, "mpi_multinode", "exact")
    assert "#SBATCH --nodes=2" in txt and "#SBATCH --ntasks=19" in txt
    assert "#SBATCH --ntasks-per-node=10" in txt
    assert "ocsmesh-test run" in txt and "--test 2 --mode mpi_multinode --method exact" in txt
    assert "mpi4py OK" in txt
    txt = slurm.render(cfg, 0, "parallel", "fast")
    assert "#SBATCH --cpus-per-task=40" in txt and "mpi4py" not in txt


def _row(test_id, mode, method, nodes, mean, meshdata=10.0):
    return {"test_id": test_id, "test_name": "t", "mode": mode, "method": method, "status": "success",
            "total_time_s": meshdata + 5, "meshdata_time_s": meshdata, "hfun": {"nodes": nodes, "mean": mean,
            "min": 50, "max": 3000}, "mesh": {"nodes": nodes * 2, "elements": nodes * 4},
            "phase_times_s": {"apply_features": {"seconds": 1.0, "calls": 1}}}


def test_report_flags_mismatch_and_speedup():
    cfg = load_config(CONFIGS / "local_synthetic.yaml")
    rows = [_row(1, "serial", "exact", 1000, 500.0, 40.0), _row(1, "parallel", "exact", 1300, 520.0, 10.0),
            _row(1, "serial", "fast", 900, 480.0, 2.0)]
    md = report.render(cfg, rows)
    assert "DIFFERS: hfun nodes, hfun mean, mesh nodes" in md
    assert "4.00x" in md                     # meshdata speedup serial->parallel
    assert "## fast vs exact" in md
    assert "## Phase breakdown" in md


def test_report_collect_reads_summaries(tmp_path):
    cfg = Config(raw={"name": "x", "results_root": str(tmp_path), "tests": [{"id": 0, "name": "a"}],
                      "modes": {"serial": {"nprocs": 1}}, "methods": ["exact"]}, path=tmp_path / "c.yaml")
    d = cfg.case_dir(0, "serial", "exact")
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(_row(0, "serial", "exact", 10, 1.0)))
    assert len(report.collect(cfg)) == 1
