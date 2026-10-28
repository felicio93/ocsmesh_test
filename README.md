# ocsmesh_test

Benchmark and regression-check harness for the [OCSMesh](https://github.com/noaa-ocs-modeling/OCSMesh)
MPI implementation. One YAML config describes a benchmark (domain, DEM stack, mesh recipe, test
matrix, execution modes, cluster); the `ocsmesh-test` command takes it from data download to a
markdown report. A separate `check` command runs regression checks for the OCSMesh bugs found
during the MPI validation, on synthetic data, so a fixed branch can be verified on a laptop or
under `srun` and the result table pasted into the tracking ticket.

Nothing here modifies OCSMesh; everything goes through its public API.

## Install

```bash
conda activate <env with ocsmesh and gmsh>   # mpi4py built against the site MPI for MPI modes
pip install -e .
ocsmesh-test --help
```

## Quick start on a laptop (synthetic stack, ~minutes)

```bash
ocsmesh-test data   -c configs/local_synthetic.yaml --validate
ocsmesh-test run    -c configs/local_synthetic.yaml --test 1 --mode serial   --method exact
ocsmesh-test run    -c configs/local_synthetic.yaml --test 1 --mode parallel --method exact
ocsmesh-test report -c configs/local_synthetic.yaml --figures
ocsmesh-test check                                   # regression checks -> reports/checks_<date>.md
```

## NJ benchmark on Gaea C6

```bash
export GEBCO_FILE=/path/to/gebco_2024.tif            # from https://download.gebco.net/ (GeoTIFF or NetCDF)
export OCSMESH_TEST_EMAIL=you@noaa.gov               # optional, SLURM mail
ocsmesh-test data   -c configs/nj_gaea.yaml --validate   # NCEI tiles are scraped + downloaded, GEBCO cropped
ocsmesh-test slurm  -c configs/nj_gaea.yaml              # writes slurm/nj/*.sh (60 scripts)
ocsmesh-test slurm  -c configs/nj_gaea.yaml --submit --test 0 1 --mode serial mpi_no_pool
ocsmesh-test report -c configs/nj_gaea.yaml --figures    # reports/nj_report_<date>.md
srun -n 6 ocsmesh-test check                             # MPI checks too
```

Clone the repo on the shared scratch filesystem (`/gpfs/f6/...` on Gaea): results and the
per-run `TMPDIR` live under it and must be visible to every node. Run on the login node only
the `data`, `slurm` and `report` steps; `run` is what the SLURM scripts execute. MPI modes
with `ntasks: auto` get one worker rank per usable tile plus the manager, counted from the
manifest when the scripts are written. `--validate` writes the valid-data fraction of each tile inside the AOI into the
manifest, which is how the all-nodata tiles that flattened the NJ size function are detected.
Tiles are selected for download by the AOI *polygon* (`dems.tile_selection: polygon`; set
`bbox` to reproduce the old behaviour on purpose), using footprints derived from the tile
names, which encode the tile's **north-west** corner. At run time the real file bounds and the
valid-data fraction decide which rasters reach OCSMesh.

## What the report contains

- setup (OCSMesh branch and commit, host, hmin/hmax, engine, modes),
- the full results table (status, total / geom / meshdata / driver seconds, hfun and mesh stats),
- speedup vs serial per method, fast vs exact,
- per-phase breakdown (contour/channel detection vs dispatch, composite, ...),
- an equivalence table: every distributed run compared with serial (hfun nodes, hfun mean, mesh
  nodes; flagged above 0.5%),
- failures, and optional hfun figures (`--figures`).

## Regression checks (`ocsmesh-test check`)

| check | issue | verifies |
|---|---|---|
| `hfun.empty_feature_keeps_values` | N1 | an empty feature/patch no longer zeroes the size function |
| `hfun.all_nodata_tile_with_channels` | N1 | an all-nodata clipped tile does not flatten hfun to hmin |
| `hfun.nodata_margin_seam_exact` | seam | a tile's nodata margin does not override lower-priority data (exact) |
| `hfun.clipped_tile_serial_vs_parallel` | N3 | parallel workers use the clipped DEM as topo |
| `hfun.clipped_tile_serial_vs_mpi` | N3 | same through MPIExecutor (needs `srun`/`mpiexec`) |
| `hfun.synthetic_recipe_serial_vs_mpi` | N3 | full recipe serial vs MPI within 0.5% (needs MPI) |
| `hfun.channel_detection_parallel` | N2 | channel detection distributed; results identical |
| `mesh.interpolate_spline_nodata` | fix 5 | spline interpolation respects raster nodata |
| `engine.gmsh_geometry_collection` | fix 1 | GeometryCollection boundaries still mesh |
| `utils.estimate_bounds_utm_east_of_180` | fix 4 | metric CRS east of 180 |
| `constraint.region_constraint_0_360` | fix 2 | region constraints in 0-360 longitudes |
| `hfun.dem_east_of_180_metric_crs` | fix 3/4 | DEM east of 180 gives a metric size function |
| `utils.clip_elevation` | feature | `utils.clip_elevation(mesh, min, max)` |

For MPI checks on more than one node set `OCSMESH_SHARED_TMPDIR` to a shared directory. A
check that runs longer than `OCSMESH_CHECK_TIMEOUT` seconds (default 1800) aborts all ranks
instead of hanging. Checks are expected to fail on an unfixed OCSMesh; the table they produce documents the state of
the branch. Under an MPI launcher every rank runs every check (constructing an `HfunCollector` is
collective); only rank 0 reports.

## Config reference (short)

| key | meaning |
|---|---|
| `domain.shapefile` | AOI polygon (any CRS, reprojected to EPSG:4326) |
| `dems.gebco.file`, `dems.ncei[]` | GEBCO file to crop; NCEI datasets/subfolders to scrape and download |
| `dems.synthetic` | generate the 5-tile synthetic stack instead |
| `mesh.*` | hmin, hmax, geom_zmin/zmax, engine, bnd_representation |
| `refinements.*` | named blocks: `depth_bands`, `contour`, `flow_limiters`, `channels`, `region_constraints` |
| `tests[]` | id, name, description, list of refinement blocks |
| `methods` | `fast`, `exact` |
| `modes.*` | `serial`/`parallel` (`nprocs`), `mpi_*` (`ntasks`, `cpus_per_task`, `nodes`) |
| `cluster.*` | account, partition, qos, clusters, walltime, modules, env, conda, launcher |
| `report_note` | text shown under the setup table of the report |

`${VAR}` and `~` are expanded when the config is loaded, except in `cluster.env` and
`cluster.modules`, which are written verbatim into the SLURM scripts and expanded on the
compute node. A value referring to an unset variable becomes empty: optional settings are
dropped, required ones (the GEBCO file) fail with a message naming the variable.

## Layout

```
ocsmesh_test/        package: config, data/{ncei,gebco,synthetic,manifest}, recipe, runner, timing, slurm, report, checks/
configs/             nj_gaea.yaml, local_synthetic.yaml
domain/              NJ AOI shapefile
tests/               pytest for the harness (no meshing)
```

Results, logs, DEMs and SLURM scripts are written under the repo (gitignored).
