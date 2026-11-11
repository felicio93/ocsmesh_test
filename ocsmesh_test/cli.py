"""Command line interface: ocsmesh-test {data, slurm, run, report, check}."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path


def _shared_tmpdir() -> None:
    """Honor OCSMESH_SHARED_TMPDIR before ocsmesh reads TMPDIR at import."""
    shared = os.environ.get("OCSMESH_SHARED_TMPDIR")
    if shared:
        os.makedirs(shared, exist_ok=True)
        os.environ["TMPDIR"] = shared
        tempfile.tempdir = shared


def _logging(verbose: bool) -> None:
    rank = os.environ.get("PMI_RANK") or os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("SLURM_PROCID") or "0"
    logging.basicConfig(
        level=logging.DEBUG if verbose else (logging.INFO if rank == "0" else logging.WARNING),
        format=f"[rank {rank}] %(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def cmd_data(args) -> int:
    from .config import load_config
    from .data import manifest
    cfg = load_config(args.config)
    entries = manifest.build(cfg, dry_run=args.dry_run, validate=args.validate)
    print(manifest.summary_table(entries))
    bad = [e for e in entries if e.get("valid_fraction") == 0.0]
    if bad:
        print(f"\nWARNING: {len(bad)} tile(s) have no valid data inside the AOI and are excluded "
              f"by the polygon filter: {', '.join(e['name'] for e in bad)}")
    return 0


def cmd_slurm(args) -> int:
    from .config import load_config
    from . import slurm
    cfg = load_config(args.config, variant=args.variant)
    paths = slurm.generate(cfg, tests=args.test, modes=args.mode, methods=args.method,
                           hfun_only=args.hfun_only)
    for p in paths:
        print(p)
    if args.submit:
        ids = slurm.submit(paths)
        print(f"{len(ids)} job(s) submitted: {' '.join(ids)}")
    return 0


def cmd_job(args) -> int:
    """Wrap any ocsmesh-test command (after --) in a SLURM script."""
    from .config import load_config
    from . import slurm
    cfg = load_config(args.config)
    cmd = list(args.args)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("give the ocsmesh-test command after --, e.g. -- check --out reports")
    path = slurm.generate_job(cfg, args.name, cmd, ntasks=args.ntasks, cpus=args.cpus,
                              nodes=args.nodes, walltime=args.walltime)
    print(path)
    if args.submit:
        ids = slurm.submit([path])
        print(f"{len(ids)} job(s) submitted: {' '.join(ids)}")
    return 0


def cmd_tile_timing(args) -> int:
    from .config import load_config
    from . import tiles
    cfg = load_config(args.config)
    tols = [None if str(t).lower() == "none" else float(t) for t in args.tolerance]
    rows = tiles.channel_timing(cfg, tols, workers=args.workers, level=args.level, width=args.width)
    out_dir = Path(args.out) if args.out else cfg.resolve("reports_dir", "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    md = out_dir / f"{cfg.name}_channel_timing_{stamp}.md"
    md.write_text(tiles.render(rows, f"{cfg.name}: channel detection time per tile"))
    tiles.write_csv(rows, out_dir / f"{cfg.name}_channel_timing_{stamp}.csv")
    print(md)
    return 0 if all(r["status"] == "ok" for r in rows) else 1


def cmd_run(args) -> int:
    _shared_tmpdir()
    from .config import load_config
    from . import runner
    cfg = load_config(args.config, variant=args.variant)
    out = Path(args.out_dir) if args.out_dir else None
    summary = runner.run_case(cfg, args.test, args.mode, args.method, out_dir=out, hfun_only=args.hfun_only)
    if runner.IS_MANAGER:
        h = summary.get("hfun", {})
        print(f"=== status={summary['status']} total={summary.get('total_time_s')}s "
              f"meshdata={summary.get('meshdata_time_s')}s hfun min/mean/max="
              f"{h.get('min')}/{h.get('mean')}/{h.get('max')} nodes={h.get('nodes')}")
    return 0 if summary.get("status") in ("success", "worker_rank") else 1


def cmd_report(args) -> int:
    from .config import load_config
    from . import report
    cfg = load_config(args.config, variant=args.variant)
    rows = report.collect(cfg)
    if not rows:
        print(f"no summary.json under {cfg.results_root}")
        return 1
    out_dir = Path(args.out) if args.out else cfg.resolve("reports_dir", "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    md = report.render(cfg, rows)
    if args.figures:
        figs = report.write_figures(cfg, rows, out_dir / "figures", modes=args.figure_modes)
        md += "\n## Figures\n\n" + "\n".join(f"![{p.stem}](figures/{p.name})" for p in figs) + "\n"
    out = out_dir / f"{cfg.name}_report_{time.strftime('%Y%m%d')}.md"
    out.write_text(md)
    print(out)
    return 0


def cmd_variant(args) -> int:
    from .config import load_config
    from .data import manifest as manifest_mod
    from .data import variants
    base = load_config(args.config)
    cfg = load_config(args.config, variant=args.variant)
    entries = variants.build(cfg, base, args.variant)
    print(manifest_mod.summary_table(entries))
    print(f"manifest: {manifest_mod.manifest_path(cfg)}")
    print(f"usable tiles: {manifest_mod.count_usable(cfg)}")
    return 0


def _write_compare(out: Path, md: str, payload) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    print(f"written: {out} (and {out.with_suffix('.json').name})")


def _compare_matrix(args) -> int:
    """compare -c CFG --test ...: each test's variants (and fast) against serial exact."""
    from . import compare
    from .config import load_config
    base = load_config(args.config)
    tests = args.test or [t["id"] for t in base.get("tests", [])]
    variants = args.variants or sorted(base.get("dems.variants", {}) or {})
    sections, payload = [], {}
    for tid in tests:
        ref_dir = base.case_dir(tid, args.ref_mode, args.ref_method)
        if not (ref_dir / "summary.json").exists():
            print(f"test {tid}: no reference at {ref_dir}; skipped")
            continue
        ref = compare.load_case(ref_dir)
        found = []
        if args.with_fast:
            found.append(("original fast", base.case_dir(tid, args.ref_mode, "fast")))
        for v in variants:
            vcfg = load_config(args.config, variant=v)
            for mode in args.mode:
                for method in args.method:
                    found.append((f"{v} {mode} {method}", vcfg.case_dir(tid, mode, method)))
        found = [(lab, d) for lab, d in found if (d / "summary.json").exists()]
        if not found:
            print(f"test {tid}: no variant runs found; skipped")
            continue
        cases = [compare.load_case(d) for _, d in found]
        labels = [lab for lab, _ in found]
        results = [compare.compare_case(ref, c, grid=args.grid) for c in cases]
        sections.append(compare.render(ref, results, labels))
        payload[str(tid)] = {"ref": ref["_dir"], "results": dict(zip(labels, results))}
    md = "\n".join(sections) if sections else "No cases found.\n"
    print(md)
    if args.out:
        _write_compare(Path(args.out), md, payload)
    return 0


def cmd_compare(args) -> int:
    if args.config:
        return _compare_matrix(args)
    if not (args.ref and args.cases):
        print("compare needs either -c CONFIG (with --test/--variants) or --ref and --cases")
        return 2
    from . import compare
    ref = compare.load_case(Path(args.ref))
    cases = [compare.load_case(Path(c)) for c in args.cases]
    labels = args.labels or [Path(c["_dir"]).as_posix() for c in cases]
    if len(labels) != len(cases):
        print("--labels needs one label per case")
        return 2
    results = [compare.compare_case(ref, c, grid=args.grid) for c in cases]
    md = compare.render(ref, results, labels)
    print(md)
    if args.out:
        _write_compare(Path(args.out), md, {"ref": ref["_dir"], "results": dict(zip(labels, results))})
    return 0


def cmd_check(args) -> int:
    _shared_tmpdir()
    if not args.verbose:  # OCSMesh is chatty at INFO; keep the check log readable
        logging.getLogger("ocsmesh").setLevel(logging.WARNING)
    from . import checks
    results = checks.run(only=args.only, skip_mpi=args.no_mpi)
    if checks.is_manager():
        md = checks.render(results)
        out_dir = Path(args.out) if args.out else Path("reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"checks_{time.strftime('%Y%m%d_%H%M')}.md"
        out.write_text(md)
        print(md)
        print(f"\nwritten: {out}")
    return 0 if all(r.status in ("pass", "skip", "info") for r in results) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ocsmesh-test", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data", help="download/crop DEMs and write the manifest")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--validate", action="store_true", help="compute valid-data fraction of each tile inside the AOI")
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("slurm", help="write SLURM scripts (and submit with --submit)")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--variant", default=None, help="DEM variant (dems.variants) to use")
    s.add_argument("--test", type=int, nargs="*", default=None)
    s.add_argument("--mode", nargs="*", default=None)
    s.add_argument("--method", nargs="*", default=None)
    s.add_argument("--hfun-only", action="store_true",
                   help="size function only (skip Geom and MeshDriver); results go to <method>-hfun")
    s.add_argument("--submit", action="store_true")
    s.set_defaults(fn=cmd_slurm)

    s = sub.add_parser("job", help="wrap another ocsmesh-test command in a SLURM script")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--name", required=True, help="job name ([A-Za-z0-9_.-])")
    s.add_argument("--ntasks", type=int, default=1, help="MPI ranks (srun -n)")
    s.add_argument("--cpus", type=int, default=1, help="cpus per task")
    s.add_argument("--nodes", type=int, default=1)
    s.add_argument("--walltime", default=None)
    s.add_argument("--submit", action="store_true")
    s.add_argument("args", nargs=argparse.REMAINDER, help="-- <ocsmesh-test arguments>")
    s.set_defaults(fn=cmd_job)

    s = sub.add_parser("tile-timing", help="time channel detection per tile (one tile per process)")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--tolerance", nargs="+", default=["none"], help="e.g. none 50")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--level", type=float, default=None, help="default: first channels block")
    s.add_argument("--width", type=float, default=None, help="default: first channels block")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_tile_timing)

    s = sub.add_parser("run", help="run one case (called by the SLURM scripts, or locally)")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--variant", default=None, help="DEM variant (dems.variants) to use")
    s.add_argument("--test", type=int, required=True)
    s.add_argument("--mode", default="serial")
    s.add_argument("--method", default="exact")
    s.add_argument("--out-dir", default=None)
    s.add_argument("--hfun-only", action="store_true", help="skip Geom and MeshDriver")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("variant", help="write a resampled/split copy of the DEMs (dems.variants)")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--variant", required=True, help="name under dems.variants")
    s.set_defaults(fn=cmd_variant)

    s = sub.add_parser("compare", help="score case directories against a reference case")
    s.add_argument("-c", "--config", default=None,
                   help="score DEM variants per test against the original runs (no --ref/--cases)")
    s.add_argument("--test", type=int, nargs="*", default=None, help="with -c: tests (default: all)")
    s.add_argument("--variants", nargs="*", default=None, help="with -c: variants (default: all)")
    s.add_argument("--mode", nargs="*", default=["serial"], help="with -c: variant run modes")
    s.add_argument("--method", nargs="*", default=["exact"], help="with -c: variant run methods")
    s.add_argument("--ref-mode", default="serial", help="with -c: reference mode")
    s.add_argument("--ref-method", default="exact", help="with -c: reference method")
    s.add_argument("--with-fast", action="store_true",
                   help="with -c: also score the original fast run (a resampled grid)")
    s.add_argument("--ref", default=None, help="reference case dir (holds summary.json)")
    s.add_argument("--cases", nargs="*", default=None, help="case dirs to score")
    s.add_argument("--labels", nargs="*", default=None, help="one label per case (default: path)")
    s.add_argument("--grid", type=int, default=1000, help="size-function grid points per axis")
    s.add_argument("--out", default=None, help="write markdown here (and a .json next to it)")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("report", help="render markdown from summary.json files")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--variant", default=None, help="DEM variant (dems.variants) to use")
    s.add_argument("--out", default=None, help="output directory (default: reports/)")
    s.add_argument("--figures", action="store_true", help="also plot hfun fields (needs hfun .2dm files)")
    s.add_argument("--figure-modes", nargs="*", default=None)
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("check", help="run OCSMesh regression checks on synthetic data")
    s.add_argument("--only", nargs="*", default=None, help="check names (substring match)")
    s.add_argument("--no-mpi", action="store_true", help="skip checks that need MPI even if under srun")
    s.add_argument("--out", default=None, help="output directory (default: reports/)")
    s.set_defaults(fn=cmd_check)

    args = p.parse_args(argv)
    _logging(args.verbose)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
