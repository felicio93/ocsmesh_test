"""Figure: reference size function, a run's size function, and their ratio.

    python -m ocsmesh_test.plot_hfun REF_DIR CASE_DIR [CASE_DIR ...] --out FILE.png
        [--labels A B ...] [--bbox LONMIN LATMIN LONMAX LATMAX] [--title TEXT]

One row per case: reference | case | case / reference. Detected channels
(when saved) are drawn as outlines on the first two panels.
"""

import argparse
from pathlib import Path

import numpy as np

from .compare import _meshdata, _read_shapes, load_case


def _tri(md):
    import matplotlib.tri as mtri
    xy = np.asarray(md.coords, dtype=float)
    return mtri.Triangulation(xy[:, 0], xy[:, 1], np.asarray(md.tria)), \
        np.asarray(md.values, dtype=float).ravel()


def _outline(ax, case, bbox):
    shapes = _read_shapes(case, "channels")
    if shapes is None or not len(shapes):
        return
    shapes = shapes.to_crs("EPSG:4326")
    if bbox:
        shapes = shapes.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    if len(shapes):
        shapes.boundary.plot(ax=ax, color="k", linewidth=0.3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref")
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--bbox", nargs=4, type=float)
    ap.add_argument("--title")
    ap.add_argument("--grid", type=int, default=800)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    from .compare import _interpolator

    ref = load_case(Path(args.ref))
    md_r = _meshdata(ref["hfun_path"], "EPSG:4326")
    tri_r, v_r = _tri(md_r)
    lo, hi = float(np.nanmin(v_r)), float(np.nanmax(v_r))
    labels = args.labels or [Path(c).resolve().parts[-4] for c in args.cases]
    bbox = args.bbox
    if bbox is None:
        (x0, y0), (x1, y1) = md_r.coords.min(axis=0), md_r.coords.max(axis=0)
        bbox = [x0, y0, x1, y1]
    gx, gy = np.meshgrid(np.linspace(bbox[0], bbox[2], args.grid),
                         np.linspace(bbox[1], bbox[3], args.grid))
    H_r = np.ma.filled(np.ma.masked_invalid(_interpolator(md_r)(gx, gy)).astype(float), np.nan)

    n = len(args.cases)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.6 * n), squeeze=False,
                             constrained_layout=True)
    aspect = 1 / np.cos(np.radians(0.5 * (bbox[1] + bbox[3])))
    norm = LogNorm(vmin=max(lo, 1.0), vmax=hi)
    for row, (case_dir, label) in enumerate(zip(args.cases, labels)):
        case = load_case(Path(case_dir))
        md_o = _meshdata(case["hfun_path"], "EPSG:4326")
        tri_o, v_o = _tri(md_o)
        H_o = np.ma.filled(np.ma.masked_invalid(_interpolator(md_o)(gx, gy)).astype(float), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = H_o / H_r
        a_ref, a_run, a_rat = axes[row]
        m = a_ref.tripcolor(tri_r, v_r, norm=norm, cmap="viridis", shading="gouraud")
        a_run.tripcolor(tri_o, v_o, norm=norm, cmap="viridis", shading="gouraud")
        _outline(a_ref, ref, args.bbox)
        _outline(a_run, case, args.bbox)
        r = a_rat.pcolormesh(gx, gy, ratio, cmap="RdBu_r", shading="auto",
                             norm=TwoSlopeNorm(vcenter=1.0, vmin=0.5, vmax=1.5))
        a_ref.set_title(f"reference (native DEM), {len(md_r.coords):,} nodes")
        a_run.set_title(f"{label}, {len(md_o.coords):,} nodes")
        a_rat.set_title(f"{label} / reference")
        for ax in (a_ref, a_run, a_rat):
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])
            ax.set_aspect(aspect)
            ax.tick_params(labelsize=7)
        fig.colorbar(m, ax=[a_ref, a_run], label="size (m)", shrink=0.85)
        fig.colorbar(r, ax=a_rat, label="ratio (red: coarser, blue: finer)", shrink=0.85)
    if args.title:
        fig.suptitle(args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
