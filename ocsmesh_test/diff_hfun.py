"""Where do two size functions differ?

    python -m ocsmesh_test.diff_hfun REF_DIR CASE_DIR --out FILE.tif [--grid 1000]

Writes a 4-band GeoTIFF on the grid ``compare`` uses: reference size,
run size, relative difference, and the shift-tolerant difference (both
NaN outside the refined area), then prints how the difference splits
with distance from the reference channels when the run saved them.
"""

import argparse
from pathlib import Path

import numpy as np

from .compare import (REL_TOL, SHIFT_M, _interpolator, _meshdata,
                      _read_shapes, _shift_tolerant_error, load_case)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref")
    ap.add_argument("case")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=1000)
    args = ap.parse_args(argv)

    import rasterio
    from rasterio.transform import from_bounds

    ref, case = load_case(Path(args.ref)), load_case(Path(args.case))
    md_r = _meshdata(ref["hfun_path"], ref.get("hfun_crs"))
    md_o = _meshdata(case["hfun_path"], case.get("hfun_crs"))
    print(f"size-function mesh nodes: reference {len(md_r.coords)}, run {len(md_o.coords)}")
    xmin, ymin = md_r.coords.min(axis=0)
    xmax, ymax = md_r.coords.max(axis=0)
    n = args.grid
    # top row first, as rasters are stored
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymax, ymin, n))
    H_r = np.ma.filled(np.ma.masked_invalid(_interpolator(md_r)(gx, gy)).astype(float), np.nan)
    H_o = np.ma.filled(np.ma.masked_invalid(_interpolator(md_o)(gx, gy)).astype(float), np.nan)
    hmax = float(np.nanmax(np.asarray(md_r.values)))
    with np.errstate(invalid="ignore"):
        refined = np.fmin(H_r, H_o) < 0.999 * hmax
    dx = (xmax - xmin) / (n - 1)
    cell_m = dx * 111_320.0 * np.cos(np.radians(0.5 * (ymin + ymax))) \
        if abs(xmax) <= 360 else dx
    k = max(1, int(np.ceil(SHIFT_M / cell_m)))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(refined, np.abs(H_o - H_r) / H_r, np.nan)
        e1 = _shift_tolerant_error(H_r, H_o, k)
        e2 = _shift_tolerant_error(H_o, H_r, k) * H_o / H_r
        shift = np.where(refined, np.fmax(e1, e2), np.nan)

    crs = ref.get("hfun_crs") or "EPSG:4326"
    half = 0.5 * dx
    transform = from_bounds(xmin - half, ymin - half, xmax + half, ymax + half, n, n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", width=n, height=n, count=4,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan,
                       compress="deflate") as dst:
        for i, (band, name) in enumerate(((H_r, "ref"), (H_o, "run"), (rel, "rel"),
                                          (shift, "shifted")), start=1):
            dst.write(band.astype("float32"), i)
            dst.set_band_description(i, name)
    print(f"written {out} (bands: ref, run, rel, shifted; cell {cell_m:.0f} m)")

    over = refined & (rel > REL_TOL)
    print(f"refined cells {int(refined.sum())}, rel >{REL_TOL:.0%}: {int(over.sum())} "
          f"({over.sum() / max(refined.sum(), 1):.0%}); run larger in "
          f"{int((over & (H_o > H_r)).sum())}, smaller in {int((over & (H_o < H_r)).sum())}")

    chan = _read_shapes(ref, "channels")
    if chan is not None and len(chan):
        from rasterio.features import rasterize
        from scipy.ndimage import distance_transform_edt
        mask = rasterize(((g, 1) for g in chan.to_crs(crs).geometry), out_shape=(n, n),
                         transform=transform, fill=0, all_touched=True).astype(bool)
        dist = distance_transform_edt(~mask) * cell_m
        print("cells over the threshold by distance from the reference channels:")
        for lo, hi in ((0, 1), (1, 250), (250, 500), (500, 1000), (1000, 1e12)):
            sel = refined & (dist >= lo) & (dist < hi)
            if sel.any():
                bad = (sel & over).sum()
                label = "inside" if lo == 0 else f"{lo:.0f}-{hi:.0f} m" if hi < 1e12 else f">{lo:.0f} m"
                print(f"  {label:>12}: {int(sel.sum()):7d} refined, {int(bad):7d} over "
                      f"({bad / sel.sum():.0%}), median rel {np.nanmedian(rel[sel]):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
