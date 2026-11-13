"""Where do runs differ from the reference in detected channels?

    python -m ocsmesh_test.diff_shapes REF_DIR CASE_DIR [CASE_DIR ...] --out DIR

For each case, writes the reference channels it missed and the channels
it added (GeoJSON, EPSG:4326) and prints the largest missed pieces. With
several cases it also prints how much of their missed areas coincide,
which tells whether they all miss the same channels.
"""

import argparse
import itertools
from pathlib import Path

from .compare import _clean_union, _read_shapes, _utm, load_case


def _parts(geom, crs, top):
    import geopandas as gpd
    import shapely
    parts = [p for p in shapely.get_parts(geom) if p.area > 0]
    gdf = gpd.GeoDataFrame(geometry=parts, crs=crs)
    gdf["km2"] = gdf.area / 1e6
    gdf = gdf.sort_values("km2", ascending=False)
    pts = gdf.geometry.representative_point().to_crs("EPSG:4326")
    gdf["lon"], gdf["lat"] = pts.x.round(4), pts.y.round(4)
    return gdf, gdf.head(top)


def _whole(src, other):
    """Polygons of ``src`` that ``other`` does not touch at all: whole
    channels missed (or added), not edges that moved."""
    import numpy as np
    import shapely
    polys = np.array([p for p in shapely.get_parts(src) if p.area > 0])
    if not len(polys):
        return 0.0, 0, 0, (0.0, 0.0)
    shapely.prepare(other)
    alone = polys[~shapely.intersects(other, polys)]
    areas = shapely.area(alone)
    sizes = (float(np.median(areas)), float(np.percentile(areas, 90))) if len(alone) else (0.0, 0.0)
    return float(areas.sum()) / 1e6, len(alone), len(polys), sizes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref")
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = load_case(Path(args.ref))
    a = _read_shapes(ref, "channels")
    if a is None:
        print(f"no channel shapes in {args.ref}")
        return 2
    crs = _utm(a)
    ua = _clean_union(a.to_crs(crs))
    print(f"reference channels: {ua.area / 1e6:.2f} km2 ({args.ref})")

    missed = {}
    for case_dir in args.cases:
        case = load_case(Path(case_dir))
        b = _read_shapes(case, "channels")
        if b is None:
            print(f"\n{case_dir}: no channel shapes, skipped")
            continue
        parts = Path(case_dir).resolve().parts
        name = (parts[-4] if "variants" in parts else "original") + "_" + parts[-2]
        ub = _clean_union(b.to_crs(crs))
        miss, add = ua.difference(ub), ub.difference(ua)
        missed[name] = miss
        all_miss, top = _parts(miss, crs, args.top)
        all_add, _ = _parts(add, crs, args.top)
        all_miss.to_crs("EPSG:4326").to_file(out / f"{name}_missed.geojson")
        all_add.to_crs("EPSG:4326").to_file(out / f"{name}_added.geojson")
        print(f"\n{name}: missed {miss.area / 1e6:.2f} km2 in {len(all_miss)} pieces, "
              f"added {add.area / 1e6:.2f} km2 in {len(all_add)} pieces; largest missed:")
        print(top[["km2", "lon", "lat"]].to_string(index=False, float_format="{:.3f}".format))
        for label, src, other in (("reference channels with no overlap", ua, ub),
                                  ("run channels with no overlap", ub, ua)):
            km2, n, total, sizes = _whole(src, other)
            print(f"  {label}: {n} of {total} polygons, {km2:.2f} km2"
                  f" (median {sizes[0]:.0f} m2, p90 {sizes[1]:.0f} m2)")

    if len(missed) > 1:
        print("\nshared missed area (km2, and share of the smaller one):")
        for (n1, g1), (n2, g2) in itertools.combinations(missed.items(), 2):
            inter = g1.intersection(g2).area
            small = min(g1.area, g2.area) or 1.0
            print(f"  {n1} & {n2}: {inter / 1e6:.2f} km2 ({100 * inter / small:.0f}%)")
    print(f"\nGeoJSON written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
