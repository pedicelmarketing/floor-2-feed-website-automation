"""
How accurately can walls be read from a PDF floor plan? Measure it, do not estimate it.

Ground truth is printed on the drawing: every room is labelled with its own area
("A1= 5,9 mt²"), roughly 19 per page. Extract the walls, derive each room's region, compute
its area, and compare against the architect's own figure. No external reference needed, and
no dependence on the DWG -- which cannot serve here anyway, since the PDF and the DWG cover
the same development but different apartments.

Run:
    python3 agent_b/real_samples/measure_pdf_accuracy.py [path/to/plan.pdf]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layer_conventions import detect, describe          # noqa: E402
from pdf_vector import extract, is_vector_plan, layer_inventory  # noqa: E402
from wall_regions import consensus_scale, fit_scale, regions_for_page  # noqa: E402

UPLOADS = "/home/openclaw/floor-2-feed-website-automation/uploads"
DEFAULT_PDF = "*_floor-plans-estado-reformado.pdf"


def main(pdf_path: str) -> int:
    print(f"PDF: {os.path.basename(pdf_path)}\n")

    probe = is_vector_plan(pdf_path)
    print(f"vector drawing: {probe['vector']}  "
          f"({probe['segments']} segments, {probe['raster_images']} raster images, "
          f"{probe['layers']} layers, {probe['pages']} pages)")
    if not probe["vector"]:
        # A scan needs a different and less accurate technique. Refusing is the correct
        # outcome, not a failure to handle it here.
        print(f"REFUSED: {probe['reason']}")
        return 1

    extraction = extract(pdf_path)
    inventory = layer_inventory(extraction)
    detection = detect(counts=inventory)
    print()
    print(describe(detection))
    print(f"  (room_boundary absent is expected: 0-AREAS is non-printing and not exported)")

    print(f"\n{'page':>5} {'labels':>7} {'clean':>6} {'merged':>7} {'mm/pt':>7} {'median err %':>13}")
    per_page, clean_total, label_total, errors = [], 0, 0, []
    for page in extraction["pages"]:
        result = regions_for_page(page)
        scale = fit_scale(result["clean"])
        per_page.append(scale)
        clean_total += len(result["clean"])
        label_total += len(result["labels"])
        if scale:
            errors.append((page["index"], scale["median_abs_residual_pct"]))
        print(f"{page['index'] + 1:>5} {len(result['labels']):>7} {len(result['clean']):>6} "
              f"{len(result['merged']):>7} "
              f"{scale['mm_per_pt'] if scale else float('nan'):>7.1f} "
              f"{scale['median_abs_residual_pct'] if scale else float('nan'):>13.1f}")

    agreed = consensus_scale(per_page)
    good = [e for i, e in errors if i not in agreed["outlier_pages"]]

    print("\n--- scale ---")
    print(f"  consensus            {agreed['mm_per_pt']:.1f} mm per point "
          f"-> drawing scale about 1:{agreed['mm_per_pt'] / 0.3528:.0f}")
    print(f"  agreement            {agreed['spread_pct']:.1f}% spread across "
          f"{agreed['pages_fitted']} independently fitted pages")
    print(f"  pages rejected       {[i + 1 for i in agreed['outlier_pages']] or 'none'}")

    print("\n--- room separation ---")
    print(f"  rooms isolated       {clean_total}/{label_total} "
          f"({100 * clean_total / max(1, label_total):.0f}%) — the rest merge with a neighbour")

    print("\n--- room area error, on isolated rooms only ---")
    median = float(np.median(good)) if good else float("nan")
    print(f"  median area error    {median:.1f}%")
    print(f"  worst page           {max(good):.1f}%" if good else "  worst page  n/a")
    print()
    print("  This is an AREA error and it is not the same thing as wall-position error. It")
    print("  carries the region-separation failure with it -- a region that swallows a slice")
    print("  of corridor is wrong about its area without any wall being in the wrong place.")
    print("  For wall position, use the direct measurement below.")

    doors = door_widths(extraction, agreed["mm_per_pt"])
    print("\n--- wall position, measured directly against known door sizes ---")
    print(f"  doors measured       {doors['count']}, median {doors['median_m']:.3f} m")
    for standard, stats in sorted(doors["clusters"].items()):
        print(f"  {standard:.2f} m doors        {stats['count']:4d} found, "
              f"measured {stats['mean_m']:.3f} m -> off by {stats['error_cm']:+.1f} cm")
    print()
    print("  Doors come in standard widths, so this compares extracted geometry against a")
    print("  known length rather than against an area. It is the honest wall-position figure.")
    return 0


# Real doors are manufactured to these widths, which makes them a free ruler lying inside
# every drawing: extract one, and any discrepancy is the pipeline's error, not the building's.
STANDARD_DOOR_WIDTHS_M = (0.70, 0.80, 0.90)


def door_widths(extraction: dict, mm_per_pt: float,
                door_layer: str = "A-PUERTAS", tolerance_m: float = 0.05) -> dict:
    """
    Measure every door in the drawing and compare against the nearest standard width.

    A door is drawn as a leaf plus its swing arc, and both are as long as the door is wide,
    so the larger side of the shape's bounding box is the door width. That slightly
    overestimates -- it includes the drawn line thickness -- which is worth remembering when
    reading a consistently positive error.
    """
    widths = []
    for page in extraction["pages"]:
        for polyline in page["layers"].get(door_layer, []):
            points = np.asarray(polyline, dtype=float)
            if len(points) < 2:
                continue
            span = points.max(axis=0) - points.min(axis=0)
            metres = float(max(span)) * mm_per_pt / 1000.0
            if 0.4 <= metres <= 1.6:           # discard hardware and mis-grouped geometry
                widths.append(metres)

    widths = np.asarray(widths)
    clusters = {}
    for standard in STANDARD_DOOR_WIDTHS_M:
        selected = widths[np.abs(widths - standard) < tolerance_m]
        if len(selected):
            clusters[standard] = {
                "count": int(len(selected)),
                "mean_m": float(selected.mean()),
                "error_cm": float((selected.mean() - standard) * 100.0),
            }
    return {
        "count": int(len(widths)),
        "median_m": float(np.median(widths)) if len(widths) else float("nan"),
        "clusters": clusters,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        matches = sorted(glob.glob(os.path.join(UPLOADS, DEFAULT_PDF)))
        if not matches:
            print(f"No PDF found matching {DEFAULT_PDF} in {UPLOADS}")
            sys.exit(2)
        target = matches[0]
    sys.exit(main(target))
