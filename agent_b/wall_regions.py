"""
Derive room regions and a real-world scale from a PDF floor plan's wall linework.

The PDF has no room-outline layer -- 0-AREAS is non-printing and absent -- so rooms have to
come from the walls. Walls, doors and glazing are buffered into solid barriers, and the free
space they leave is split into regions. Each region that contains exactly one area label is a
room whose printed area can be checked against its measured one.

WHAT WORKS AND WHAT DOES NOT, measured on the real client PDF (11 pages, 212 labelled rooms):

  SCALE: solved. Fitting metres-per-point independently on each page gives 35.0-37.0 mm per
  point on 10 of 11 pages -- a ±3% spread across pages that never see each other's data. That
  is 1:100, and it contradicts the "AR-50" title-block hint of 1:50, which is why that hint is
  used only as a sanity check and never as the answer.

  ROOM SEPARATION: NOT solved. Only 111 of 212 rooms (52%) end up in a region of their own;
  the rest merge with a neighbour, so one region carries two or more labels. Buffering the
  barriers more thickly does not help -- 0.8, 1.5 and 2.5 pt all yield the same merge count --
  which means these are real gaps in the linework, not walls drawn too thin to close.

  AREA ERROR on the rooms that do separate: median 8.8%, implying roughly 17 cm of wall
  position error on a 4 m room. Note this is FIVE TIMES the ±3% spread of the scale fit, so it
  is dominated by region boundaries being wrong, not by scale being wrong. Fixing separation
  should bring it down; no amount of better calibration will.

So this module measures honestly and does not pretend. `regions_for_page` reports how many
labels merged, and any caller drawing conclusions from the areas must account for it.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Layers buffered into barriers. Walls alone leave doorways open, which merges every room on a
# floor into one blob; adding door and glazing geometry plugs most openings.
BARRIER_LAYERS = ["A-MUROS", "A-MUROS-PROYECTADO", "A-PUERTAS", "A-HTCH", "A-VIDRIO"]
# Half the drawn line weight, in points. Measured to make no difference between 0.8 and 2.5,
# so it is not a tuning knob -- see the module docstring.
BARRIER_BUFFER_PT = 0.8
MIN_REGION_PT2 = 500.0

# "A1= 5,9 mt²" / "HL= 7,80mt²" / "S/C= 35 mt²" -- comma decimals, optional space, m2 or mt2.
AREA_LABEL = re.compile(r'^(.{0,8}?)=\s*([\d]+[,\.]?[\d]*)\s*m')

# A page whose independently fitted scale disagrees with the document consensus by more than
# this has failed, not discovered a differently-scaled page. Page 11 of the client file fits
# 7.9 mm/pt against a consensus of ~36 and produces 95% area errors.
SCALE_OUTLIER_RATIO = 1.5


def area_labels(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Room labels that state their own area, with positions."""
    out = []
    for text in page["texts"]:
        match = AREA_LABEL.search(text["content"])
        if match:
            out.append({
                "name": match.group(1).strip(),
                "printed_area_m2": float(match.group(2).replace(",", ".")),
                "pos": text["pos"],
            })
    return out


def regions_for_page(page: Dict[str, Any],
                     barrier_layers: List[str] = None,
                     buffer_pt: float = BARRIER_BUFFER_PT,
                     min_region_pt2: float = MIN_REGION_PT2) -> Dict[str, Any]:
    """
    Split the plan into free-space regions and attach each area label to the one containing it.

    Returns the cleanly separated rooms AND the merged ones. Reporting only the clean rooms
    would make the method look far better than it is: on this file that is half the rooms.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    solids = []
    for layer in (barrier_layers or BARRIER_LAYERS):
        for polyline in page["layers"].get(layer, []):
            if len(polyline) >= 2:
                try:
                    solids.append(LineString(polyline).buffer(buffer_pt))
                except Exception:              # noqa: BLE001 - degenerate path, skip
                    continue
    if not solids:
        return {"regions": [], "clean": [], "merged": [], "labels": area_labels(page),
                "reason": "no barrier geometry on this page"}

    barriers = unary_union(solids)
    free = barriers.convex_hull.difference(barriers)
    regions = [g for g in getattr(free, "geoms", [free]) if g.area > min_region_pt2]

    labels = area_labels(page)
    occupants: Dict[int, List[Dict[str, Any]]] = {}
    for index, region in enumerate(regions):
        for label in labels:
            if region.contains(Point(label["pos"])):
                occupants.setdefault(index, []).append(label)

    clean, merged = [], []
    for index, found in occupants.items():
        if len(found) == 1:
            clean.append({**found[0], "region_area_pt2": float(regions[index].area)})
        else:
            merged.append({"region_area_pt2": float(regions[index].area),
                           "labels": [f["name"] for f in found]})

    return {"regions": regions, "clean": clean, "merged": merged, "labels": labels,
            "unmatched": len(labels) - sum(len(v) for v in occupants.values())}


def fit_scale(clean_rooms: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """
    Least-squares metres² per point², through the origin, from rooms with a printed area.

    Overdetermined by roughly ten rooms per page, so the residual spread is a usable
    confidence signal rather than just a fit statistic.
    """
    if len(clean_rooms) < 3:
        return None
    x = np.array([r["region_area_pt2"] for r in clean_rooms], dtype=float)
    y = np.array([r["printed_area_m2"] for r in clean_rooms], dtype=float)
    if not (x > 0).all():
        return None
    m2_per_pt2 = float((x * y).sum() / (x * x).sum())
    mm_per_pt = float(np.sqrt(m2_per_pt2) * 1000.0)
    residuals = 100.0 * (x * m2_per_pt2 - y) / y
    return {
        "m2_per_pt2": m2_per_pt2,
        "mm_per_pt": mm_per_pt,
        # 1 pt is 0.3528 mm of paper, so this is the drawing's stated scale denominator.
        "drawing_scale": mm_per_pt / 0.3528,
        "median_abs_residual_pct": float(np.median(np.abs(residuals))),
        "worst_abs_residual_pct": float(np.abs(residuals).max()),
        "rooms_used": len(clean_rooms),
    }


def consensus_scale(per_page: List[Optional[Dict[str, float]]]) -> Dict[str, Any]:
    """
    Document-wide scale, with pages whose own fit disagrees wildly marked as failures.

    A page is not permitted to define its own scale when it disagrees with every other page by
    more than SCALE_OUTLIER_RATIO -- that is a broken fit, and averaging it in would corrupt
    the consensus rather than being outvoted by it.
    """
    values = [p["mm_per_pt"] for p in per_page if p]
    if not values:
        return {"mm_per_pt": None, "outliers": [], "reason": "no page produced a fit"}
    median = float(np.median(values))
    outliers = [i for i, p in enumerate(per_page)
                if p and (p["mm_per_pt"] > median * SCALE_OUTLIER_RATIO
                          or p["mm_per_pt"] < median / SCALE_OUTLIER_RATIO)]
    kept = [p["mm_per_pt"] for i, p in enumerate(per_page) if p and i not in outliers]
    return {
        "mm_per_pt": float(np.median(kept)) if kept else median,
        "spread_pct": float(100.0 * (max(kept) - min(kept)) / np.median(kept)) if kept else None,
        "pages_fitted": len(kept),
        "outlier_pages": outliers,
    }
