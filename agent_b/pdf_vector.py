"""
Read vector geometry out of a PDF floor plan, grouped by the drawing layer it came from.

An architect's PDF exported from CAD is not a picture of a drawing -- it is the drawing, in a
different container. The real client file here is an AutoCAD 2018 export carrying 42 named
layers as PDF optional content groups, with the same names the DWG uses (A-MUROS, A-PUERTAS,
A-VIDRIO), so layer_conventions.detect() applies to it unchanged.

Two things a PDF does not carry, both handled elsewhere:
  - real-world units; PDF space is 1/72 inch and means nothing on its own (scale_calibration)
  - the room-outline layer; 0-AREAS is non-printing and is absent from all 42 (wall_regions)

WHY PyMuPDF AND NOT A HAND-ROLLED READER. Layer attribution comes from BDC/EMC marked-content
operators, which nest and whose operands may be dictionaries rather than names. A regex scan
of the decompressed streams was written first and measured against this: the two agree on
total geometry to 0.1% (379,405 vs 379,798 line segments), but disagree by 11% on how much
belongs to A-MUROS (3,102 vs 2,773). The regex version is the wrong one -- it cannot track
nesting, so paths inside an inner /Span leak into the enclosing layer. It is not kept as a
fallback, because a reader that silently mis-attributes geometry is worse than no reader.

Coordinates are returned with Y INCREASING UPWARD, flipped from PDF's top-left origin, so
downstream geometry code sees the same handedness as the DXF path and areas come out
positive.
"""
import collections
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Segment types in a PyMuPDF drawing item: line, curve, rectangle, quad.
_SEGMENT_TYPES = ("l", "c", "re", "qu")
# Curves are flattened to this many chords. Wall outlines are overwhelmingly straight; this
# only matters for door swings and the odd rounded corner, where 8 is visually exact at the
# scale a floor plan is drawn.
_CURVE_CHORDS = 8


def _flip(points: np.ndarray, page_height: float) -> np.ndarray:
    out = points.copy()
    out[:, 1] = page_height - out[:, 1]
    return out


def _bezier(p0, p1, p2, p3, chords: int = _CURVE_CHORDS) -> List[Tuple[float, float]]:
    t = np.linspace(0.0, 1.0, chords + 1)[:, None]
    pts = ((1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1)
           + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3))
    return [tuple(p) for p in pts]


def _drawing_polylines(drawing: Dict[str, Any]) -> List[List[Tuple[float, float]]]:
    """Flatten one PyMuPDF drawing into polylines in PDF page coordinates."""
    polylines: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []

    for item in drawing["items"]:
        kind = item[0]
        if kind == "l":
            a, b = item[1], item[2]
            if not current:
                current = [(a.x, a.y)]
            current.append((b.x, b.y))
        elif kind == "c":
            p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
            if not current:
                current = [(p0.x, p0.y)]
            current.extend(_bezier((p0.x, p0.y), (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y))[1:])
        elif kind in ("re", "qu"):
            if current:
                polylines.append(current)
                current = []
            if kind == "re":
                r = item[1]
                polylines.append([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1),
                                  (r.x0, r.y0)])
            else:
                q = item[1]
                polylines.append([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y),
                                  (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)])
        else:                                  # 'm' and anything unrecognised break the run
            if current:
                polylines.append(current)
                current = []
    if current:
        polylines.append(current)

    if drawing.get("closePath"):
        for polyline in polylines:
            if len(polyline) > 2 and polyline[0] != polyline[-1]:
                polyline.append(polyline[0])
    return polylines


def is_vector_plan(pdf_path: str, min_segments: int = 500) -> Dict[str, Any]:
    """
    Decide whether this PDF holds a vector drawing at all before anything tries to measure it.

    A scanned plan is a different problem needing a different technique, and one that cannot
    reach the same accuracy. It must be refused here rather than silently producing geometry
    from whatever stray vectors a page border contributes.
    """
    import fitz

    with fitz.open(pdf_path) as doc:
        segments = 0
        images = 0
        for page in doc:
            images += len(page.get_images(full=True))
            for drawing in page.get_drawings():
                segments += sum(1 for item in drawing["items"] if item[0] in _SEGMENT_TYPES)
        layers = len(doc.get_ocgs())
        pages = doc.page_count

    return {
        "vector": segments >= min_segments,
        "segments": segments,
        "raster_images": images,
        "layers": layers,
        "pages": pages,
        "reason": (None if segments >= min_segments
                   else f"only {segments} vector segments; looks scanned or is not a drawing"),
    }


def extract(pdf_path: str, pages: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Per-page vector geometry grouped by layer, plus positioned text.

    Returns {"pages": [{"index", "width", "height", "layers": {name: [polyline...]},
                        "texts": [{"content", "pos"}], "unlayered": n}, ...]}
    with coordinates in PDF points, Y up.
    """
    import fitz

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    out_pages = []
    with fitz.open(pdf_path) as doc:
        indices = pages if pages is not None else range(doc.page_count)
        for index in indices:
            page = doc[index]
            height = page.rect.height
            layers: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
            unlayered = 0

            for drawing in page.get_drawings():
                layer = drawing.get("layer")
                if not layer:
                    unlayered += 1
                    continue
                for polyline in _drawing_polylines(drawing):
                    if len(polyline) >= 2:
                        layers[layer].append(_flip(np.asarray(polyline, dtype=float), height))

            texts = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        content = (span.get("text") or "").strip()
                        if not content:
                            continue
                        x0, y0, x1, y1 = span["bbox"]
                        texts.append({
                            "content": content,
                            "pos": ((x0 + x1) / 2.0, height - (y0 + y1) / 2.0),
                        })

            out_pages.append({
                "index": index,
                "width": page.rect.width,
                "height": height,
                "layers": {k: v for k, v in layers.items()},
                "texts": texts,
                "unlayered": unlayered,
            })

    return {"path": pdf_path, "pages": out_pages}


def layer_inventory(extraction: Dict[str, Any]) -> Dict[str, int]:
    """Segment count per layer across every extracted page, busiest first."""
    counts: collections.Counter = collections.Counter()
    for page in extraction["pages"]:
        for layer, polylines in page["layers"].items():
            counts[layer] += sum(max(0, len(p) - 1) for p in polylines)
    return dict(counts.most_common())
