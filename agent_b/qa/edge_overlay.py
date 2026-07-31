"""
Score how well a generated frame's structure lands where the CAD says it should.

This is the second, independent accuracy measure. The existing depth check
(agent_b/qa/depth_metrics.py) asks whether the *relative* depth ordering of the generated
video tracks the CAD depth -- it is rank-based, so it is blind to a space that is uniformly
mis-scaled or shifted. This one asks a different question: are there edges where the drawing
puts edges? A wall that is in the right depth order but 30 cm to the left fails here and
passes there.

Method, and why not SSIM. The obvious approach is SSIM between the CAD edge map and a Canny
of the render, but edge maps are sparse -- typically 2-5% of pixels -- so SSIM is dominated
by the empty background the two images trivially agree on, and moves very little when the
edges themselves disagree. Instead this measures distance to the nearest edge, via a
distance transform, and reports:

    recall     of the CAD's edge pixels, the fraction with a rendered edge within
               `tolerance_px`. This is the number that matters: it asks whether the
               architecture the drawing specifies actually got drawn.
    precision  of the rendered edge pixels, the fraction near a CAD edge.

EXPECT LOW PRECISION, AND DO NOT TUNE FOR IT. The render legitimately contains edges the
blockout has no concept of -- the join between floorboards, the edge of a bed, a shadow
line, a picture frame. Those are the model doing its job. Precision is reported for context,
never gated on. Only recall is a defect signal.
"""
import os
from typing import Any, Dict, List

import numpy as np

# Structure may be a couple of pixels off without anyone perceiving it as wrong; a wall in
# the wrong place is off by far more. 3 px at 480x832 is roughly 0.6% of frame width.
DEFAULT_TOLERANCE_PX = 3

# PROVISIONAL -- this is a guess, not a calibrated pass mark, and nothing should be failed on
# it yet. Measured on the one clip available: mean recall 0.52, and still only 0.50 when the
# render's edge threshold is loosened until a fifth of all pixels count as edges. So roughly
# half the drawing's edge pixels have no rendered edge near them.
#
# That is not necessarily a defect, which is exactly why this does not gate. The model is
# asked to invent furnishings, and a bed against a wall legitimately hides the wall-to-floor
# junction that the blockout says is there. Occlusion by invented furniture and genuine
# structural drift are indistinguishable to this metric. Separating them needs either clips
# with known-good and known-bad structure to calibrate against, or a furniture-free render.
#
# The instrument itself IS verified: 1.000 against itself, and degrading as an image is
# shifted away from itself.
PROVISIONAL_EDGE_RECALL = 0.55


def _edges_from_render(gray: np.ndarray, percentile: float = 92.0) -> np.ndarray:
    """
    Binary edge map of a generated frame, via Sobel magnitude thresholded at a percentile.

    A percentile rather than a fixed threshold because generated frames vary a lot in
    contrast: a fixed value finds nothing in a dim frame and everything in a bright one, and
    that variation would show up as a quality trend that is really an exposure trend.
    """
    from scipy import ndimage

    gx = ndimage.sobel(gray, axis=1, mode="nearest")
    gy = ndimage.sobel(gray, axis=0, mode="nearest")
    magnitude = np.hypot(gx, gy)
    return magnitude >= np.percentile(magnitude, percentile)


def _directed_hit_rate(source: np.ndarray, target: np.ndarray, tolerance_px: int) -> float:
    """Fraction of `source` pixels lying within tolerance_px of any `target` pixel."""
    from scipy import ndimage

    if not source.any():
        return float("nan")
    if not target.any():
        return 0.0
    distance = ndimage.distance_transform_edt(~target)
    return float((distance[source] <= tolerance_px).mean())


def compare_frame(cad_edge_png: str, render_png: str,
                  tolerance_px: int = DEFAULT_TOLERANCE_PX) -> Dict[str, float]:
    """Compare one CAD edge map against one generated frame."""
    from PIL import Image

    cad = np.asarray(Image.open(cad_edge_png).convert("L"))
    render = np.asarray(Image.open(render_png).convert("L")).astype(np.float32)
    if cad.shape != render.shape:
        render = np.asarray(
            Image.open(render_png).convert("L").resize((cad.shape[1], cad.shape[0])),
            dtype=np.float32)

    cad_edges = cad > 127                      # the edge map is already binary 0/255
    render_edges = _edges_from_render(render)

    return {
        "edge_recall": _directed_hit_rate(cad_edges, render_edges, tolerance_px),
        "edge_precision": _directed_hit_rate(render_edges, cad_edges, tolerance_px),
        "cad_edge_fraction": float(cad_edges.mean()),
    }


def compare_sequence(control_dir: str, render_dir: str,
                     tolerance_px: int = DEFAULT_TOLERANCE_PX) -> Dict[str, Any]:
    """
    Score every frame of a clip. Returns per-frame numbers plus a verdict.

    Pairs files by index rather than by sorted order, so a missing frame is reported instead
    of silently shifting the whole sequence by one and scoring the wrong pairs against each
    other.
    """
    cad_frames = sorted(f for f in os.listdir(control_dir)
                        if f.startswith("edges_") and f.endswith(".png"))
    per_frame: List[Dict[str, float]] = []
    missing: List[str] = []

    for name in cad_frames:
        index = name[len("edges_"):-len(".png")]
        render_png = os.path.join(render_dir, f"result_{index}.png")
        if not os.path.exists(render_png):
            missing.append(index)
            continue
        scores = compare_frame(os.path.join(control_dir, name), render_png, tolerance_px)
        scores["frame"] = int(index)
        per_frame.append(scores)

    if not per_frame:
        return {"ok": False, "reason": "no frame pairs found", "missing": missing}

    recalls = np.array([f["edge_recall"] for f in per_frame], dtype=float)
    precisions = np.array([f["edge_precision"] for f in per_frame], dtype=float)
    weak = [int(f["frame"]) for f in per_frame if f["edge_recall"] < PROVISIONAL_EDGE_RECALL]

    # Deliberately no pass/fail. The measurement is trustworthy; the pass mark is not, so
    # emitting a verdict would launder a guess into a result. Compare clips against each
    # other -- which this supports -- rather than against PROVISIONAL_EDGE_RECALL.
    return {
        "calibrated": False,
        "frames_scored": len(per_frame),
        "missing_render_frames": missing,
        "edge_recall_mean": round(float(np.nanmean(recalls)), 4),
        "edge_recall_min": round(float(np.nanmin(recalls)), 4),
        "edge_precision_mean": round(float(np.nanmean(precisions)), 4),
        "tolerance_px": tolerance_px,
        "provisional_threshold": PROVISIONAL_EDGE_RECALL,
        "frames_below_provisional": weak,
        "per_frame": per_frame,
    }
