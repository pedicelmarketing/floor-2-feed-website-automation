"""
How much do two generated videos of the same room disagree?

Every other check in this package compares a generated clip against the control track it was
given -- depth_metrics, edge_overlay, opening_count. None of them compares two generated clips
against each other, so the complaint that actually matters commercially has never had a
number: run the same scene twice and one clip has a sofa, the other a window, and neither is
wrong against the drawing because the drawing never said.

That gap is the reason this exists. A claim to have made generation reproducible is
unfalsifiable without a baseline, and the two fixes that worked today -- the depth encoding and
the camera -- were only demonstrable because a measurement existed beforehand.

Two quantities, because they answer different questions:

  STRUCTURAL agreement asks whether both clips put the walls, floor and openings in the same
  place. Measured by taking one clip's edges as the reference and scoring the other against
  them, reusing edge_overlay's distance-transform matching. This should already be high --
  structure is the part the control track constrains.

  CONTENT agreement asks whether both clips contain the same things. Measured as how much the
  two differ pixel for pixel, normalised against how much each one differs from itself over
  time. Normalising matters: two clips of a fast-moving camera differ more than two of a slow
  one for reasons that have nothing to do with agreement, so the raw difference is not
  comparable between scenes.

Same-model pairs (one model, different seeds) and cross-model pairs are reported separately.
The first asks "is this reproducible", the second asks "is this model-independent". A pipeline
can be one and not the other, and conflating them hides which.
"""
import itertools
import os
from typing import Any, Dict, List, Sequence

import numpy as np

# Frames are compared at this width; full resolution buys nothing here and costs time, since
# the question is "is there a sofa" rather than "is this edge 2 px out".
COMPARE_WIDTH = 240


def _load(directory: str, prefix: str = "result_") -> np.ndarray:
    from PIL import Image

    names = sorted(f for f in os.listdir(directory)
                   if f.startswith(prefix) and f.endswith(".png"))
    frames = []
    for name in names:
        image = Image.open(os.path.join(directory, name)).convert("L")
        scale = COMPARE_WIDTH / image.width
        frames.append(np.asarray(
            image.resize((COMPARE_WIDTH, max(1, int(image.height * scale)))), dtype=np.float32))
    return np.stack(frames) if frames else np.zeros((0, 1, 1), dtype=np.float32)


def _self_motion(clip: np.ndarray) -> float:
    """How much a clip differs from itself frame to frame -- the scale to normalise against."""
    if len(clip) < 2:
        return 1.0
    return float(np.mean(np.abs(np.diff(clip, axis=0)))) or 1.0


def content_agreement(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    1.0 means the two clips are identical; 0.0 means they differ as much as a clip differs
    from itself across the whole shot.

    Comparing only the frames both clips have. Cosmos emits 93 where Wan emits 97, and
    silently comparing a 93-frame clip against the first 93 of a 97-frame one would align
    different moments in the camera move and report disagreement that is really a timing
    offset.
    """
    n = min(len(a), len(b))
    if n == 0:
        return {"agreement": float("nan"), "frames": 0}
    a, b = a[:n], b[:n]
    difference = float(np.mean(np.abs(a - b)))
    scale = max(_self_motion(a), _self_motion(b))
    return {
        "agreement": float(max(0.0, 1.0 - difference / (scale * len(a) ** 0.5))),
        "raw_difference": difference,
        "frames": n,
    }


def structural_agreement(dir_a: str, dir_b: str) -> float:
    """
    Do both clips put the architecture in the same place?

    edge_overlay.compare_sequence expects a directory of `edges_*.png` as the reference, so
    one clip cannot be handed to it directly. The edge extraction it uses is reusable on its
    own, which is what happens here.
    """
    from edge_overlay import _directed_hit_rate, _edges_from_render

    a, b = _load(dir_a), _load(dir_b)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    scores = []
    for i in range(n):
        edges_a = _edges_from_render(a[i])
        edges_b = _edges_from_render(b[i])
        scores.append(_directed_hit_rate(edges_a, edges_b, 3))
    return float(np.nanmean(scores))


def compare_runs(runs: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    `runs` is [{"model": name, "seed": s, "dir": path}, ...].

    Returns per-pair scores plus the two headline numbers: agreement between runs of the same
    model, and agreement across models.
    """
    loaded = {r["dir"]: _load(r["dir"]) for r in runs}
    pairs = []
    for x, y in itertools.combinations(runs, 2):
        content = content_agreement(loaded[x["dir"]], loaded[y["dir"]])
        pairs.append({
            "a": f"{x['model']}/{x['seed']}",
            "b": f"{y['model']}/{y['seed']}",
            "same_model": x["model"] == y["model"],
            "content_agreement": content["agreement"],
            "raw_difference": content.get("raw_difference"),
            "structural_agreement": structural_agreement(x["dir"], y["dir"]),
            "frames": content["frames"],
        })

    def mean_of(selected, key):
        values = [p[key] for p in selected if p[key] == p[key]]     # drop NaN
        return float(np.mean(values)) if values else float("nan")

    same = [p for p in pairs if p["same_model"]]
    cross = [p for p in pairs if not p["same_model"]]
    return {
        "pairs": pairs,
        "same_model": {"n": len(same),
                       "content": mean_of(same, "content_agreement"),
                       "structural": mean_of(same, "structural_agreement")},
        "cross_model": {"n": len(cross),
                        "content": mean_of(cross, "content_agreement"),
                        "structural": mean_of(cross, "structural_agreement")},
    }


def report(result: Dict[str, Any]) -> str:
    lines = [f"{'pair':34s} {'same model':>11s} {'structure':>10s} {'content':>9s}"]
    for p in sorted(result["pairs"], key=lambda q: (not q["same_model"], q["a"])):
        lines.append(f"{p['a'] + ' vs ' + p['b']:34s} {'yes' if p['same_model'] else 'no':>11s} "
                     f"{p['structural_agreement']:10.3f} {p['content_agreement']:9.3f}")
    lines.append("")
    for label, key in (("same model, different seed", "same_model"), ("across models", "cross_model")):
        block = result[key]
        lines.append(f"{label:28s} n={block['n']:2d}  "
                     f"structure {block['structural']:.3f}  content {block['content']:.3f}")
    lines.append("")
    lines.append("structure: do both put the walls in the same place. content: do both contain")
    lines.append("the same things. 1.0 is identical; content is normalised against how much each")
    lines.append("clip moves, so it is comparable between scenes.")
    return "\n".join(lines)
