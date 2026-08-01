"""
Score a generated clip against the control track it was steered by.

Wraps edge_overlay.compare_sequence with the frame-extraction step, because every previous
measurement did that by hand and the two naming conventions (edges_%04d / result_%04d) are
easy to get one apart -- which silently scores frame N against frame N+1 and reports a
plausible-looking number.

    python3 measure_generated.py <generated.mp4> [--control DIR]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_overlay import compare_sequence                            # noqa: E402

DEFAULT_CONTROL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "real_samples", "output", "pdf_walkthrough", "frames")


def _null_baseline(control_dir: str, tolerance_px: int, samples: int = 6) -> float:
    """What random noise scores against this control at this tolerance. The floor to beat."""
    import numpy as np
    from PIL import Image
    from edge_overlay import _directed_hit_rate, _edges_from_render

    names = sorted(f for f in os.listdir(control_dir)
                   if f.startswith("edges_") and f.endswith(".png"))
    if not names:
        return 0.0
    rng = np.random.RandomState(0)
    picks = names[:: max(1, len(names) // samples)][:samples]
    scores = []
    for name in picks:
        cad = np.asarray(Image.open(os.path.join(control_dir, name)).convert("L")) > 127
        noise = rng.rand(*cad.shape).astype("float32") * 255.0
        scores.append(_directed_hit_rate(cad, _edges_from_render(noise), tolerance_px))
    return round(float(sum(scores) / len(scores)), 4)


def measure(video: str, control_dir: str, tolerance_px: int) -> dict:
    n_control = len([f for f in os.listdir(control_dir)
                     if f.startswith("edges_") and f.endswith(".png")])

    with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as work:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                        os.path.join(raw, "f_%05d.png")], check=True)
        frames = sorted(os.listdir(raw))
        if not frames:
            return {"ok": False, "reason": "ffmpeg extracted no frames"}

        # Resample onto the control's timeline instead of pairing by raw index. A generator
        # that returns a different frame count still covers the SAME camera move, so index i
        # of a 236-frame clip is a different moment from index i of the 97-frame control --
        # pairing them directly scores the wrong instants against each other and returns a
        # confidently wrong number. Gemini Omni returns 24 fps regardless of input, and Cosmos
        # caps at 93, so this is the normal case, not an edge case.
        last_src, last_dst = len(frames) - 1, max(n_control - 1, 1)
        for i in range(n_control):
            src = frames[round(i * last_src / last_dst)] if last_src else frames[0]
            os.link(os.path.join(raw, src), os.path.join(work, f"result_{i:04d}.png"))

        result = compare_sequence(control_dir, work, tolerance_px=tolerance_px)
        result["source_frames"] = len(frames)
        result["resampled_to"] = n_control

        # DENSITY-WEIGHTED RECALL. A plain recall average treats a frame showing a doorway, a
        # window and furniture as worth exactly as much as a frame showing one blank wall -- and
        # measured on the anchor shot, blank frames carry 0.16% edge pixels against 1.00% for a
        # rich one. There is almost nothing to match on the blank ones, so they score ~1.000
        # whatever the model does, and they quietly inflate every mean in this repo.
        #
        # Weighting by how much the CONTROL has to say makes a frame count for as much as it
        # actually tests. Reported alongside the unweighted number rather than replacing it, so
        # older figures stay comparable.
        #
        # THE NULL BASELINE. Score random noise against the same control at the same tolerance.
        # Without it a recall number cannot be read at all: at 5 px -- the tolerance every figure
        # in this repo was computed at until now -- pure noise scores 0.991, because the edge
        # detector always marks the top 8% of gradients and 8% coverage inside an 11x11
        # neighbourhood almost always contains a hit. A metric that scores noise at 0.99 is not
        # measuring the model.
        #
        # Reported with every run so the headline number is self-interpreting. Anything at or
        # below the null is not evidence of anything.
        result["null_baseline"] = _null_baseline(control_dir, tolerance_px)

        per = result.get("per_frame") or []
        if per:
            weights, values = [], []
            for f in per:
                w = float(f.get("cad_edge_fraction", 0.0))
                weights.append(w)
                values.append(f["edge_recall"])
            total = sum(weights)
            if total > 0:
                result["edge_recall_weighted"] = round(
                    float(sum(v * w for v, w in zip(values, weights)) / total), 4)
                result["control_edge_density_mean"] = round(float(total / len(weights)), 5)
        return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--control", default=DEFAULT_CONTROL)
    # Tolerance is in pixels AT THE CONTROL'S RESOLUTION, so it is not comparable across control
    # sizes: 3 px on a 720-wide control is a 1.5x stricter test than the same 3 px on the
    # 480-wide controls the oldest figures in MANIFEST.md were scored against.
    #
    # Default 2, not 3 and emphatically not 5. Measured against random noise on this control:
    # 5 px scores noise at 0.991, 3 px at 0.829, 2 px at 0.572, 1 px at 0.306. The separation
    # between a correct render and a wrong one is 0.29 at 5 px and 0.64 at 1 px. Tolerance was
    # raised to 5 earlier in this project purely to keep 720p numbers comparable with older 480p
    # ones -- a comparability fix that quietly destroyed the metric's ability to discriminate.
    ap.add_argument("--tolerance", type=int, default=2)
    args = ap.parse_args()

    result = measure(args.video, os.path.normpath(args.control), args.tolerance)
    if not result.get("frames_scored"):
        print(json.dumps(result, indent=2))
        return 1

    print(f"frames scored      {result['frames_scored']}")
    print(f"edge recall mean   {result['edge_recall_mean']}")
    print(f"edge recall min    {result['edge_recall_min']}")
    print(f"edge precision     {result['edge_precision_mean']}")
    print(f"missing frames     {result['missing_render_frames'] or 'none'}")
    print(f"tolerance          {result['tolerance_px']} px")
    if result.get("edge_recall_weighted") is not None:
        print(f"density-weighted   {result['edge_recall_weighted']}  "
              f"(control edge density {result['control_edge_density_mean']*100:.2f}%)")
    null = result.get("null_baseline")
    if null is not None:
        margin = result["edge_recall_mean"] - null
        verdict = "ABOVE noise" if margin > 0.05 else "AT OR BELOW noise -- not evidence"
        print(f"null baseline      {null}  (random noise on this control at {result['tolerance_px']} px)")
        print(f"margin over null   {margin:+.3f}  -> {verdict}")
    print("uncalibrated: compare this against another clip, not against a fixed pass mark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
