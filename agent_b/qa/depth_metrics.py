"""
Numeric geometry QA: does a generated video's apparent depth track the ground-truth depth
we rendered from the CAD-derived mesh?

WHAT THIS DOES AND DOES NOT MEASURE
-----------------------------------
Ground truth here is a true ray-cast metric depth (agent_b/3d_room_builder.py), but the
comparison signal is Depth Anything run over the generated video, which returns *relative*
depth with an unknown scale and offset per frame. So this compares ORDERING, not metres:

  - Spearman rank correlation is the headline metric precisely because it is invariant to
    any monotonic rescaling. It answers "is what should be far actually farther?"
  - It CANNOT catch a uniformly mis-scaled space. A room generated twice as deep, with
    every relative distance preserved, scores ~1.0 here. That failure mode is the
    perceptual judge's job (gemini_judge.py), which is why both exist.
  - Absolute-error metrics (RMSE etc.) are deliberately not reported: against a
    scale-ambiguous estimate they would be a meaningless number that looks authoritative.

The doorway-aperture metric is the targeted counterpart to the global correlation: it
tracks one interpretable, high-signal feature (the opening the camera moves toward) rather
than averaging the whole frame, so a localised failure cannot hide behind a good average.

WHAT "APERTURE" MEANS HERE, AND WHY IT IS NOT AREA
-------------------------------------------------
An earlier version of this compared the ground-truth void's pixel AREA against the
estimate's farthest-N-percent area. That was tautological -- a fixed percentile is a fixed
fraction of the frame by construction, so it could never track anything. Worse, the
comparison was incoherent in principle: ground truth has literal void past the doorway
(nothing was modelled there), whereas a good generation legitimately INVENTS a room in that
space. Demanding the output keep a hole where we left one would penalise correct behaviour.

What is actually checkable is CONTRAST: at the screen region where ground truth says there
is an opening, does the generated video read as markedly farther than the wall immediately
around it? That tests "the model put an opening where we specified one" without dictating
what lies beyond it.

STRUCTURAL LIMIT -- UNVERIFIABLE FRAMES
--------------------------------------
Once the camera passes through the opening, most of the frame is unmodelled space. There is
no ground truth there, so no numeric check is possible -- the model is inventing, and we
have nothing to compare against. Those frames are excluded and counted in
`frames_unverifiable`. A high count is a signal to model the ADJACENT rooms and render the
path against the whole unit, not a signal that the video is bad.
"""
import os
import re
import glob
import json
import subprocess
import tempfile
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from scipy.ndimage import binary_dilation

# Beyond this void fraction the camera has effectively passed through the opening and most
# of the frame is unmodelled space -- nothing to verify against, so the frame is skipped.
MAX_VERIFIABLE_VOID_FRACTION = 0.40

# Width in pixels of the wall ring sampled just outside the opening, for the contrast test.
APERTURE_RING_PX = 12

# Thresholds. Starting points, calibrated only against the single 49-frame dolly validated
# in this session -- treat as provisional until scored against more clips.
MIN_MEAN_CORRELATION = 0.70
MIN_WORST_FRAME_CORRELATION = 0.45
MAX_CORRELATION_JITTER = 0.15      # std-dev of per-frame correlation
MIN_APERTURE_CONTRAST = 8.0        # 0-255 depth units; opening must read farther than wall
MIN_APERTURE_HONOURED_FRACTION = 0.80  # of verifiable frames


def extract_frames(video_path: str, out_dir: str, prefix: str = "frame") -> List[str]:
    """Explodes a video to greyscale PNG frames via ffmpeg. Returns sorted frame paths."""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
         "-start_number", "0", os.path.join(out_dir, f"{prefix}_%04d.png")],
        check=True,
    )
    return sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.png")))


def _load_estimate(path: str) -> np.ndarray:
    """Depth Anything frame -> float array where LARGER = FARTHER.

    The estimate PNG follows the same convention our renderer writes (near = bright), so
    it is inverted here to put both signals in 'larger means farther' terms before ranking.
    """
    img = np.array(Image.open(path).convert("L"), dtype=np.float64)
    return 255.0 - img


def compare_frame(truth_depth: np.ndarray, truth_void: np.ndarray,
                  estimate: np.ndarray) -> Dict[str, Any]:
    """One frame's metrics. truth_depth is raw metric depth with inf at misses."""
    if estimate.shape != truth_depth.shape:
        estimate = np.array(
            Image.fromarray(estimate.astype(np.uint8)).resize(
                (truth_depth.shape[1], truth_depth.shape[0]), Image.BILINEAR),
            dtype=np.float64,
        )

    # Rank-correlate only over real surfaces. Void pixels are excluded because ground truth
    # has no depth there at all (inf), so including them would score the model on inventing
    # something we never specified.
    surface = np.isfinite(truth_depth) & (~truth_void)
    if surface.sum() < 100:
        return {"correlation": None, "aperture_truth_px": int(truth_void.sum()),
                "aperture_estimate_px": None, "note": "too few surface pixels"}

    rho, _ = spearmanr(truth_depth[surface].ravel(), estimate[surface].ravel())

    void_fraction = float(truth_void.mean())
    row: Dict[str, Any] = {
        "correlation": None if rho is None or np.isnan(rho) else float(rho),
        "void_px": int(truth_void.sum()),
        "void_fraction": void_fraction,
        "aperture_contrast": None,
        "verifiable": bool(0 < truth_void.sum() and void_fraction <= MAX_VERIFIABLE_VOID_FRACTION),
    }

    # Aperture contrast: at the known opening, does the generated video read as farther than
    # the wall immediately surrounding it? Compares estimated depth inside the void mask
    # against a thin ring just outside it. Positive => the opening was honoured.
    if row["verifiable"]:
        ring = binary_dilation(truth_void, iterations=APERTURE_RING_PX) & (~truth_void) & surface
        if ring.sum() >= 50:
            row["aperture_contrast"] = float(estimate[truth_void].mean() - estimate[ring].mean())
        else:
            row["verifiable"] = False

    return row


def evaluate(truth_dir: str, estimate_video: str,
             work_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Scores a generated video against the ground-truth frames that controlled it.

    truth_dir       directory of depth_XXXX.npy + void_XXXX.png (render_camera_path output)
    estimate_video  Depth Anything estimate over the GENERATED video
    """
    truth_depths = sorted(glob.glob(os.path.join(truth_dir, "depth_*.npy")))
    truth_voids = sorted(glob.glob(os.path.join(truth_dir, "void_*.png")))
    if not truth_depths:
        raise FileNotFoundError(
            f"No depth_*.npy in {truth_dir}. Re-run render_camera_path() with save_raw=True "
            f"-- the 8-bit PNGs alone cannot separate void from the farthest surface.")
    if len(truth_depths) != len(truth_voids):
        raise ValueError(f"{len(truth_depths)} depth arrays but {len(truth_voids)} void masks")

    temp_dir = work_dir or tempfile.mkdtemp(prefix="qa_depth_")
    estimate_frames = extract_frames(estimate_video, os.path.join(temp_dir, "estimate"), "est")

    n = min(len(truth_depths), len(estimate_frames))
    if len(truth_depths) != len(estimate_frames):
        print(f"WARNING: {len(truth_depths)} ground-truth frames vs {len(estimate_frames)} "
              f"estimate frames -- comparing the first {n}. A mismatch usually means the "
              f"generated video was resampled to a different frame rate.")

    per_frame = []
    for i in range(n):
        truth_depth = np.load(truth_depths[i])
        truth_void = np.array(Image.open(truth_voids[i]).convert("L")) > 127
        estimate = _load_estimate(estimate_frames[i])
        row = compare_frame(truth_depth, truth_void, estimate)
        row["frame"] = i
        per_frame.append(row)

    correlations = [r["correlation"] for r in per_frame if r["correlation"] is not None]
    contrasts = [r["aperture_contrast"] for r in per_frame if r["aperture_contrast"] is not None]
    honoured = [c for c in contrasts if c >= MIN_APERTURE_CONTRAST]

    summary: Dict[str, Any] = {
        "frames_compared": n,
        "frames_verifiable": len(contrasts),
        "frames_unverifiable": n - len(contrasts),
        "mean_correlation": float(np.mean(correlations)) if correlations else None,
        "worst_frame_correlation": float(np.min(correlations)) if correlations else None,
        "worst_frame_index": int(np.argmin(correlations)) if correlations else None,
        "correlation_jitter": float(np.std(correlations)) if correlations else None,
        "mean_aperture_contrast": float(np.mean(contrasts)) if contrasts else None,
        "aperture_honoured_fraction": (len(honoured) / len(contrasts)) if contrasts else None,
    }

    failures = []
    if summary["mean_correlation"] is None:
        failures.append("no usable frames")
    else:
        if summary["mean_correlation"] < MIN_MEAN_CORRELATION:
            failures.append(
                f"mean depth correlation {summary['mean_correlation']:.2f} < {MIN_MEAN_CORRELATION}")
        if summary["worst_frame_correlation"] < MIN_WORST_FRAME_CORRELATION:
            failures.append(
                f"frame {summary['worst_frame_index']} correlation "
                f"{summary['worst_frame_correlation']:.2f} < {MIN_WORST_FRAME_CORRELATION}")
        if summary["correlation_jitter"] > MAX_CORRELATION_JITTER:
            failures.append(
                f"correlation jitter {summary['correlation_jitter']:.2f} > "
                f"{MAX_CORRELATION_JITTER} (geometry unstable across frames)")

    honoured_fraction = summary["aperture_honoured_fraction"]
    if honoured_fraction is None:
        # Not a failure: it means no frame had a bounded, checkable opening. Surfaced as a
        # warning so a silently unchecked run is never mistaken for a clean pass.
        summary.setdefault("warnings", []).append(
            "no verifiable frames -- the opening was never a bounded region in view, so "
            "aperture was not checked at all")
    elif honoured_fraction < MIN_APERTURE_HONOURED_FRACTION:
        failures.append(
            f"opening reads as farther than the surrounding wall in only "
            f"{honoured_fraction:.0%} of verifiable frames "
            f"(need {MIN_APERTURE_HONOURED_FRACTION:.0%}; mean contrast "
            f"{summary['mean_aperture_contrast']:.1f})")

    # Two OPPOSITE causes of "unverifiable", and conflating them produces a misleading
    # warning. Either there is too much void (camera out in unmodelled space -> a real gap
    # in coverage), or there is none at all (no opening in frame -> nothing to measure,
    # which is perfectly fine). Report them separately.
    no_opening = sum(1 for r in per_frame if r["void_px"] == 0)
    too_much_void = sum(1 for r in per_frame
                        if r["void_px"] > 0 and r["void_fraction"] > MAX_VERIFIABLE_VOID_FRACTION)
    summary["frames_no_opening_in_view"] = no_opening
    summary["frames_camera_in_void"] = too_much_void

    if too_much_void:
        summary.setdefault("warnings", []).append(
            f"{too_much_void}/{n} frames unverifiable -- camera is in unmodelled space "
            f"beyond an opening. Build the mesh from the whole unit rather than one room.")
    if no_opening:
        summary.setdefault("warnings", []).append(
            f"{no_opening}/{n} frames had no opening in view, so the aperture check does "
            f"not apply to them. Not a defect -- correlation still covers these frames.")

    summary["failures"] = failures
    summary["verdict"] = "PASS" if not failures else "FAIL"
    summary["per_frame"] = per_frame
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Numeric geometry QA for a generated video.")
    parser.add_argument("--truth-dir", required=True, help="dir with depth_*.npy + void_*.png")
    parser.add_argument("--estimate-video", required=True,
                        help="Depth Anything estimate over the GENERATED video")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--json", action="store_true", help="print full JSON incl. per-frame")
    args = parser.parse_args()

    result = evaluate(args.truth_dir, args.estimate_video, args.work_dir)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"verdict:                 {result['verdict']}")
        print(f"frames compared:         {result['frames_compared']} "
              f"({result['frames_verifiable']} verifiable, "
              f"{result['frames_unverifiable']} unverifiable)")
        print(f"mean correlation:        {result['mean_correlation']}")
        print(f"worst frame correlation: {result['worst_frame_correlation']} "
              f"(frame {result['worst_frame_index']})")
        print(f"correlation jitter:      {result['correlation_jitter']}")
        print(f"mean aperture contrast:  {result['mean_aperture_contrast']}")
        print(f"aperture honoured:       {result['aperture_honoured_fraction']}")
        for w in result.get("warnings", []):
            print(f"  WARN: {w}")
        for f in result["failures"]:
            print(f"  FAIL: {f}")
