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
        return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--control", default=DEFAULT_CONTROL)
    # Tolerance is in pixels AT THE CONTROL'S RESOLUTION, so it is not comparable across
    # control sizes: the default 3 px on a 720-wide control is a 1.5x stricter test than the
    # same 3 px on the 480-wide controls every earlier figure in MANIFEST.md was scored
    # against. Pass --tolerance 5 to read a 720p run on roughly the old 480p footing.
    ap.add_argument("--tolerance", type=int, default=3)
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
    print("uncalibrated: compare this against another clip, not against a fixed pass mark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
