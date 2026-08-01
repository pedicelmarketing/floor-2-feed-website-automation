"""
Score every generated clip on BOTH axes and write one table.

The whole point is the pairing. Adherence alone drove this pipeline from 0.24 to 0.90 while the
output got no better to look at, because nothing was watching the other side. A clip that is
accurate and dead and a clip that is beautiful and wrong both fail; only the two numbers
together say which failure you have.

    python3 score_library.py                       # everything in outputs/generated
    python3 score_library.py --repeats 2           # average two judge samples per clip
    python3 score_library.py --only wan,ltx        # substring filter
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gemini_judge import QualityJudge                                # noqa: E402
from measure_generated import measure                                # noqa: E402

REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
GENERATED = os.path.join(REPO, "outputs", "generated")
CONTROL = os.path.join(REPO, "agent_b", "real_samples", "output", "ground_truth", "frames")
RESULTS = os.path.join(REPO, "outputs", "scores.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--tolerance", type=int, default=5)
    ap.add_argument("--skip-adherence", action="store_true",
                    help="judge only; useful when the control track has been re-rendered and "
                         "the old clips are no longer comparable to it")
    args = ap.parse_args()

    clips = sorted(glob.glob(os.path.join(GENERATED, "*.mp4")))
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        clips = [c for c in clips if any(w in os.path.basename(c) for w in wanted)]
    if not clips:
        print("no clips found")
        return 1

    judge = QualityJudge()
    if judge.client is None:
        print("GEMINI_API_KEY not set -- cannot score quality")
        return 1

    previous = {}
    if os.path.exists(RESULTS):
        previous = {r["clip"]: r for r in json.load(open(RESULTS))}

    rows = []
    for path in clips:
        name = os.path.basename(path)[:-4]
        row = {"clip": name}

        if not args.skip_adherence:
            try:
                m = measure(path, CONTROL, args.tolerance)
                row["adherence"] = m.get("edge_recall_mean")
            except Exception as e:
                row["adherence"] = None
                row["adherence_error"] = f"{type(e).__name__}"

        q = judge.score(path, repeats=args.repeats)
        if q.get("skipped"):
            row["quality_error"] = q.get("reason")
        else:
            row.update({k: q[k] for k in
                        ("photorealism", "lighting", "composition", "motion_quality",
                         "temporal_stability", "marketing_grade", "quality",
                         "reads_as_real_footage", "worst_defect", "single_biggest_fix")})
        rows.append(row)

        a = row.get("adherence")
        a_txt = "  --  " if a is None else f"{a:.3f}"
        print(f"{name[:46]:<46} adherence {a_txt:>6}  "
              f"quality {str(row.get('quality', '--')):>5}  "
              f"marketing {str(row.get('marketing_grade', '--')):>4}")

    merged = {**previous, **{r["clip"]: r for r in rows}}
    with open(RESULTS, "w") as fh:
        json.dump(sorted(merged.values(), key=lambda r: r["clip"]), fh, indent=2)
    print(f"\nwrote {RESULTS}  ({len(merged)} clips on record)")

    scored = [r for r in rows if r.get("quality") is not None]
    if scored:
        best_q = max(scored, key=lambda r: r["quality"])
        print(f"best looking : {best_q['clip']}  quality {best_q['quality']}")
        with_both = [r for r in scored if r.get("adherence")]
        if with_both:
            best_a = max(with_both, key=lambda r: r["adherence"])
            print(f"most accurate: {best_a['clip']}  adherence {best_a['adherence']:.3f} "
                  f"(quality {best_a['quality']})")
            # The pairing is the point: name the clip that is least bad on both at once.
            best_both = max(with_both, key=lambda r: min(r["adherence"] * 10, r["quality"]))
            print(f"best balance : {best_both['clip']}  "
                  f"adherence {best_both['adherence']:.3f}  quality {best_both['quality']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
