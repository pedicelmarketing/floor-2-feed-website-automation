"""
Combined QA gate + bounded auto-retry loop.

COMBINING THE TWO CHECKS
------------------------
Geometry is non-negotiable: if the numeric check fails, the run fails regardless of how
good it looks. A render that is beautiful and not this building is worse than useless for
real-estate marketing -- it is a false claim about a property.

The perceptual judge can only downgrade a numeric PASS to MARGINAL, never override it to
FAIL on its own. Rationale: the numeric check is deterministic and grounded in the actual
CAD geometry; the judge is a language model whose subtler verdicts vary run to run
(measured -- see below). Letting the less reliable signal veto the more reliable one would
put billable regeneration at the mercy of sampling noise.

WHY THE JUDGE IS POLLED k TIMES
-------------------------------
Measured on one fixed 49-frame clip, gemini-2.5-flash, four identical calls:
  - doorway tracking OK  4/4  (unanimous)
  - wall stability   OK  4/4  (unanimous)
  - flicker          OK  2/4  (split)
So structural judgements are stable, but the marginal texture call is a coin flip. A single
call would therefore trigger -- or skip -- a regeneration essentially at random. Majority
voting over k calls collapses that variance, and `judge_agreement` is reported so a
borderline result is visibly borderline rather than silently rounded.
"""
import os
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List

from . import depth_metrics
from .gemini_judge import GeminiJudge, DEFAULT_MODEL


# Dial -> concrete parameter change. Values map onto the Wan VACE graph used for the
# validated run (WanVideoVACEEncode.strength, WanVideoSampler steps/denoise).
DIAL_ACTIONS: Dict[str, Dict[str, Any]] = {
    "control_strength": {"vace_strength_delta": +0.15,
                         "why": "walls drift or geometry does not track -- bind control harder"},
    "add_canny":        {"add_canny_control": True,
                         "why": "lines wobble though volume is right -- add an edge signal"},
    "denoise_up":       {"denoise_delta": +0.10,
                         "why": "reads as a CG render -- let the model render more freely"},
    "denoise_down":     {"denoise_delta": -0.10,
                         "why": "photoreal but ignoring control geometry -- preserve input"},
    "increase_steps":   {"steps_multiplier": 3,
                         "why": "surfaces swim -- draft sampling, needs more steps"},
    "none":             {},
}


@dataclass
class RetryPolicy:
    """Hard caps. Both are enforced; whichever binds first stops the loop."""
    max_attempts: int = 3
    max_spend_units: float = 3.0        # caller-defined units (e.g. credits per generation)
    cost_per_attempt: float = 1.0


@dataclass
class QAResult:
    verdict: str                         # PASS | MARGINAL | FAIL
    numeric: Dict[str, Any]
    perceptual: Dict[str, Any]
    implicated_dial: str
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "implicated_dial": self.implicated_dial,
                "reasons": self.reasons, "numeric": self.numeric,
                "perceptual": self.perceptual}


def poll_judge(video_path: str, votes: int = 3, model: str = DEFAULT_MODEL,
               judge: Optional[GeminiJudge] = None) -> Dict[str, Any]:
    """Runs the perceptual judge `votes` times and returns a consensus."""
    judge = judge or GeminiJudge(model=model)
    ballots = [judge.judge(video_path) for _ in range(votes)]
    usable = [b for b in ballots if not b.get("skipped")]

    if not usable:
        reason = ballots[0].get("reason", "unknown") if ballots else "no ballots"
        return {"skipped": True, "reason": reason, "verdict": None, "votes": len(ballots)}

    verdicts = [b["verdict"] for b in usable]
    tally = Counter(verdicts)
    majority, majority_n = tally.most_common(1)[0]

    # Pick the dial from ballots that actually found a problem; "none" is not actionable.
    dials = [b["implicated_dial"] for b in usable if b["implicated_dial"] != "none"]
    dial = Counter(dials).most_common(1)[0][0] if dials else "none"

    return {
        "skipped": False,
        "verdict": majority,
        "implicated_dial": dial,
        "judge_agreement": majority_n / len(usable),
        "vote_tally": dict(tally),
        "votes_usable": len(usable),
        "votes_requested": votes,
        "mean_confidence": sum(b.get("confidence", 0) for b in usable) / len(usable),
        "ballots": usable,
    }


def run_qa(truth_dir: str, estimate_video: str, sidebyside_video: Optional[str] = None,
           votes: int = 3, model: str = DEFAULT_MODEL,
           judge: Optional[GeminiJudge] = None) -> QAResult:
    """Runs both checks and merges them into one verdict."""
    numeric = depth_metrics.evaluate(truth_dir, estimate_video)

    perceptual: Dict[str, Any] = {"skipped": True, "reason": "no side-by-side video supplied"}
    if sidebyside_video:
        perceptual = poll_judge(sidebyside_video, votes=votes, model=model, judge=judge)

    reasons: List[str] = []
    dial = "none"

    if numeric["verdict"] == "FAIL":
        verdict = "FAIL"
        reasons.extend(f"numeric: {f}" for f in numeric["failures"])
        # Numeric failure is geometric by construction, so bind the control harder unless
        # the judge names something more specific.
        dial = perceptual.get("implicated_dial", "none") if not perceptual.get("skipped") else "none"
        if dial == "none":
            dial = "control_strength"
    elif perceptual.get("skipped"):
        verdict = "MARGINAL"
        reasons.append(f"perceptual QA unavailable ({perceptual.get('reason')}) -- geometry "
                       f"verified but motion quality unchecked")
    elif perceptual["verdict"] == "PASS":
        verdict = "PASS"
    else:
        # Judge says MARGINAL/FAIL but geometry is sound: cap at MARGINAL, never FAIL.
        verdict = "MARGINAL"
        reasons.append(f"perceptual: {perceptual['verdict']} "
                       f"(agreement {perceptual['judge_agreement']:.0%})")
        dial = perceptual["implicated_dial"]

    for w in numeric.get("warnings", []):
        reasons.append(f"warning: {w}")

    return QAResult(verdict=verdict, numeric=numeric, perceptual=perceptual,
                    implicated_dial=dial, reasons=reasons)


def retry_loop(generate: Callable[[Dict[str, Any]], Dict[str, str]],
               qa: Callable[[Dict[str, str]], QAResult],
               params: Dict[str, Any],
               policy: RetryPolicy = RetryPolicy()) -> Dict[str, Any]:
    """
    generate(params) -> {"estimate_video":…, "sidebyside_video":…, …}
    qa(artifacts)    -> QAResult

    Both are injected so the loop is testable without spending anything. Stops on PASS, on
    attempt cap, on spend ceiling, or when the verdict implicates no actionable dial.
    """
    spent = 0.0
    history: List[Dict[str, Any]] = []
    current = dict(params)

    for attempt in range(1, policy.max_attempts + 1):
        if spent + policy.cost_per_attempt > policy.max_spend_units:
            history.append({"attempt": attempt, "stopped": "spend ceiling reached",
                            "spent": spent})
            break

        artifacts = generate(current)
        spent += policy.cost_per_attempt
        result = qa(artifacts)

        entry = {"attempt": attempt, "params": dict(current), "verdict": result.verdict,
                 "dial": result.implicated_dial, "reasons": result.reasons, "spent": spent}
        history.append(entry)
        print(f"[qa] attempt {attempt}: {result.verdict} "
              f"(dial={result.implicated_dial}, spent={spent}/{policy.max_spend_units})")

        if result.verdict == "PASS":
            return {"outcome": "PASS", "attempts": attempt, "spent": spent,
                    "final_params": current, "history": history}

        action = DIAL_ACTIONS.get(result.implicated_dial, {})
        if not action:
            entry["stopped"] = "no actionable dial -- needs human review"
            break

        # Apply the corrective change for the next attempt.
        if "vace_strength_delta" in action:
            current["vace_strength"] = round(
                min(2.0, current.get("vace_strength", 1.0) + action["vace_strength_delta"]), 3)
        if "denoise_delta" in action:
            current["denoise"] = round(
                max(0.05, min(1.0, current.get("denoise", 0.6) + action["denoise_delta"])), 3)
        if "steps_multiplier" in action:
            current["steps"] = int(min(40, current.get("steps", 6) * action["steps_multiplier"]))
        if action.get("add_canny_control"):
            current["add_canny"] = True
        entry["next_params"] = dict(current)
        print(f"[qa]   -> {action['why']}; next params: {current}")

    return {"outcome": "NEEDS_HUMAN_REVIEW", "attempts": len(history), "spent": spent,
            "final_params": current, "history": history}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combined geometry + perceptual QA gate.")
    parser.add_argument("--truth-dir", required=True)
    parser.add_argument("--estimate-video", required=True)
    parser.add_argument("--sidebyside-video", default=None)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_qa(args.truth_dir, args.estimate_video, args.sidebyside_video,
                    votes=args.votes, model=args.model)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        n, p = result.numeric, result.perceptual
        print(f"COMBINED VERDICT : {result.verdict}")
        print(f"implicated dial  : {result.implicated_dial}")
        print(f"-- numeric  : {n['verdict']} | corr {n['mean_correlation']:.3f} "
              f"jitter {n['correlation_jitter']:.3f} | aperture honoured "
              f"{n['aperture_honoured_fraction']} | {n['frames_verifiable']}/"
              f"{n['frames_compared']} frames verifiable")
        if p.get("skipped"):
            print(f"-- perceptual: SKIPPED ({str(p.get('reason'))[:100]})")
        else:
            print(f"-- perceptual: {p['verdict']} | agreement {p['judge_agreement']:.0%} "
                  f"{p['vote_tally']} | mean conf {p['mean_confidence']:.2f}")
        for r in result.reasons:
            print(f"   - {r}")
