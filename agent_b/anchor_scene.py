"""
THE ANCHOR SCENE. One fixed shot, unchanged, in every experiment from here on.

Why this file exists at all: nothing measured so far is strictly comparable to anything else.
The camera route changed between runs, the frame count changed, the resolution changed, the
control render changed, and the tolerance the scores were computed at changed with it. Some of
those changes were improvements; all of them broke comparability, and a couple of "findings"
turned out to be the setup moving rather than the result. Pinning ONE scene is the cheapest
fix available and costs nothing but the discipline of not editing this file.

Treat these constants as frozen. If a change is genuinely needed, bump ANCHOR_VERSION and
re-score the library rather than silently invalidating every number already recorded.

THE SHOT
--------
Hall into the living room, through the doorway the drawing actually puts there. It was chosen
over the prettier bedroom view for three reasons that matter for scoring rather than for looks:

  - it CONTAINS a doorway, so door invention -- the failure mode that recurs across every model
    tried -- is visible in the frame rather than something you have to go looking for;
  - it passes a window, so cast sunlight and the "is this lit or is it flat" question are
    testable in the same shot;
  - it travels far enough to expose mid-clip drift. A short hop hides the exact failure that
    first-to-last-frame generation is worst at.

10 seconds at 24 fps because that is what a social feed wants and what the frontier models take
natively. Portrait, for the same reason.
"""

ANCHOR_VERSION = 1

# Rooms as the extractor names them: HL hall, GG living, D4 bedroom, A1 bathroom.
ROUTE = ["HL", "GG"]

# 241 frames at 24 fps is 10.0 s. 4n+1 so the clip can still be handed to Wan, which requires
# it, without resampling and therefore without a second variable creeping in.
FRAME_COUNT = 241
FPS = 24

# Portrait, and a real 9:16 rather than the 0.577 the earlier 480x832 renders used. Anything
# generated at another aspect has to be letterboxed or cropped to be scored against this, and
# a crop already cost one whole 15-second run in this project.
WIDTH, HEIGHT = 720, 1280

EYE_HEIGHT_M = 1.60
FOV_DEG = 70.0

# The same apartment, region and scale every other measurement in this repo used.
PDF = ("/home/openclaw/floor-2-feed-website-automation/uploads/"
       "12f909d4-7599-4940-aa09-79108a7625d8_floor-plans-estado-reformado.pdf")
PAGE = 0
MM_PER_PT = 36.1
REGION_M = (1.2, 15.0, 9.8, 27.0)

# Fixed so "same seed" means something across models that honour one. Models that do not --
# Gemini Omni states results vary regardless of seed -- are simply not reproducible, and that
# is a property of the model to report rather than a setting to fix.
SEED = 18

# The one prompt. Style only: measured across 8 cells, adding a truthful scene description moved
# adherence by +0.014 on average with a spread of 0.057, so the extra words buy nothing and
# would only add a variable.
PROMPT = ("Photorealistic interior photograph of a modern Spanish apartment. Matte white "
          "plaster walls, pale oak plank flooring, soft natural daylight. Architectural "
          "photography, sharp focus, no people.")

# What a run has to beat to count as progress. Both, not either.
#   adherence -- edge recall against the drawing at 5 px on a 720-wide control. Best measured
#                so far is 0.898 (clay + depth chained, mean of three seeds).
#   quality   -- the weighted Gemini score. Best measured so far is 7.27 (Wan 14B + sim2real),
#                which the judge also called publishable at 7/10.
# Neither is calibrated in any absolute sense; they are only meaningful against each other and
# against these baselines.
BASELINE_ADHERENCE = 0.898
BASELINE_QUALITY = 7.27

# A run that improves one while wrecking the other is not an improvement. This is the bar.
TARGET_ADHERENCE = 0.90
TARGET_QUALITY = 8.0


def summary() -> str:
    return (f"anchor v{ANCHOR_VERSION}: {'->'.join(ROUTE)}, {FRAME_COUNT} frames @ {FPS} fps "
            f"({FRAME_COUNT / FPS:.1f} s), {WIDTH}x{HEIGHT}, seed {SEED}\n"
            f"  baselines: adherence {BASELINE_ADHERENCE}, quality {BASELINE_QUALITY}\n"
            f"  targets:   adherence {TARGET_ADHERENCE}, quality {TARGET_QUALITY}")


if __name__ == "__main__":
    print(summary())
