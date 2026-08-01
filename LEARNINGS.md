# Learnings

What this project has actually established, and what it only appears to have established.
Ordered by how much it should change what you do next.

Running record, kept current: https://claude.ai/code/artifact/e7e10d77-187b-4c00-95bc-c3f9689201a3

## 1. There are two axes, and for most of this project only one was measured

Every number recorded before 1 Aug 2026 -- edge recall, depth correlation, run variance,
opening counts -- answers one question: do the walls land where the architect drew them. None
answered whether the result is worth publishing. Adherence was driven from 0.24 to 0.90 while
the reference render was deliberately kept flat *for legibility*, and a model steered by a flat
evenly-lit reference returns a flat evenly-lit photograph.

`qa/gemini_judge.py :: QualityJudge` is the second axis. Scored across the whole library, the
two axes anti-correlate:

| run | follows drawing | quality | marketing |
|---|---|---|---|
| Wan, semantic clay | **0.906** | 6.64 | 5 |
| **Wan 14B + sim2real** | **0.769** | **7.12** | 5 |
| Omni, edit, furniture | 0.719 | 6.01 | 5 |
| Seedance, both ends | 0.409 | 6.66 | 4 |
| Omni, edit, children | 0.384 | 8.42 | 8 |
| Omni, generate | 0.243 | **8.67** | **9** |

The most accurate clip is mediocre to look at. The best-looking clip is a different flat.
**Nothing has yet cleared 0.90 and 8.0 together, and nothing is close.**

## 2. Single runs are not measurements

Changing only the random seed moves edge recall by up to **0.21**. An apparent +0.161 gain from
adding a truthful scene description to the prompt evaporated to **+0.014** (spread 0.057, better
in two of four) when repeated on fresh cameras and seeds.

Any A/B here needs at least three seeds, and paired-by-seed comparison beats comparing means.
The judge is a sampler too -- re-scoring one clip moved it 7.27 to 7.12, so treat quality
differences under ~0.2 as noise.

## 3. What actually moves adherence, in order

| lever | effect |
|---|---|
| **Chaining two structural controls** (clay + depth) | **+0.098**, better on all three seeds |
| Render type (clay/material vs depth vs flat colour) | 0.294 spread |
| Seed | 0.200 spread -- noise, not a lever |
| Prompt wording | +0.014 -- nothing |

Chaining works because `QwenImageDiffsynthControlnet` takes a MODEL and returns one, so patches
stack. Structural + structural helps; structural + appearance (tile) gained +0.026, inside noise.

**The flat colour-coded pass is the WORST input tested (0.528).** It carries perfect information
about what every surface *is* and none about light. These ControlNets read structure through
shading, not through labels.

## 4. Control every frame, or accept drift

The 15-second LTX walkthrough scores **1.00 at its four anchors and 0.22 between them**. Anchors
are perfect because they *are* the picture; five seconds is too far to leave a model unsupervised.
Anchor spacing is the knob. Per-frame control (`ltx-2-19b-ic-lora-depth-control`) is untested.

Models given geometry every frame obey it; models given a picture or a paragraph cannot -- not
because they are worse, but because nothing in their input carries a measurement.

## 5. Depth cannot forbid a door

A door leaf is ~4 cm thick against a 0.54-9.26 m depth range quantised to 256 levels: the whole
door fits inside one grey level of the wall behind it. Painting a door onto a flat wall costs the
model nothing, and a blank corridor wall is implausible to anything trained on real interiors.

Language does not fix this (see 2). **Geometry does** -- put the thing in the model and the
control carries it. Same lesson as furniture.

## 6. A second reference image imports geometry, not style

Nano Banana given a clay render plus a photograph of another room, told explicitly to take only
materials and light from it, scored **0.454** -- worse than the clay render alone at 0.667. The
corridor stretched and a unit appeared. No prompt wording fixes it; for an editor every reference
image is a claim about what the scene looks like, shape included.

## 7. Lighting is a render decision, not a model decision

The reference render was flat on purpose so a person could read it. That is why output was
"accurate and lifeless". It now casts one shadow ray per pixel with the sun placed from the
actual glazing positions.

Two wrong turns worth keeping:
- Physically sensible values (sun 1.55 / sky 0.42) rendered corridors at mean **38 of 255**.
  Most of a flat is never in direct sun, so the fill IS the picture. Contrast belongs in the
  ratio, not in starving the shadows.
- A true sky-blue fill turned every corridor to moonlight. Indoor bounce has already hit warm
  plaster and a wood floor before it arrives, so it lands near neutral.

## 8. What the source can and cannot give

Horizontal accuracy is **1-4 cm**, checked against 278 doors of known width, with the scale
agreed by all 11 pages to within 5.5%. Ceiling height 2.70 m is the drawing's own note. **Sill
and head heights are assumed** -- the PDF states no vertical dimension anywhere.

Millimetre accuracy is not reachable from this source and should not be claimed.

## 9. Determinism differs by model, and it matters commercially

Wan on a fixed seed reproduces **pixel for pixel** -- the mp4s differ by MD5 only through
container timestamps. Gemini Omni is **non-deterministic by Google's own statement**: the same
prompt, input and seed scored 0.384 and 0.344. Anything Omni touches is a one-off master to
archive, never something regenerable from inputs.

## 10. The anchor scene

Nothing measured before 1 Aug was strictly comparable to anything else -- route, frame count,
resolution and scoring tolerance all moved between runs, and at least two "findings" were the
setup shifting rather than a result. `agent_b/anchor_scene.py` freezes one shot: hall to living
room, 241 frames at 24 fps, 720x1280, seed 18. It contains a doorway (the recurring failure), a
window (the open question), and enough travel to expose drift.

Baselines to beat: **adherence 0.898, quality 7.27**. Target: **0.90 and 8.0 together**.

## 11. A featureless control invites decoration

Anchor v1, frame 120: the control is a bare wall filling the frame -- 54% wall, zero doors,
windows or furniture. Z-Image returned the wall covered in **vertical panelling stripes**, and
adherence collapsed to 0.373 while its four neighbours scored 0.995-1.000.

This is lesson 5 generalised. It is not specifically about doors: **any large surface carrying
no information gets furnished by the model**, because a blank plane is implausible to anything
trained on real interiors. Depth cannot forbid it and language does not fix it.

Practical consequence: **choose anchor positions by information content, not by even spacing.**
`scene_description.describe_frame` already reports doors, windows, furniture and material
fractions per frame, so bad anchors are detectable before a single credit is spent.

## 12. Anchor v1 is a bad shot, and the reason is upstream

Scanning the frozen shot frame by frame: only frames 0-30 contain any feature at all. From
frame 40 to the end it is blank wall -- frame 240 is 93% wall. The shot walks out of the
interesting part of the flat and down a corridor.

That is not really a framing mistake. **The 3D world is under-furnished**: 15 extruded boxes
standing in for 731 furniture polylines in the drawing. Most views are empty because most of the
model is empty. Furnishing is therefore not a cosmetic step -- it is what gives the controls
something to say.

Anchor v2 should route through the furnished rooms rather than between them.

## 13. Sparse edge maps flatter the score, but not by much

Control edge density on the anchor shot averages 0.33% of pixels, ranging 0.16% (blank wall) to
1.00% (doorway plus windows plus furniture). On a 0.16% frame there is almost nothing to match,
so recall lands near 1.000 whatever the model does -- frames 60, 180 and 240 all scored 1.000
that way.

`measure_generated` now also reports `edge_recall_weighted`, weighting each frame by how much
the control actually has to say. **Checked against the existing library, the correction is
small**: 0.906 -> 0.900, 0.769 -> 0.793, 0.407 -> 0.401. No ranking changes and no headline
figure was materially inflated. The flaw is real at the level of individual frames and modest at
the level of clips -- worth having, not worth rewriting history over.

## 14. Sunlit reference: first result

Five anchors generated from the newly sunlit render with clay + depth chained, seed 18:
0.995, 1.000, **0.373**, 0.997, 1.000. The outlier is the blank-wall frame from 11 above.
Excluding it the shot holds essentially perfectly; including it the mean is 0.873.

Whether the sunlight improves the QUALITY axis is still unmeasured -- the clip has not been
assembled or judged yet. That is the next measurement, not a claim.

## 15. The furniture extractor was already right, and "731 polylines to 15 objects" is not a bug

It looks like catastrophic loss and it is not. Inspected cluster by cluster, the 15 objects are
real: a 1.09 x 2.30 m bed, a 3.79 m shelf, a 2.11 x 0.64 m sideboard, sanitary fittings. The
1909 A-MOB entries include hatching, cushion lines and drawer fronts -- many lines per object is
the normal shape of CAD furniture symbols, not a failure.

**An attempted improvement here was a regression, caught by an unrelated check.** Raising the
area cap from 8 to 16 m2 recovered a 9.69 m2 cluster and took the footprint from 14.4 to 24.1 m2
-- a 67% gain by the summary statistics, and wrong. Testing containment afterwards showed that
object swallows the living room's CENTRE POINT: a 2.08 x 5.12 m solid block sitting in the
middle of the room. The route planner found it before any render did, because every route in the
flat went from clear to impossible.

Two things worth keeping from that:

- **Bounding-box dimensions are not enough to validate a footprint.** 2.08 x 5.12 m reads like a
  plausible fitted run. Containment against known free space is what exposed it.
- **A summary statistic moving in the right direction is not evidence.** Footprint went up 67%
  and the geometry got worse. The independent check -- can a camera still walk through the flat
  -- is what caught it.

The guard is now two-shaped, because furniture is: COMPACT up to 6 m2 whatever its proportions
(bed, sofa, table), or a RUN up to 16 m2 but no deeper than 1.0 m (fitted units, worktops). A
sofa is about 0.9 m deep and a wardrobe 0.6 m, so anything both large and deep is a merge
whatever its area. Same 15 objects as before, on a rule that states why.

**Consequence for the plan:** furnishing is NOT the cheap win it looked like. The flat really
does hold about 14 m2 of extractable furniture, and no amount of parameter tuning will conjure
more from this drawing. Making the rooms feel furnished has to come from somewhere else --
typed proxy geometry, or the generative pass -- not from the extractor.

## 16. Anchor spacing is not the knob, and endpoint interpolation loses to per-frame control

The 15-second LTX clip scored 1.00 at its anchors and 0.22 between them, which read like a
spacing problem. It is not. Halving the gap from 5 s to 3 s and re-running on a better shot
moved the whole clip from 0.491 to **0.524** -- and the quality judge from 2.55 to **2.88**.
Both axes agree the change was marginal.

The anchors themselves are not the problem. On anchor v2 they reproduce **93.4% of the drawing's
edges within a single pixel** (1.000 at 5 px, 0.998 at 3, 0.985 at 2, 0.934 at 1 -- the metric is
nowhere near saturated, it is genuinely that accurate). Everything between them is invention.

**The comparison that matters:**

| approach | follows drawing | quality |
|---|---|---|
| Wan VACE, control every frame | 0.769 | **7.12** |
| Anchors + LTX first-to-last-frame | 0.524 | 2.88 |

**Per-frame control beats endpoint interpolation decisively on BOTH axes.** The anchor-and-
interpolate architecture built over several rounds is worse than the approach already in hand.
The judge names the mechanism without being told it: "visible temporal morphing where wall
textures change and outline artifacts flicker during the pan."

That is worth stating plainly because it reverses a plan: first-to-last-frame is attractive
because stills are cheap and controllable, and the anchors really are near-perfect. It does not
matter. What happens between them is unsupervised, and no achievable anchor density fixes it.

Next test is therefore per-frame control on the good input, not more anchors.

## 15. The metric scored random noise at 0.99, and I chose the setting that did it

Tested by scoring pure random noise against the same control, which nothing in this project had
ever done:

| tolerance | correct render | wrong anchor | **random noise** |
|---|---|---|---|
| 1 px | 0.855 | 0.211 | **0.306** |
| 2 px | 0.963 | 0.333 | 0.572 |
| 3 px | 0.994 | 0.481 | 0.829 |
| **5 px** | 1.000 | 0.707 | **0.991** |

**At 5 px, noise scores 0.991.** The cause is structural: `_edges_from_render` always marks the
top 8% of gradients, and 8% coverage inside an 11x11 neighbourhood almost always contains a hit.
Separation between a correct and a wrong render is 0.29 at 5 px and 0.64 at 1 px.

5 px was **my** choice, made deliberately so 720p numbers would stay comparable with older 480p
ones. That comparability fix quietly destroyed the metric's ability to discriminate, and every
adherence figure quoted between then and now was computed at it.

Re-measured at 1 px, the ranking survives but the picture is far starker:

| clip | 5 px | **1 px** | vs noise floor 0.306 |
|---|---|---|---|
| Wan, semantic clay | 0.906 | **0.673** | above |
| Wan 14B + sim2real | 0.769 | **0.609** | above |
| Omni, edit, furniture | 0.719 | **0.518** | above |
| LTX 15 s | 0.407 | 0.240 | **below noise** |
| Omni, edit, children | 0.384 | 0.185 | **below noise** |
| Seedance, both ends | 0.409 | 0.178 | **below noise** |
| Omni, generate | 0.243 | 0.116 | **below noise** |

**Four of seven clips score below what random noise achieves.** This does not weaken the central
finding -- it sharpens it. Per-frame control clears the floor comfortably; endpoint- and
reference-conditioned generation does not clear it at all.

`measure_generated` now computes and prints the null baseline on every run, and the default
tolerance is 2. A recall number that is not quoted against its noise floor cannot be read.

## 16. Anchor v2 measured

Five anchors, clay + depth chained, seed 18, against the v2 sunlit reference: **1.000 at every
anchor at 5 px** -- which is exactly the saturation above, since the null is 0.991 there. At an
honest tolerance the anchors still lead the library, but "perfect" was an artefact of the
setting, not a result. Recorded as a caution: a clean sweep of 1.000 is a reason to check the
metric, not to celebrate.

## 17. Per-frame control is confirmed as the mechanism -- on a model that cannot draw

LTX's own depth IC-LoRA gives it a depth frame for every output frame instead of two endpoints.
Built by tracing `depth_to_video_ltx_2_0` and patching it: the stock graph runs a **Lotus depth
estimator** over the input video, so feeding it our ray-cast depth would have made it compute the
depth of a depth image -- the exact trap that wasted credits on LTX Union Control months ago.

| LTX configuration | adherence @1px | vs null 0.303 | quality |
|---|---|---|---|
| 4 anchors, first-to-last-frame | 0.240 | **below** | 2.55 |
| per-frame, true depth (Lotus bypassed) | 0.464 | +0.161 | 4.17 |
| per-frame, colour in, Lotus derives depth | **0.550** | **+0.247** | 3.06 |

**Per-frame control roughly doubles adherence and lifts LTX from below chance to clearly above
it.** The mid-clip collapse disappears: FLF2V scored 1.00 at anchors and 0.22 between, while the
IC-LoRA runs flat at 0.21-0.66 with no dip.

Two things worth keeping:
- Letting Lotus derive depth from our COLOUR render beat injecting our true depth (0.550 vs
  0.464). Estimated-from-colour looks more like the depth the LoRA was trained on than a
  ray-cast map does. The "obviously better" input was worse.
- Injecting raw depth drags the OUTPUT toward grey. The judge called it "unfinished monochrome
  clay render appearance" -- the model copies the guide's look, not only its shape.

LTX-2 19b tops out around quality 3-4 either way, against Wan's 7.12. The mechanism is right and
the model is not.

## 18. Omni cannot be used as a finishing pass, and this was the fair test

The remaining hope for Omni was that it had never been given a good input. So it was given the
best one available -- the Wan clip at 0.609 adherence and 7.12 quality -- with an explicitly
modest ask: keep every wall, window and camera move, change only surfaces and light, add a throw
and one plant.

| | adherence @1px | quality |
|---|---|---|
| Wan 14B + sim2real, the input | **0.609** | 7.12 |
| after the Omni dressing pass | **0.138** | **8.07** |

Scored against the same control, at the same tolerance. **It made the best-looking clip in the
library and pushed the geometry below the noise floor**, from a correct starting point, while
being told not to.

That closes the question. Omni is not a finishing pass over geometry that matters. It is a
generator whose output happens to be beautiful, and it rewrites the room whatever it is given
and whatever it is told. Its adherence trend (0.243 -> 0.384 -> 0.719 as the ask shrank) looked
like it might extrapolate; given the best possible input it went the other way.

---

Comfy-specific operational traps are in
`.claude/skills/running-comfy-cloud-workflows/SKILL.md`, not here.
