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

---

Comfy-specific operational traps are in
`.claude/skills/running-comfy-cloud-workflows/SKILL.md`, not here.
