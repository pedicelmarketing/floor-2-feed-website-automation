# HANDOFF — geometrically accurate marketing video from an architect's drawing

**Written for a model or engineer picking this project up cold.** Everything needed to continue
without re-deriving it. Last updated **2 Aug 2026**, branch `feature/anchor-frame-generation`,
15 commits, **nothing pushed**.

Read this file first, then `LEARNINGS.md` (what was established), then
`.claude/skills/running-comfy-cloud-workflows/SKILL.md` (how not to lose an afternoon on Comfy).

**If you are not Claude Code, read §10.3 before planning anything.** Generation runs through a
Comfy Cloud connection that only exists inside a Claude Code session. Every other part of this
project — rendering, measuring, scoring, judging — is local Python and runs anywhere.

---

## 1. The goal, verbatim

> Iterate across the multiple options — Comfy UI, previous assets and techniques — to achieve
> marketing-grade video that's geometrically accurate. The videos will be used in social media,
> so finding a good balance between architectural accuracy (millimetre) and visually striking
> would be a good way to go about it. Establish a judge using our Gemini account to assess the
> quality of the videos and the motion. Explore the multiple alternatives, including open source
> and frontier models. We already know for f2f and visuality Seedance and Google Omni are best,
> but not necessarily the best geometrically. Look at options such as workflow chaining,
> ControlNet chaining. Make sure you read well the documentation of the models in Comfy and in
> their official pages. Create a simple story or scene which can serve as an anchor for quality
> across experiments. Save all your findings in the skill MD and/or a learnings MD. Present each
> meaningful step as a constantly evolving Claude artifact that serves as historical record.

Plus a standing instruction added later: **don't stop until the goal is completed.**

### Standing user rules (these override defaults)

| Rule | Where it comes from |
|---|---|
| **Comfy only** for generation. Not fal, not Higgsfield. | Direct instruction: *"we are only using comfy remember"* |
| **Plain language** in anything user-facing — chat, artifacts, PR text. Define any term on first use. Never drop a caveat to simplify. | `AGENTS.md` |
| **Publish every result as a Claude artifact with the media embedded.** Never hand over a local file path. Republish the same file path to update in place. | `AGENTS.md` |
| Code comments and commit messages stay technical. | `AGENTS.md` |
| Don't discard Omni prematurely — it's a world model. *(This has now been tested to exhaustion; see §7.)* | Direct instruction |

**Millimetre accuracy is not reachable from this source and must not be claimed.** See §4.

---

## 2. Status in one paragraph

There are two things being measured: does the video match the architect's drawing, and is it
good enough to publish. **No clip has scored well on both.** The best compromise remains
`pdf_apartment_v5_wan14b_sim2real_720p.mp4` at **0.609 accuracy / 7.12 quality**, produced early
in this work; roughly ten rounds of experiments since have not beaten it. Along the way the
accuracy metric was found to be broken (it scored random noise at 0.99) and was fixed, which
invalidated the stated targets. **The central mechanism is understood and confirmed**: models
given the geometry on *every frame* follow the drawing; models given a start image, an end image
or a paragraph do not — four such clips score *below random noise*. The unresolved problem is
that the open models which accept per-frame control top out around quality 7, and the frontier
models that look beautiful cannot be constrained.

---

## 3. The two axes — how anything here is measured

Nothing in this project means anything without both numbers. Quoting one alone has misled this
project more than once.

### Axis A — adherence ("does it match the drawing")

`agent_b/qa/measure_generated.py`. Renders the drawing's edges, detects edges in the generated
video, reports what fraction of drawn edges have a generated edge within N pixels.

```bash
python3 agent_b/qa/measure_generated.py outputs/generated/<clip>.mp4 \
    --control agent_b/real_samples/output/pdf_walkthrough/frames \
    --tolerance 2
```

**Three things about this metric you must not forget:**

1. **Always read it against the null baseline.** The tool prints one on every run: the score
   random noise achieves against the same control at the same tolerance. A recall number quoted
   without its floor cannot be interpreted.

   | tolerance | correct render | wrong anchor | **random noise** |
   |---|---|---|---|
   | 1 px | 0.855 | 0.211 | **0.306** |
   | 2 px | 0.963 | 0.333 | 0.572 |
   | 3 px | 0.994 | 0.481 | 0.829 |
   | 5 px | 1.000 | 0.707 | **0.991** |

2. **Tolerance 5 is poison and was used for most of this project's history.** It was chosen
   deliberately so 720p figures would stay comparable with older 480p ones, and that
   comparability fix destroyed the metric's ability to discriminate. Default is now 2. Honest
   comparisons are quoted at 1 px.

3. **Tolerance is in pixels at the control's resolution**, so it is not comparable across
   control sizes. 3 px on a 720-wide control is a 1.5× stricter test than 3 px on a 480-wide one.

Also reported: `edge_recall_weighted`, weighting each frame by how much the control actually has
to say. Blank frames score ~1.000 whatever the model does and inflate plain means. The correction
turned out **small at clip level** (0.906→0.900, 0.769→0.793) and changed no ranking.

Frames are **resampled onto the control's timeline**, not paired by index — a 241-frame clip and
a 97-frame control cover the same camera move, so index *i* is a different *moment* in each.
Pairing by index reported Seedance at 0.325 when it was actually 0.409.

### Axis B — quality ("is it worth publishing")

`agent_b/qa/gemini_judge.py :: QualityJudge`. Uploads the clip to the Gemini Files API and asks
for a structured score. Model `gemini-3.6-flash`. Needs `GEMINI_API_KEY`.

Weighted: photorealism 0.30, lighting 0.25, composition 0.15, motion quality 0.18, temporal
stability 0.12. Also returns `marketing_grade`, `reads_as_real_footage`, `worst_defect`,
`single_biggest_fix`. The rubric pins what 0, 5 and 10 mean so scores don't drift.

**The judge is a sampler too.** Re-scoring one unchanged clip moved it 7.27 → 7.12. Treat
quality differences under ~0.2 as noise.

### Score everything at once

```bash
python3 agent_b/qa/score_library.py     # writes outputs/scores.json
```

⚠️ **`outputs/scores.json` currently holds STALE 5 px adherence values.** Its `quality` figures
are current; its `adherence` figures are the inflated ones. The honest 1 px numbers are in the
table below and in `LEARNINGS.md` §17. **Re-running `score_library.py` at the new default
tolerance is an outstanding job.**

---

## 4. The scoreboard

Adherence at **1 px**, null baseline **0.306**. Quality from the Gemini judge.

| clip | adherence | vs noise | quality | what it was |
|---|---|---|---|---|
| `v4_wan_semantic_clay` | **0.673** | above | 6.64 | Wan VACE, semantic clay control, every frame |
| **`v5_wan14b_sim2real_720p`** | **0.609** | above | **7.12** | **Wan 2.1 VACE 14B + sim2real LoRA — the best compromise** |
| `v12_ltx_depth_lotus` | 0.550 | above | 3.06 | LTX depth IC-LoRA, Lotus derives depth from our colour render |
| `v9_omni_edit_nopeople` | 0.518 | above | 6.01 | Gemini Omni, edit framing, no people |
| `v11_ltx_depth_iclora` | 0.464 | above | 4.17 | LTX depth IC-LoRA, our true ray-cast depth injected |
| `v10_ltx23_walkthrough_15s` | 0.240 | **below** | 2.55 | 4 anchors + first-to-last-frame interpolation |
| `v7_omni_editmode` | 0.185 | **below** | 8.42 | Omni edit, with children |
| `v3_seedance_bothends` | 0.178 | **below** | 6.66 | Seedance, first and last frame only |
| `v13_omni_dress_on_wan` | **0.138** | **below** | **8.07** | Omni asked only to restyle the good Wan clip |
| `v6_omni_addscene` | 0.116 | **below** | 8.67 | Omni, pure generation |

**The pattern is the whole project.** Everything above the noise floor was given geometry on
every frame. Everything below it was given endpoints, references or words. The two axes
anti-correlate almost perfectly: the most accurate clip is dull, the best-looking clip is a
different apartment.

**Targets: adherence 0.90 and quality 8.0, together. Neither has been met, and the 0.90 was set
on the broken 5 px metric — it is not a meaningful target any more.** The best honest adherence
ever observed on a *video* is 0.673. On single anchor *frames* it is 0.934 at 1 px.

---

## 5. The pipeline, end to end

```
architect's PDF  ──►  vector extract  ──►  rooms + walls + doors + windows + furniture
                                                      │
                                                      ▼
                                          watertight 3D mesh (metres)
                                                      │
                                       camera route through free space
                                                      │
                          ┌───────────────────────────┼───────────────────────────┐
                          ▼                           ▼                           ▼
                   depth track              shaded clay track              edge track
                          └───────────────────────────┼───────────────────────────┘
                                                      ▼
                                        Comfy Cloud generator (per-frame control)
                                                      │
                                    ┌─────────────────┴─────────────────┐
                                    ▼                                   ▼
                         adherence vs edge track                 Gemini quality judge
```

### The source

One apartment, page 1 of
`uploads/12f909d4-7599-4940-aa09-79108a7625d8_floor-plans-estado-reformado.pdf` — a vector plan
exported from AutoCAD. **No DWG is involved anywhere in this chain.**

- Rooms as the extractor names them: `GG` living, `D4` bedroom, `HL` hall, `A1` bathroom
- Region: x 1.2–9.8 m, y 15.0–27.0 m — selected by position, not by any internal file id
- Scale **36.1 mm per PDF point** (1:100), agreed by all 11 pages within 5.5%

### What the source can and cannot give — §4's honesty section

| dimension | status |
|---|---|
| Horizontal wall position | **verified 1–4 cm**, against 278 doors of known width |
| Ceiling height 2.70 m | the drawing's own note — the DWG carries exactly two `h=` notes for the whole building, 2.70 and 3.40 |
| Window sill 0.90 m, head 2.10 m | **assumed.** The PDF states no vertical dimension anywhere |
| Furniture heights | **assumed by class**, and flagged as such in the data |

⚠️ **`outputs/MANIFEST.md` says ceiling 2.60 m and calls it assumed. That is stale.** The code
(`pdf_blockout.ASSUMED_CEILING_HEIGHT_M`) uses **2.70**, sourced from the drawing's own note.
Fix the manifest when convenient.

### The mesh

`agent_b/pdf_blockout.py` — ~6,300 faces, watertight, walls with real thickness. Window openings
are cut **by banding, not by mesh boolean**: wall solid up to the sill, wall alone between sill
and head so the glazing gap stays open, lintel above. Floor and ceiling slabs are included
because without them a camera looking up or down sees into space.

Face materials are labelled: wall, ceiling, floor, door, window, furniture. Window labelling was
broken until recently (it matched **zero** faces); it now labels 110 faces across 3 openings by
testing wall-classified faces against the glazing footprints.

### Furniture — read this before "improving" it

`agent_b/furniture_volumes.py`. The drawing holds 731 furniture polylines inside this flat, of
which **only 18 close into a polygon**. The rest are cushion seams, drawer fronts and hatching —
symbols, not shapes. So lines are clustered by proximity (6 cm gap), each cluster's convex hull
is taken as a footprint, and it is extruded to a height chosen by layer and proportion.

**Result: 15 objects, 14.45 m².** That looks like catastrophic loss from 731 lines and **it is
not** — inspected cluster by cluster the objects are real: a 1.09 × 2.30 m bed, a 3.79 m shelf, a
2.11 × 0.64 m sideboard, sanitary fittings.

**An attempted improvement here was a regression.** Raising the area cap 8→16 m² recovered a
9.69 m² cluster, taking total footprint from 14.4 to 24.1 m² — a 67% gain by summary statistic,
and wrong. That object **contains the living room's centre point**: a 2.08 × 5.12 m solid block
in the middle of the room. It was caught not by inspection but by the route planner, because
every route in the flat went from clear to impossible.

Two lessons kept in the code comments:
- Bounding-box dimensions cannot validate a footprint. 2.08 × 5.12 m reads like a plausible
  fitted run. **Containment against known free space** is what exposed it.
- A summary statistic moving the right way is not evidence.

The guard is now two-shaped because furniture is: **COMPACT** up to 6 m² whatever its
proportions, or a **RUN** up to 16 m² but no deeper than 1.0 m. A sofa is ~0.9 m deep, a wardrobe
0.6 m, so anything both large and deep is a merge whatever its area.

**Consequence: furnishing is not the cheap win it looked like.** This flat really does hold about
14 m² of extractable furniture and no parameter tuning will conjure more. Making rooms feel
furnished must come from typed proxy geometry or the generative pass.

### The renderer

`agent_b/3d_room_builder.py` — ray-cast against the mesh with trimesh + Embree. Produces:

| pass | what it is | who wants it |
|---|---|---|
| **depth** | distance to every surface, one shared scale across the whole sequence | VACE, IC-LoRA depth |
| **clay** | shaded grey render with sun, sky fill, ambient occlusion | render-to-real models — *this is what they actually expect* |
| **edges** | discontinuities in depth and normals | the adherence metric |
| **material / semantic** | per-material tint or flat colour key | tested; see below |

**Lighting is a render decision, not a model decision.** The reference render was deliberately
flat for legibility, and a model steered by a flat evenly-lit reference returns a flat evenly-lit
photograph — that is the whole explanation for "accurate but lifeless". It now casts **one shadow
ray per pixel** with the sun placed from the actual glazing positions:
`inward = mesh.centroid[:2] - glazing[:2]`, `sun_dir = [inward*0.72, -0.69]`.

Two wrong turns worth not repeating:
- Physically sensible values (sun 1.55 / sky 0.42) rendered corridors at mean **38 of 255**. Most
  of a flat is never in direct sun, so **the fill is the picture**. Contrast belongs in the ratio,
  not in starving the shadows. Current: `SUN_STRENGTH 2.40`, `SKY_STRENGTH 1.00`.
- A true sky-blue fill turned every corridor to moonlight. Indoor bounce has already hit warm
  plaster and a wood floor before it arrives, so it lands near neutral —
  `SKY_FILL_RGB = (226, 229, 234)`.

**The flat colour-coded semantic pass is the worst control tested (0.528).** It carries perfect
information about what every surface *is* and none about light. These ControlNets read structure
through shading, not through labels.

### The camera

`agent_b/route_planner.py` plans through free space; furniture is included in the obstacle map.
Minimum clearance 0.40 m against a 0.30 m camera radius. Verified: zero frames outside the model,
zero with the camera inside geometry.

---

## 6. The anchor scene — the frozen contract

`agent_b/anchor_scene.py`. **Treat as frozen.** Before it existed, route, frame count, resolution
and scoring tolerance all moved between runs, and at least two "findings" turned out to be the
setup shifting rather than a result.

| | value |
|---|---|
| version | 2 |
| route | `GG` living → `D4` bedroom |
| frames / fps | 241 @ 24 = 10.0 s (4n+1, so Wan takes it without resampling) |
| resolution | 720 × 1280, a true 9:16 |
| eye height / FOV | 1.60 m / 70° |
| seed | 18 |
| prompt | style only — *"Photorealistic interior photograph of a modern Spanish apartment. Matte white plaster walls, pale oak plank flooring, soft natural daylight. Architectural photography, sharp focus, no people."* |

Chosen for what it lets you **measure**: it stays inside the furnished part of the flat, it
contains a doorway (the recurring failure), and it travels far enough to expose mid-clip drift.

**v1 was retired after one run.** Scanned frame by frame it held a feature only in frames 0–30
and was 93% bare wall by the end. Blank frames fail twice: the model decorates them (one v1
anchor came back covered in invented vertical panelling stripes, scoring 0.373 against
0.995–1.000 for its neighbours) and the metric cannot see it, because a frame carrying 0.16% edge
pixels scores near 1.000 whatever happens. **Choose anchor positions by information content, not
by even spacing** — `scene_description.describe_frame` reports doors, windows, furniture and
material fractions per frame, so bad anchors are detectable before a credit is spent.

⚠️ **`BASELINE_ADHERENCE = 0.898` and `TARGET_ADHERENCE = 0.90` in this file are stale** — both
were set from the broken 5 px metric. Recalibrating them to the 1 px metric is an outstanding
job. `BASELINE_QUALITY = 7.27` should probably be 7.12 (the re-score).

Render it:
```bash
python3 agent_b/real_samples/render_anchor.py
```

---

## 7. What has been established, and what is closed

`LEARNINGS.md` holds the full record with numbers. Condensed, in order of how much it should
change what you do next.

### Confirmed mechanisms

**1. Per-frame control is the mechanism.** LTX's own depth IC-LoRA gives it a depth frame for
every output frame instead of two endpoints:

| LTX configuration | adherence @1px | vs null 0.306 | quality |
|---|---|---|---|
| 4 anchors, first-to-last-frame | 0.240 | **below** | 2.55 |
| per-frame, our true depth injected | 0.464 | +0.158 | 4.17 |
| per-frame, colour in, Lotus derives depth | **0.550** | **+0.244** | 3.06 |

Per-frame control roughly doubles adherence and lifts LTX from below chance to clearly above it.
The mid-clip collapse disappears entirely: the endpoint clip scored 1.00 at anchors and 0.22
between; the IC-LoRA runs stay flat at 0.21–0.66 with no dip.

**2. Endpoint interpolation is the wrong architecture, and anchor spacing is not the knob.**
Halving the anchor gap from 5 s to 3 s moved a clip from 0.491 to 0.524 and quality 2.55 to 2.88
— marginal on both axes. The anchors themselves are near-perfect: on anchor v2 they reproduce
**93.4% of the drawing's edges within a single pixel**. Everything between them is invention, and
no achievable anchor density fixes it.

**3. ControlNets chain, and chaining beats any prompt change.** `QwenImageDiffsynthControlnet`
takes a MODEL and returns a MODEL, so patches stack. `WanVideoVACEEncode` chains the same way via
`prev_vace_embeds`.

| lever | effect on adherence |
|---|---|
| **chaining two structural controls** (clay + depth) | **+0.098**, better on all three seeds |
| render type (clay vs depth vs flat colour) | 0.294 spread |
| seed | 0.200 spread — noise, not a lever |
| prompt wording | +0.014 — nothing |

Structural + structural helps. Structural + appearance (tile) gained +0.026, inside noise.

**4. Geometry beats language, always.** A door leaf is ~4 cm thick against a 0.54–9.26 m depth
range quantised to 256 levels — **the whole door fits inside one grey level of the wall behind
it**. Painting a door onto a blank wall costs the model nothing. No prompt wording prevents it.
Putting the thing in the mesh does. Same lesson as furniture.

**5. Determinism differs by model and matters commercially.** Wan on a fixed seed reproduces
**pixel for pixel** — the mp4s differ by MD5 only through container timestamps. Gemini Omni is
non-deterministic by Google's own statement: identical inputs scored 0.384 and 0.344. Anything
Omni touches is a **one-off master to archive**, never something regenerable from inputs.

Seed-independence is *not* solved and probably shouldn't be — different seeds furnish and light
rooms differently because the drawing constrains architecture and nothing constrains styling.
There is no ground truth for what colour the sofa is. Only determinism is needed for "the client
sees the same flat every time".

### Closed questions — do not re-litigate these

**Gemini Omni cannot be used as a finishing pass.** This was the fair test the user specifically
asked for. Omni was given the *best available input* — the Wan clip at 0.609/7.12 — with an
explicitly modest ask: keep every wall, window and camera move, change only surfaces and light,
add a throw and one plant.

| | adherence @1px | quality |
|---|---|---|
| the Wan input | **0.609** | 7.12 |
| after Omni's dressing pass | **0.138** | **8.07** |

**It made the best-looking clip in the library and pushed the geometry below the noise floor**,
from a correct starting point, while being told not to. Its adherence trend across earlier runs
(0.116 → 0.185 → 0.518 as the ask shrank) looked like it might extrapolate; given the best
possible input it went the other way. Omni is a generator whose output happens to be beautiful,
and it rewrites the room whatever it is given.

**A second reference image imports geometry, not style.** Nano Banana given a clay render plus a
photograph of another room, told explicitly to take only materials and light from it, scored
0.454 — worse than the clay render alone at 0.667. The corridor stretched and a unit appeared.
For an editor, every reference image is a claim about what the scene *looks like*, shape included.
There is no prompt wording that fixes this.

**Higgsfield adds nothing on geometry.** Searched its catalogue for depth, pose, per-frame or
structural conditioning: **zero results.** It is a wrapper over the same frontier
reference-conditioned APIs already measured below the noise floor (Seedance 2.0, Kling 3.0, Omni,
Grok Video). It *does* offer `upscale_video` (Topaz or ByteDance, with interpolation to 24/30/60
fps), which is the one quality lever that carries no accuracy risk — by then the wall is pixels.
**Blocked: account balance is 0.09 credits.** Also out of scope under the Comfy-only rule unless
the user reverses it.

**Cosmos 3 on fal has no per-frame control** — inputs are `prompt` and `image_url` only. Cosmos
Transfer, which does accept control modalities, is not hosted on fal at all. (Moot under
Comfy-only, recorded so it isn't rediscovered.)

### Wrong hypotheses, recorded so they aren't repeated

- **"The sharper picture scores worse because wood grain crowds out architectural edges."** No.
  Dropping the edge detector from the 92nd to the 80th percentile lifted the sharp clip by +0.075
  and the soft one by +0.071 — the same amount. The gap is real, not an artefact.
- **"Adding truthful scene facts to the prompt helps."** An apparent +0.161 on one frame became
  **+0.014** (spread 0.057, better in two of four) across fresh camera/seed pairs.
- **"Injecting our true ray-cast depth must beat an estimator."** It didn't — 0.464 vs 0.550.
  Estimated-from-colour looks more like the depth the LoRA was trained on than a ray-cast map
  does. Injecting raw depth also drags the *output* toward grey; the judge called it "unfinished
  monochrome clay render appearance". The model copies the guide's look, not only its shape.

### **Single runs are not measurements.**

Changing only the seed moves adherence by up to **0.21**. Any A/B here needs at least three
seeds, and paired-by-seed comparison beats comparing means.

---

## 8. File map

### Core pipeline — `agent_b/`

| file | what it does |
|---|---|
| `pdf_vector.py` | extract vector geometry from the PDF by layer |
| `layer_conventions.py` | which layer names mean walls, doors, glazing, furniture |
| `wall_regions.py` | recover rooms; `rooms_for_page` |
| `room_extractor.py` | room labelling and areas |
| `door_pairing.py` | pair door symbols to openings; the 278-door accuracy check |
| `pdf_blockout.py` | **the mesh.** `blockout_from_page`, `wall_polygons`, banded window openings, face materials |
| `furniture_volumes.py` | cluster symbol linework into extruded objects (see §5) |
| `route_planner.py` | camera path through free space; `plan_route`, `to_waypoints` |
| `camera_paths.py` | path shaping and `describe` |
| `3d_room_builder.py` | **the renderer.** `render_camera_path(..., tint_map, semantic, exposure, sun_dir)`, `sun_visibility`, `_shade_from_normals` |
| `scene_description.py` | truthful per-frame description from the semantic pass + real metric facts |
| `anchor_scene.py` | **the frozen shot.** All anchor constants live here |
| `comfy_ui_client.py` | thin Comfy helper |
| `dwg_*.py` | DWG route — **not used by the current pipeline** |

### Measurement — `agent_b/qa/`

| file | what it does |
|---|---|
| `measure_generated.py` | **adherence.** Wraps `edge_overlay`, resamples onto the control timeline, computes the null baseline |
| `edge_overlay.py` | `compare_sequence`, `_directed_hit_rate`, `_edges_from_render` |
| `gemini_judge.py` | **quality.** `QualityJudge`, rubric, schema, weights |
| `score_library.py` | score every clip on both axes → `outputs/scores.json` |
| `run_variance.py` | compare two *generated* clips against each other (reproducibility) |
| `depth_metrics.py` | depth correlation against control |
| `opening_count.py` | count doors/windows in output |
| `stage_classifier.py` | Gemini vs Cosmos Reason harness |
| `qa_runner.py` | orchestration |

### Rendering entry points — `agent_b/real_samples/`

| file | what it does |
|---|---|
| `render_anchor.py` | **render the frozen anchor scene** — the one to use |
| `render_ground_truth.py` | sunlit reference renders |
| `render_pdf_walkthrough.py` | the 97-frame control track most figures are scored against |
| `measure_pdf_accuracy.py` | the door-width accuracy check |
| `output/` | gitignored; thousands of PNGs and raw depth arrays |

### Records

| file | what it holds |
|---|---|
| `LEARNINGS.md` | 20 numbered findings, each with its numbers |
| `outputs/MANIFEST.md` | what was generated, from what, with which settings, and what it measured |
| `outputs/README.md` | naming convention and why media isn't committed |
| `.claude/skills/running-comfy-cloud-workflows/SKILL.md` | **Comfy operational traps — read before touching Comfy** |
| `HANDOFF.md` | this file |

### Scratchpad (session-local, not committed)

`/tmp/claude-1000/-home-openclaw-.../scratchpad/`

| file | what it does |
|---|---|
| `mk_ltx.py` | LTX first-to-last-frame graph surgery — patches **internal** nodes 215/216/198/205/196/222 |
| `mk_d2v.py` | LTX depth-to-video — bypasses the Lotus estimator, fixes portrait, fixes length |
| `build_v5.py` | the artifact for the best Wan run |
| `build_log.py` | the running artifact (see §11) |
| `build_grid.py`, `build_omni.py` | comparison artifacts |
| `ltx_*.json` | fetched and patched subgraph blueprints |

---

## 9. Operational traps

The full list is in `.claude/skills/running-comfy-cloud-workflows/SKILL.md`. **These are the ones
that cost the most time here.**

### Comfy

- **A job can succeed and return nothing.** `Preview*` nodes render to the web UI only.
  `get_output` returns files for `SaveText`, `SaveImage`, `VHS_VideoCombine`. Results are under
  **`results`**, not `outputs`: `get_output(id)["results"][0]["url"]`. URLs are short-lived.
- **`run_template` slot_overrides and input_overrides do not work on subgraph templates.** Tested
  on four (Z-Image, Qwen, LTX FLF2V, LTX depth) — all returned `validation.reference`. The
  blueprint also can't be used as a `class_type`.
  **The working path:** fetch from `/api/global_subgraphs/<name>`, edit the JSON, POST to userdata,
  `run_saved_workflow`.
- **Widget values on a subgraph *instance* are ignored.** You must patch the nodes *inside* the
  subgraph definition. This produced a landscape LTX render twice.
- **`get_template_schema` returns defaults rotated by one slot** (by two where image inputs carry
  no widget). `shown_default[i] == real_default[i+1]`.
- **Never strip `pos`/`size`/`flags`/`order`** from save-format JSON — they're load-bearing for
  the save→API converter, and `get_saved_workflow` parses the stripped version happily.
- **`dry_run` proves less than it looks.** It catches unknown `class_type` and bad model dropdown
  values. It does *not* catch a missing required input, a wrong dotted sub-field, or a nonexistent
  filename. `status: "validated"` can arrive alongside real errors — branch on `warnings`.
- **Nodes with dotted input names.** `GeminiNodeV2`'s video input is `model.video.video_1`, not
  `video`. Always `get_node` first.
- **`pending` means queued, not executing.** `wait_for_job` reports `polling` regardless.
  Cancel-and-resubmit does **not** jump a queue — measured.
- **Step-distillation LoRAs (CausVid, Lightning) exist to make FEW steps work**, at cfg 1. Going
  6→20 steps measured 55% *worse* temporal stability.

### The Lotus trap (specific and expensive)

The stock `depth_to_video_ltx_2_0` subgraph runs a **Lotus depth estimator** over its input video.
Feeding it our ray-cast depth makes it compute *the depth of a depth image*. This is the exact
trap that wasted credits on LTX Union Control earlier. `mk_d2v.py` bypasses it by rewiring links
with `origin_id == 190` → `189`.

Counter-intuitively, **leaving Lotus in and feeding it our colour render scored better** (0.550 vs
0.464). Both configurations are worth keeping.

### Gemini / Omni

- **Omni refuses any video containing the generated children.** Isolated by elimination over six
  runs — same clip at 10 and 24 fps, with and without audio, all failed with children; identical
  format without people went through. The API returns no reason. **Practical consequence: do
  iterative passes while the scene is empty, and add people last.**
- **`GeminiVideoOmni` refuses a second video** — `model.videos.video_2` fails even though the
  schema advertises three slots.
- **`gemini-2.5-flash` returns 404 "no longer available to new users"** while still being listed
  by the models endpoint. **A listing is not an entitlement.**
- **Partner models silently 404 on valid-looking parameters.** `Nano Banana 2 Lite` failed with no
  error text on 9:16 + 2K + `thinking_level: HIGH`, and succeeded on auto + 1K + MINIMAL. When a
  partner node fails with `error_type: unknown`, strip optional parameters to defaults first.
- **"Nano Banana 3 Lite" does not exist.** Only `nano-banana-2` and `nano-banana-pro`; the node
  exposes "Nano Banana 2 Lite".

### Shell

Earlier `cd` commands left the working directory inside `outputs/` and `agent_b/real_samples`,
producing false "file missing" results. **Use absolute paths.**

---

## 10. Credentials, environment and tooling

### 10.1 Where things live

| what | where |
|---|---|
| repo root (git) | `/home/openclaw/floor-2-feed-website-automation` |
| **working directory for everything below** | `/home/openclaw/floor-2-feed-website-automation/floor-2-feed-website-automation` |
| branch | `feature/anchor-frame-generation`, at `c81a115`, **15 commits, nothing pushed** |
| Python | `venv/bin/python` inside the working directory, **Python 3.12.3, already active** — plain `python3` resolves to it |
| Node | v22.23.1, npm 10.9.8 (`npx` available) |
| dependencies | `requirements.txt`. Every comment in it explains *why* a pin exists — read before upgrading anything |
| media | gitignored (`outputs/**/*.mp4`, `*.png`, `*.jpg`). `MANIFEST.md` is the record that survives the files |

Two pins that will break the pipeline silently if moved: **trimesh 4.12.2** (4.0.8 breaks under
numpy 2) and **embreex** (the fast ray-casting backend — without it a frame takes 115 seconds
instead of 0.17, a 94-minute render instead of an 8-second one; trimesh falls back quietly rather
than erroring).

`opencv` is **not** installed and nothing here needs it. Gemini uses the current
`google-genai` package, **not** the retired `google.generativeai` one.

### 10.2 Secrets

All keys live in **`/home/openclaw/.config/secrets.env`**, chmod 600, **outside the repo**. Keys
present, by name only:

`MINIMAX_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `NVIDIA_API_KEY`, `FAL_KEY`,
`HOSTINGER_API_TOKEN`, `CLOUDFLARE_API_TOKEN`

Load with `set -a; . /home/openclaw/.config/secrets.env; set +a`. Only `GEMINI_API_KEY` is
required by anything currently in use (the quality judge). `FAL_KEY` and `NVIDIA_API_KEY` belong to
earlier experiments the user has since ruled out — see the Comfy-only rule in §1.

**Never** commit these, echo them into an artifact, print them to the terminal, or include them in
a commit message. Comfy upload commands embed **short-lived bearer credentials** — run them exactly
as emitted, never log or share them.

### 10.3 The MCP connections — and what breaks without Claude Code

Generation runs through **Comfy Cloud over MCP** (`mcp__claude_ai_Comfy__*`). This is the single
biggest thing to understand before switching tools.

These are **claude.ai hosted connectors**, authenticated through the user's Claude account, not
local processes with an API key in a config file. They are not declared in any `.mcp.json` — there
is no `.mcp.json` in this repo at all, and `~/.claude.json` declares no MCP servers for this
project. They arrive with the Claude Code session and disappear with it.

Connectors this project has used:

| connector | used for | status |
|---|---|---|
| `claude.ai Comfy` | **all video and image generation** | disconnected as of this writing |
| `claude.ai Higgsfield-Pedicel` | assessed and rejected (§7) | disconnected |
| `claude.ai Gmail / Calendar / Drive / Notion` | unrelated to this project | available |
| Academic Research | unrelated | available |

**Consequence for any non-Claude-Code agent (Kimi Code, or anything launched via `npx`): you
cannot generate.** Everything else in this repo works unchanged — the mesh builder, the ray-cast
renderer, the route planner, the adherence metric, the null baseline, the Gemini quality judge
(plain HTTPS with `GEMINI_API_KEY`), the scoring library and the artifact builder are all local
Python with no MCP dependency. But the step that turns a control track into a video is a Comfy
Cloud call, and that call is only reachable from a session where the Comfy connector is live.

Practical split of work:

- **Do in any agent:** rendering control tracks, measuring, re-scoring, fixing the four stale
  items in §12, reading and extending `LEARNINGS.md`, building comparison pages, pushing the
  branch.
- **Needs a Claude Code session with the Comfy connector connected:** generating any new clip,
  including the two untried levers in §13 (Wan 2.2 Fun VACE, and varying the VACE strength dials
  that have never once moved off their defaults).

If the connector shows as disconnected, it is reconnected from the Claude.ai connector settings,
not from a file in this repo. Do not try to reconstruct it as a local `npx` MCP server — it is not
one, and there is no local Comfy install here.

### 10.4 The other Claude Code features this repo leans on

If you move to a different agent, these have no equivalent and you should know what you are giving
up rather than discovering it mid-task:

- **Skills** — `.claude/skills/running-comfy-cloud-workflows/SKILL.md` is loaded automatically in
  Claude Code. Elsewhere it is just a Markdown file: **read it manually**, it is the accumulated
  list of Comfy traps and each one cost real time to find.
- **Artifacts** — the standing rule in §11 is that every result is published as a Claude artifact
  with the media embedded. That publishing step is a Claude Code tool. Another agent can still
  *build* the page (`scratchpad/build_log.py` produces it) but cannot publish it, so the running
  record at the §11 link will go stale. Say so plainly to the user rather than substituting a file
  path — a local path cannot be opened from their side, which is the whole reason the rule exists.
- **`CLAUDE.md` / `AGENTS.md`** — the project rules are in `AGENTS.md` at both the repo root and
  the working directory; `CLAUDE.md` merely includes it. Most agents read `AGENTS.md`, so the rules
  travel. **Read both**: the root one carries only the Next.js warning, the working-directory one
  carries the plain-language and artifact rules that actually govern this work.

---

## 11. The artifact — the required deliverable

The user's standing instruction: **every result worth reporting is published as a Claude artifact
with the media embedded**, never described in chat and never handed over as a local file path.

- **Running record:** https://claude.ai/code/artifact/e7e10d77-187b-4c00-95bc-c3f9689201a3
- Built by `scratchpad/build_log.py`. **Republish the same file path** so the shared link stays
  current rather than going stale beside a newer one.
- Videos are embedded as base64 data URIs — a strict CSP blocks every external host.
- The artifact carries the caveats too: what was assumed, what is uncalibrated, what a number does
  not establish.

---

## 12. Known-stale and outstanding

| item | state |
|---|---|
| `outputs/scores.json` adherence column | **stale** — 5 px values. Re-run `score_library.py` at tolerance 2 or 1 |
| `anchor_scene.py` `BASELINE_ADHERENCE` / `TARGET_ADHERENCE` | **stale** — calibrated on the broken metric. Best honest video score is 0.673, target says 0.90 |
| `anchor_scene.py` `BASELINE_QUALITY = 7.27` | should be 7.12 (the re-score) |
| ~~`LEARNINGS.md` section numbering~~ | ~~duplicate §15 and §16~~ — **fixed**, now runs 1–20 |
| ~~`outputs/MANIFEST.md` ceiling height~~ | ~~said 2.60 m assumed~~ — **fixed**, now 2.70 m from the drawing's note |
| Branch `feature/anchor-frame-generation` | **15 commits, nothing pushed** |
| Comfy Cloud connector | **disconnected** as of 2 Aug 2026 — no new clip can be generated until it is reconnected (§10.3) |
| Null baseline quoted as both 0.303 and 0.306 | same measurement, different sample draws — harmless, but pick one |

---

## 13. What to try next, ranked

**The wall this project is against:** open models that accept per-frame control top out around
quality 7; frontier models that look beautiful cannot be constrained. Everything below is an
attempt to move one of those two facts.

### 1. Wan 2.2 Fun VACE — the strongest untried generative lever

Per-frame conditioning on a model that already reaches quality 7+. This is the only remaining
candidate that could plausibly move *both* axes, because it combines the mechanism that works
(per-frame control) with the model family that scores best on looks. **Not yet run.**

### 2. Vary the VACE dials — never once varied

Every Wan run used `strength 1.0` over the full denoising range, the default. `vace_start_percent`
and `vace_end_percent` are the dials for *how hard* geometry is bound. Releasing control in the
late denoising steps is the standard way to let a model add realism after the structure is fixed —
and it has never been tested here. **Cheapest experiment with a real mechanism behind it.**

### 3. Resolution and upscaling

Everything generated is 704–720 wide. The judge's complaint about the best clip was not geometry
but *"tight vertical crop limits room context during the rapid pan"* — framing and resolution.
**An upscaler cannot move a wall**, so this is the only quality gain with zero accuracy risk.
Comfy has upscale nodes; Higgsfield has Topaz but is out of scope and out of credits.

### 4. Typed furniture proxies

The extractor is right and this flat holds ~14 m² of furniture. Making rooms feel furnished has to
come from typed proxy geometry — a bed-shaped mesh where the drawing says bed — not from tuning
the extractor. This attacks the finding in §6 that **any large surface carrying no information gets
furnished by the model**.

### 5. A second architect's drawing

Every number in this project comes from one apartment, one drawing, one practice's symbol
conventions. Nothing here has been shown to generalise.

### The alternative worth pricing

**Render-then-restyle.** Build the scene properly in Blender / V-Ray / Twinmotion, render it
accurately, and use the generative pass only for finishing. This trades the whole geometry problem
for production cost. The user has been asked about this and **has not yet answered** — it remains
the open strategic decision.

---

## 14. Reproduce it

```bash
cd /home/openclaw/floor-2-feed-website-automation/floor-2-feed-website-automation

# 1. render the frozen anchor scene (mesh, route, all control passes)
python3 agent_b/real_samples/render_anchor.py

# 2. generate — via Comfy MCP. The reliable path:
#    fetch /api/global_subgraphs/<name>  ->  patch INTERNAL nodes  ->
#    POST to userdata  ->  run_saved_workflow
#    See scratchpad/mk_d2v.py and mk_ltx.py for worked examples.

# 3. measure adherence, always with the null baseline
python3 agent_b/qa/measure_generated.py outputs/generated/<clip>.mp4 --tolerance 1

# 4. measure quality
python3 -c "
from agent_b.qa.gemini_judge import QualityJudge
print(QualityJudge().score('outputs/generated/<clip>.mp4'))
"

# 5. re-score everything
python3 agent_b/qa/score_library.py

# 6. check reproducibility between two generated clips
python3 agent_b/qa/run_variance.py <clip_a>.mp4 <clip_b>.mp4
```

---

## 15. Rules of engagement for whoever continues

1. **Never quote an adherence number without its noise floor.** This project's single worst error
   was a metric that scored random noise at 0.99, reported as success for weeks.
2. **Never quote one axis alone.** They anti-correlate. A win on accuracy that wrecks quality is
   not a win, and vice versa.
3. **Three seeds minimum before an A/B means anything.** Seed alone moves adherence by 0.21.
4. **Verify a summary statistic with an independent check.** Footprint area rose 67% and the
   geometry got worse; the route planner caught what the statistic hid.
5. **Look at the image.** The ambient-occlusion, camera and furniture changes were all wrong on
   the first attempt and only the picture showed it.
6. **Record failures with their numbers**, in `LEARNINGS.md`. Half the value here is knowing what
   not to try again.
7. **Publish as an artifact, with the media embedded and the caveats attached.**

---

### Changelog for this file

- **1 Aug 2026** — created. Covers everything through the Omni finishing-pass test (LEARNINGS §20)
  and the Higgsfield assessment. Flags four stale items and two documentation inconsistencies
  found while writing it.
- **2 Aug 2026** — §10 rewritten for a handover to a different agent (Kimi Code). Adds the exact
  paths, the Python and Node versions, the dependency pins that fail silently, the names of the
  keys in the secrets file, and — the important part — that Comfy generation runs over a Claude
  account connector that no other agent can reach, with the work split into what can and cannot be
  done without it. Records the connector as currently disconnected.
