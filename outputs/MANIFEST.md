# Manifest

What was generated, from what, and what it measured. The media files are gitignored; this is
the record that survives them.

## Source

One apartment from `uploads/*_floor-plans-estado-reformado.pdf`, page 1 — a vector floor plan
exported from AutoCAD. No DWG anywhere in this chain.

- rooms: living (GG), bedroom (D4), hall (HL), bathroom (A1)
- region: x 1.2–9.8 m, y 15.0–27.0 m, selected by position rather than by any id internal to
  the file
- scale: 36.1 mm per PDF point (1:100), agreed by all 11 pages within 5.5%, and independently
  corroborated by 278 doors of known width landing within 1–4 cm
- ceiling **2.70 m**, taken from the drawing's own note — the DWG carries exactly two `h=` notes
  for the whole building, reading 2.70 and 3.40. This is the only real vertical dimension in the
  source.
- window sill 0.90 m, head 2.10 m — **assumed**. The PDF states no sill or head height anywhere.

## Control tracks — `control/`

97 frames, 480×832, 16 fps, from `agent_b/real_samples/render_pdf_walkthrough.py`.

| file | what it is |
|---|---|
| `pdf_apartment_depth.mp4` | distance to every surface, one shared scale across the sequence |
| `pdf_apartment_edges.mp4` | edge map from depth and normal discontinuities |
| `pdf_apartment_clay.mp4` | shaded grey render — what render-to-real models expect |

Camera routed automatically through free space (`agent_b/route_planner.py`), minimum
clearance 0.40 m against a 0.30 m camera radius. Zero frames outside the model, zero with the
camera inside geometry.

## Generated — `generated/`

All four ran the same apartment, same camera, seed 18.

| file | generator | control fed | frames |
|---|---|---|---|
| `pdf_apartment_wan21vace.mp4` | Wan 2.1 VACE 14B (Comfy) | depth | 97 |
| `pdf_apartment_ltx3dreal_strong.mp4` | LTX 2.3 3DREAL (fal) | clay | 97 |
| `pdf_apartment_ltx3dreal_light.mp4` | LTX 2.3 3DREAL (fal) | clay | 97 |
| `pdf_apartment_cosmos_predict25.mp4` | Cosmos Predict 2.5 (fal) | clay | 93 |

### Measured

| generator | median Δ | mean Δ | max Δ | edge recall | first↔last |
|---|---|---|---|---|---|
| control (clay) | — | — | — | — | **45.0** |
| Wan 2.1 VACE | 11.21 | 14.87 | 47.23 | **0.346** | 34.7 |
| LTX 3DREAL strong | 17.13 | 20.45 | 62.67 | 0.194 | 58.5 |
| LTX 3DREAL light | 18.55 | 23.73 | 65.44 | 0.222 | 84.0 |
| Cosmos Predict 2.5 | **5.21** | **5.52** | **16.66** | 0.217 | **44.1** |

`Δ` is mean absolute frame-to-frame change; lower is steadier. `first↔last` is how different
the final frame is from the first, and the control's own value is the target — it says how far
the camera should have travelled. Reading it alongside Δ separates "smooth" from "barely
moving", which Δ alone cannot do.

### What the numbers say

**Cosmos Predict 2.5 is the steadiest by a wide margin** — a third of Wan's frame-to-frame
change — *and* its first-to-last difference of 44.1 is almost exactly the control's 45.0, so
the camera genuinely travels the right distance. Smooth without being static.

**Wan travels too little** (34.7 against a target of 45.0) but **follows the drawn lines
best** (0.346 edge recall, roughly half again the others).

**LTX overshoots**, and light intensity overshoots more than strong (84.0 vs 58.5) while
scoring no better on lines. Raising intensity did not trade accuracy for realism the way the
parameter name suggests.

### Caveats that matter

- Edge recall is **uncalibrated** and is measured against a control track the LTX and Cosmos
  runs never saw — they were fed the clay pass. Directional only.
- Cosmos produced 93 frames, not 97 (its documented maximum), so its clip is fractionally
  shorter and the comparison is not perfectly like-for-like.
- One run per configuration. No seed sweep.

## Not usable for this pipeline

**Cosmos 3 on fal** (`nvidia/cosmos-3-super/*`) takes only text or a single starting image —
its inputs are `prompt` and `image_url`, with no per-frame control. It would invent the camera
path and the geometry after the first frame, which is the one thing this pipeline exists to
prevent. Cosmos Transfer, which does accept control modalities, is not hosted on fal at all.

**LTX Union Control on Comfy** (`video_ltx2_3_ic_lora`) billed credits and returned no output.
Its published schema is misaligned (a field named `text`, typed STRING, defaulting to `1280`),
slot overrides were rejected twice, and its depth-estimation node could not be switched off —
so it was likely re-estimating depth from our depth map.

---

# Run-to-run variance — `variance/`

The question that prompted this: two runs of the same scene produce different furniture, so is
there any ground truth at all? Measured with `agent_b/qa/run_variance.py`, which compares two
generated clips against each other rather than against their control track — something no
existing check did.

Furnished apartment, living → bedroom, 97 frames.

| files | model | seeds |
|---|---|---|
| `wan_18.mp4`, `wan_18_repeat.mp4`, `wan_7.mp4` | Wan 2.1 VACE, depth control | 18, 18 again, 7 |
| `cosmos_18.mp4`, `cosmos_42.mp4`, `cosmos_7.mp4` | Cosmos Predict 2.5, clay control | 18, 42, 7 |

## Result

| comparison | structure | content |
|---|---|---|
| **Wan, seed 18, run twice** | **1.000** | **1.000** |
| same model, different seed (n=4) | 0.455 | 0.309 |
| across models (n=6) | 0.303 | 0.450 |

`structure` = do both put the architecture in the same place. `content` = do both contain the
same things, normalised against how much each clip moves so it compares across scenes.

**Determinism is solved.** Wan run twice on one seed gives a raw pixel difference of **0.00** —
every frame identical. The mp4s differ by MD5, but that is container timestamps and encoder
metadata, not pixels. Lock a seed per project and any regeneration reproduces exactly.

**Seed-independence is not solved and probably should not be.** Different seeds furnish and
light the rooms differently, because the drawing constrains the architecture and nothing
constrains the styling. There is no ground truth for what colour the sofa is.

Only the first property is needed for "the client sees the same flat every time".

## The apparent contradiction

Wan follows the drawing at 0.958 while two Wan runs agree with each other at 0.517. Both are
true. Edge recall asks whether a line exists near where the drawing says — a wall repainted a
different white still passes. Run-to-run agreement asks whether it is the same picture, which
that same repaint fails.

## Not established

- **No baseline before furniture was added**, so this cannot say whether furnishing the world
  improved reproducibility. The plan called for measuring first; the extractor was built first.
- Two Wan runs failed outright — `job_failed`, seeds 42 and 99, no error detail from Comfy — so
  the different-seed Wan figure rests on **one pair** against Cosmos's three.
- Whether Cosmos is also deterministic is unknown; its repeat run was still queued.

---

# v5 — the 14B model at 720p with sim2real

`generated/pdf_apartment_v5_wan14b_sim2real_720p.mp4`, Comfy job `85760217`, ~18 min.

Every earlier clip was **480x832 at 15 fps** against Seedance's 720x1280 at 24 — 2.4x the pixels
and 1.6x the frame rate. The quality gap that had been blamed on the model was partly a
comparison between a thumbnail and a photograph. The only saved VACE workflow in the account
loads `wan2.1_vace_1.3B` (a tenth the size of 14B), so which model the earlier apartment runs
used is **still unconfirmed**.

| | v5 |
|---|---|
| model | `wan2.1_vace_14B_fp16` (confirmed) |
| LoRA | `Wan21_14B_VACE_lora_ditto_sim2real_bf16.safetensors.safetensors` at 1.0 |
| control | semantic clay, 720x1280, 97 frames, every frame |
| seed / steps / cfg | 18 / 20 / 6, uni_pc + simple |
| VACE strength | 1.0, full denoising range — **still the default, never yet varied** |

Note the doubled `.safetensors.safetensors` in the LoRA filename; it is an upstream quirk and
must be passed verbatim.

## Result

Scored by `qa/measure_generated.py` (new — wraps `edge_overlay.compare_sequence` with frame
extraction, because the `edges_%04d` / `result_%04d` conventions are one apart and doing it by
hand silently scores frame N against N+1). All four re-scored against the **same** 720p control
at tolerance 5:

| clip | follows the drawing |
|---|---|
| v4 semantic clay (480p) | **0.906** |
| **v5 14B + sim2real (720p)** | **0.769** |
| v3 depth furnished (480p) | 0.711 |
| v3 Seedance (endpoints only) | 0.325 |

**Photorealism improved a lot; adherence dropped ~14 points.** Frame 72 has blown-out window
daylight, creased linen and a switch plate on the door reveal — a different class of image from
anything before it.

## Two measurement traps found here

**Tolerance is in pixels at the control's resolution**, so it is not comparable across control
sizes: the default 3 px on a 720-wide control is a 1.5x stricter test than the same 3 px on the
480-wide controls every figure above it in this file was scored against. `--tolerance 5` reads a
720p run on roughly the old footing. The old 480p control frames were overwritten by this
render, so earlier numbers cannot be reproduced as-measured — only re-derived by scoring the old
clips against the new control, which is what the table above does.

**A wrong hypothesis, recorded because it was cheap to test and would otherwise be repeated:**
that the sharper picture scores worse only because wood grain and fabric crowd out architectural
edges in the 92nd-percentile edge detector. Dropping to the 80th percentile lifted v5 by +0.075
and v4 by +0.071 — the same amount. The gap is real, not an artefact of sharpness.

## Not established

- **Three variables moved at once** (1.3B?→14B, 480p→720p, sim2real added), so the 14-point drop
  cannot be attributed. The add-on is the prime suspect on mechanism alone: its job is to push
  toward real-world detail, which is what a flat grey control does not contain.
- The comparison mildly **favours the older clip**, which is enlarged from 480p to be scored and
  therefore softer.
- One seed, one apartment, one drawing.
