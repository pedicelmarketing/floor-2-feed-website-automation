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
- ceiling 2.60 m, window sill 0.90 m, head 2.10 m — **all assumed**. The PDF states no
  vertical dimension anywhere.

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
