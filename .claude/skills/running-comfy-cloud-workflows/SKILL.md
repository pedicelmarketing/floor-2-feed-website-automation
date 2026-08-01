---
name: running-comfy-cloud-workflows
description: Use when building, editing, saving, submitting or debugging ComfyUI / Comfy Cloud workflows via MCP — including getting text or files back out of a finished job, a job that succeeded but returned no output, a saved workflow that suddenly fails validation, wiring a node whose inputs have dotted names, stacking two control signals, picking a Gemini or partner-API model, or a job stuck in pending.
---

# Running Comfy Cloud workflows

Reference for the parts of Comfy Cloud that are expensive to rediscover. Claims below were
verified against the live service; anything unverified is marked as such.

## Getting output back out of a job

**A job can succeed and return nothing.** This is the most common way to lose an afternoon.

| Node | `get_output` returns | Use for |
|---|---|---|
| `SaveText` | ✅ a real `.txt`/`.md`/`.json` | getting **text** back programmatically |
| `SaveImage`, `VHS_VideoCombine` | ✅ image / video file | images, video |
| `PreviewAny`, `PreviewImage`, any `Preview*` | ❌ **nothing** — renders to the web UI only | eyeballing in the UI, never automation |

**Do not conclude "Comfy can't return text" when `PreviewAny` fails.** That inference is
wrong and has been made in this codebase before. `SaveText` is a core node and works.

### The exact `get_output` response shape

Results are under **`results`**, not `outputs`. Verbatim, from a real run — copy this shape,
do not guess it:

```json
{"results": [{"source_node_id": "1", "class_type": "SaveText",
              "filename": "ad7339fa….txt",
              "url": "https://cloud.comfy.org/api/s/nWJFaOQn3UqFEHKlY6PH_g?raw=1",
              "inline_url": "https://storage.googleapis.com/…"}]}
```

```python
url = get_output(prompt_id)["results"][0]["url"]   # NOT ["outputs"][0]
verdict = requests.get(url).text                   # round-trips byte-identical
```

URLs are **short-lived** — fetch immediately, never cache. An expired one 404s; re-call
`get_output` with the same `prompt_id` to mint a fresh link.

## Nodes with dotted input names (dynamic combos)

Some nodes hide their real inputs behind a selector. `GeminiNodeV2`'s video input is **not**
a flat `video` field — it is `model.video.video_1`, because `model` is a dynamic dropdown
whose sub-fields hang off it. Send it flat and you get `required_input_missing`, which reads
like the field is absent rather than misnamed.

**Always `get_node` before wiring one of these**, and send sub-fields with the exact dotted
prefix. `dry_run` will not catch a wrong prefix (see below), so this survives validation and
fails only on a paid run.

## dry_run — free, but proves less than it looks

`submit_workflow(dry_run: true)` costs nothing and does not execute. Always use it before a
paid submit. Then read it correctly:

| Catches | Does **not** catch |
|---|---|
| unknown `class_type` → `error_type: validation.reference` | a **missing required input** |
| bad value in a *model* dropdown | a **wrong dotted sub-field** (`model.video.videoclip_99`) |
| | a **nonexistent filename** in a file dropdown |

**`status: "validated"` can arrive alongside real errors.** A bad model value returns
`validated` *plus* an `input_validation` warning — it does not block submission. Branch on
the `warnings` array, never on `status` alone, or you will pay for a broken graph.

A dry-run PASS confirms node names exist. It does not confirm your inputs are wired.

## Editing and saving a graph

Two formats, not interchangeable:

- **API format** — `{node_id: {class_type, inputs}}` → `submit_workflow`
- **Save/graph format** — top-level `nodes` array → `save_workflow` → `run_saved_workflow`

**Never strip "cosmetic" fields from save format.** `pos`, `size`, `flags`, `order` are
load-bearing for the save→API converter. Removing them to save context yields
`validation.schema` at run time, while the identical graph with them intact runs fine — and
`get_saved_workflow` parses the stripped version happily, so it looks valid.

**Large graphs: bypass your own context.** A 20 KB+ workflow need not pass through the
conversation to be saved:

```bash
curl -X POST "https://cloud.comfy.org/api/userdata/workflows%2Fmy-flow.json?overwrite=true" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  --data-binary @my-flow.json      # then: run_saved_workflow(filename="my-flow.json")
```

`$TOKEN` comes from the command `upload_file` emits (it embeds a short-lived credential).

## Benign noise — do not chase

- `"video"/"image" value "<hash>.mp4" was not found in the bundled node index` — fires on
  every uploaded file. The local catalog doesn't know your upload; the cloud does.
- `Node N (LoadImage): 1 extra widget values not mapped` — harmless.
- The **paid-API-node warning fires on `dry_run` too**. Nothing was spent; `submitted: false`.
- `deprecated` means superseded, **not** removed. Deprecated nodes still execute. Prefer the
  current node for new graphs; don't panic-rewrite a working one.
- `wait_for_job` returning `timed_out: true` is normal — it means "call me again", not failure.

## Stacking two control signals

`WanVideoVACEEncode` outputs `WANVIDIMAGE_EMBEDS` and also accepts one as
`prev_vace_embeds`, so encoders **chain**: encode depth, feed that into a second encode
carrying canny, point the sampler at the last. Verified end to end. Its `strength`,
`vace_start_percent` and `vace_end_percent` are the dials for how hard geometry is bound.

## Choosing a model — two separate decisions

**Don't hardcode model names from any doc, including this one — they age out. `get_node`
the node for its current list.**

1. **Via a Comfy partner node** (bills Comfy credits at a markup). The node's dropdown uses
   human-readable labels like `Gemini 3.5 Flash`, not API slugs like `gemini-2.5-flash`.
   Passing a slug yields a `not found in the bundled node index` warning.
2. **Via the provider's own API** from your process (bills your own quota, cheaper). Here you
   use real API slugs, and availability differs from the node's list — e.g. `gemini-2.5-pro`
   returns `429 … limit: 0` on a free-tier key while `gemini-2.5-flash` works. List them:
   `GET generativelanguage.googleapis.com/v1beta/models?key=…`

**Prefer (2) for anything your code must branch on** — it returns a native object instead of
a job handle, poll loop, expiring link and file download, and supports structured-output
schemas. Use (1) when the step must live inside the same graph, or there's no provider key.

**Step-distillation LoRAs (CausVid, Lightning) exist to make FEW steps work** and run at
`cfg 1`. Adding steps fights them: 6→20 steps on such a graph measured **55% worse**
temporal stability. "More steps" is not a free quality win.

## Jobs that sit in `pending`

`pending` means **queued, not executing**. `wait_for_job` hides this: it reports its own
`raw_status: "polling"` regardless, so a job waiting behind someone else's looks identical
to one actively generating. Call `get_job_status` for the real state before concluding
anything about progress.

Diagnose with `get_queue`, and read it as a **rate**, not a level:

| `running` over several minutes | Meaning | Action |
|---|---|---|
| changes | Jobs are completing; you are in a normal queue | Wait |
| pinned at the same nonzero value | Another job holds the slot and is not finishing | See below |
| `0` while yours stays `pending` | Your submission never reached the queue | Cancel and resubmit |

**Cancel-and-resubmit does not jump a queue.** An earlier version of this file claimed the
resubmit "typically runs immediately". Measured against the live service: with `running: 1`
held constant for ~15 minutes by an unrelated job, cancelling a `pending` job and
resubmitting the identical workflow produced a *new* job that also sat in `pending`, and the
queue counts did not move. The remedy only helps the third row above, where the submission
itself was lost — it cannot evict whatever occupies the slot.

`get_queue` is account-wide, so the occupying job may belong to another session or an
earlier run of yours that never terminated. There is no MCP call that reveals which
`prompt_id` is running; you can only cancel jobs whose id you already hold. If the slot is
held by something you cannot identify, waiting is the only option from here — say so rather
than resubmitting repeatedly, since each attempt just lengthens the queue.

## Common mistakes

| Mistake | Consequence | Fix |
|---|---|---|
| `Preview*` for automated output | job succeeds, `get_output` empty | `SaveText` / `SaveImage` |
| Reading `get_output()["outputs"]` | KeyError | it is `["results"][0]["url"]` |
| Flat input on a dynamic-combo node | `required_input_missing` on a paid run | `get_node`, use dotted prefix |
| Branching on `status` alone | broken graph submitted and billed | read the `warnings` array |
| Minifying save-format JSON | `validation.schema` at run time | keep `pos`/`size`/`flags`/`order` |
| Trusting dry-run to prove wiring | unwired inputs reach a paid run | it checks names, not wiring |
| Caching an output URL | 404 later | re-call `get_output` |
| Raising steps on a distill-LoRA graph | worse output, credits spent | change a different dial |

## run_template's slot_overrides do not work on subgraph templates

Tested on three: `image_z_image_turbo_fun_union_controlnet`, `image_qwen_Image_2512_controlnet`,
`video_ltx2_3_flf2v`. All three returned `error_type: validation.reference` for addresses taken
straight out of `get_template_schema`. Nothing was billed, but nothing ran either.

The subgraph blueprint cannot be rescued by calling it as a node, either: putting
`controlnet_z_image_turbo` in as a `class_type` in an API-format workflow also returns
`validation.reference`. The blueprint only exists inside the cloud graph format.

**Hand-build from primitives instead.** `search_nodes` on the template's description lists the
nodes it bundles ("Built from: CLIPLoader, UNETLoader, ... QwenImageDiffsynthControlnet ..."),
`get_node` gives their exact inputs, and the resulting graph passed `dry_run` first time and ran.
That path took three calls; fighting the template took six and produced nothing.

## get_template_schema returns defaults ROTATED BY ONE SLOT

Not corrupt -- shifted. On `image_z_image_turbo_fun_union_controlnet` the `image` slot showed a
prompt string, the `text` slot showed a seed, the `seed` slot showed a checkpoint filename. The
rule is `shown_default[i] == real_default[i+1]`, and it held on `video_ltx2_3_flf2v` too.

This is worth reading rather than dismissing: rotating it back is how the real model filenames
were recovered (`z_image_turbo_bf16.safetensors`, `qwen_3_4b.safetensors`, `ae.safetensors`,
`Z-Image-Turbo-Fun-Controlnet-Union.safetensors`) without opening the workflow JSON.

## ControlNets chain, and chaining is worth more than any prompt change

`QwenImageDiffsynthControlnet` takes a MODEL and returns a MODEL, so patches stack: load two
`ModelPatchLoader`s, apply the first to one control image, feed its MODEL output into the second
applied to a different control image, point the sampler at the last. Verified end to end.

Measured on an architectural anchor frame, 3 seeds each: clay alone 0.800, clay + depth chained
**0.898** (better on all three seeds), clay + tile-material chained 0.825. The same experiment
found prompt wording worth +0.014.

A working Z-Image-Turbo + Fun Union graph: UNETLoader(z_image_turbo_bf16) -> ModelPatchLoader ->
QwenImageDiffsynthControlnet(image=control) -> ModelSamplingAuraFlow(shift 1.73) -> KSampler
(8 steps, cfg 1, euler/simple) with CLIPLoader(qwen_3_4b, type `qwen_image`),
VAELoader(ae.safetensors), ConditioningZeroOut as the negative, and EmptySD3LatentImage sized
from GetImageSize. There is no `z_image` CLIP type; `qwen_image` is correct.

## Partner models silently refuse valid-looking parameters

`GeminiNanoBanana2V2` at `Nano Banana 2 Lite` failed with NO error text on
9:16 + 2K + `thinking_level: HIGH`, and succeeded immediately on auto + 1K + MINIMAL. The graph
passed `dry_run` both times. When a partner node fails with `error_type: unknown`, strip the
optional parameters back to defaults before assuming the input is at fault.

`GeminiVideoOmni` refuses a second video: `model.videos.video_2` fails even though the schema
advertises three slots.

**Omni also refuses video input containing children.** Isolated by elimination over six runs --
the same clip at 10 fps and at 24 fps, with and without audio, all failed when it contained the
generated children; the identical format carrying no people went through. Stated as inference,
since the API returns no reason, but the control is tight. Practical consequence: do iterative
passes while the scene is empty, and add people last.

## A second reference image imports geometry, not just style

Nano Banana given a clay render plus a photograph of a different room, and told in the prompt to
use the second image "only for materials, colour and light", scored **0.454** against the drawing
-- worse than the same clay render alone at 0.667. The corridor stretched and a unit appeared.
There is no prompt wording that makes an editor take only the palette from a reference.

## Single runs are not measurements here

Changing only the seed moved edge recall by up to **0.21** on identical inputs. An apparent
+0.161 from a prompt change evaporated to +0.014 across four fresh camera/seed pairs. Any A/B in
this pipeline needs at least three seeds before it means anything, and paired-by-seed comparison
is far more informative than comparing means.
