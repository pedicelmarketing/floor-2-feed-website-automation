"""
Classify what construction stage a room photo shows, via any vision model, for comparison.

Built to answer one question with evidence rather than preference: for "is this room at
rough-in, plastered, or finished?", is a specialist model such as NVIDIA's Cosmos Reason
actually better than the Gemini already wired into this project and running on cheaper
credits? Both backends therefore return the SAME schema against the SAME prompt, so the only
variable is the model.

Two backends, because the two services speak different protocols:

  gemini      Google's own SDK, with a response schema enforced server-side.
  openai      Any OpenAI-compatible chat endpoint. This covers build.nvidia.com, which hosts
              nvidia/cosmos-reason2-8b -- confirmed present in its model list. It needs an
              Authorization header; a keyless request is refused with "Header of type
              `authorization` was missing", so NVIDIA_API_KEY must be set to use it.

ON LABELS, WHICH IS WHERE THIS GETS DANGEROUS. A model's accuracy is only as meaningful as
the ground truth it is scored against. Wikimedia Commons categories were tried as a free
source and are NOT usable: a photo drawn from a timber-framing category turned out to be a
furnished historic attic, complete with stove and armchairs. Scoring against labels like that
produces a percentage that looks like evidence and is not. `compare` therefore reports raw
agreement between models and, separately, agreement with whatever labels were supplied --
never a single "accuracy" figure that hides which is which.
"""
import base64
import json
import os
from typing import Any, Dict, List, Optional

STAGES = ["structure", "rough_in", "plastered", "finished"]

PROMPT = (
    "You are inspecting a photograph of a room in a building, for a construction progress "
    "report that releases a stage payment.\n\n"
    "Classify the room into exactly one stage:\n"
    "  structure  - bare structural shell. Concrete or blockwork, no internal partitions.\n"
    "  rough_in   - partitions framed but open. Studs, joists or services visible; no wall "
    "boards.\n"
    "  plastered  - walls closed and boarded or plastered, but unfinished. No final paint, "
    "no flooring, no fittings.\n"
    "  finished   - decorated and habitable. Finished floor, painted walls, fittings, and "
    "usually furniture.\n\n"
    "Judge only what is visible. If the room is finished but old or in poor repair, it is "
    "still 'finished' -- age is not an earlier stage."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "stage": {"type": "string", "enum": STAGES},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": ["stage", "confidence", "evidence"],
}


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def classify_gemini(image_path: str, model: str = "gemini-2.5-flash") -> Dict[str, Any]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"skipped": True, "reason": "GEMINI_API_KEY not set"}
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=_read(image_path), mime_type="image/jpeg"),
                      PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "stage": {"type": "STRING", "enum": STAGES},
                        "confidence": {"type": "NUMBER"},
                        "evidence": {"type": "STRING"},
                    },
                    "required": ["stage", "confidence", "evidence"],
                }),
        )
        return {"skipped": False, "model": model, **json.loads(response.text)}
    except Exception as exc:                     # noqa: BLE001 - an outage is not a verdict
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}


def classify_openai_compatible(image_path: str,
                               model: str = "nvidia/cosmos-reason2-8b",
                               base_url: str = "https://integrate.api.nvidia.com/v1",
                               api_key_env: str = "NVIDIA_API_KEY") -> Dict[str, Any]:
    """
    Any OpenAI-shaped chat endpoint. Used for Cosmos Reason via build.nvidia.com.

    No response_schema here -- the endpoint does not enforce one the way Gemini does -- so the
    prompt asks for JSON and the reply is parsed leniently. That difference is itself worth
    recording when comparing: a model that will not reliably emit parseable output costs
    engineering effort that raw accuracy does not show.
    """
    import urllib.error
    import urllib.request

    key = os.environ.get(api_key_env)
    if not key:
        return {"skipped": True, "reason": f"{api_key_env} not set"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT + "\n\nReply with JSON only: "
                                              '{"stage": ..., "confidence": ..., "evidence": ...}'},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(_read(image_path)).decode()}},
        ]}],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
        text = body["choices"][0]["message"]["content"]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return {"skipped": True, "reason": f"no JSON in reply: {text[:120]}"}
        return {"skipped": False, "model": model, **json.loads(text[start:end + 1])}
    except Exception as exc:                     # noqa: BLE001
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}


BACKENDS = {"gemini": classify_gemini, "cosmos": classify_openai_compatible}


def compare(images: List[Dict[str, Any]], backends: List[str] = None) -> Dict[str, Any]:
    """
    Run each backend over each image and report where they agree.

    `images` is [{"file": path, "stage": optional_label}]. When a label is present its
    agreement is reported SEPARATELY from model-to-model agreement, and never merged into one
    accuracy score -- the labels available so far are not trustworthy enough to carry that
    weight, and merging would disguise which of the two the number describes.
    """
    backends = backends or list(BACKENDS)
    rows, unavailable = [], {}

    for item in images:
        row = {"file": os.path.basename(item["file"]), "label": item.get("stage")}
        for name in backends:
            result = BACKENDS[name](item["file"])
            if result.get("skipped"):
                unavailable[name] = result["reason"]
                row[name] = None
            else:
                row[name] = result["stage"]
                row[f"{name}_confidence"] = result.get("confidence")
                row[f"{name}_evidence"] = (result.get("evidence") or "")[:120]
        rows.append(row)

    live = [n for n in backends if n not in unavailable]
    summary: Dict[str, Any] = {"rows": rows, "unavailable": unavailable, "backends_live": live}

    if len(live) >= 2:
        a, b = live[0], live[1]
        both = [r for r in rows if r.get(a) and r.get(b)]
        summary["model_agreement"] = {
            "pairs": len(both),
            "agree": sum(1 for r in both if r[a] == r[b]),
            "between": [a, b],
        }
    for name in live:
        labelled = [r for r in rows if r.get("label") and r.get(name)]
        if labelled:
            summary[f"{name}_vs_labels"] = {
                "n": len(labelled),
                "match": sum(1 for r in labelled if r[name] == r["label"]),
            }
    return summary
