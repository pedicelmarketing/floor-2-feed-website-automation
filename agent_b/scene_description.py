"""
Write a truthful sentence about what a rendered frame actually contains.

The models keep inventing doors, and the reason is that our depth map cannot forbid one: a door
leaf is about 4 cm thick against a depth range of 0.54-9.26 m quantised to 256 levels, so the
whole door -- leaf, frame, architrave -- fits inside one or two grey levels of the wall behind
it. Painting a door onto a flat wall costs the model nothing geometrically, and a blank corridor
wall is implausible to anything trained on real interiors, so it fills one in.

What the control cannot carry, language can. We know exactly what is in shot because we rendered
it, so the prompt can say so.

Reads the SEMANTIC pass rather than re-deriving from the mesh, because that pass already answers
"what is this pixel" and is produced by the same ray cast as the frame being described -- so the
description cannot drift out of sync with the picture it describes.
"""
import os
from typing import Any, Dict, List

import numpy as np

# Must match MATERIAL_RGB in 3d_room_builder.
MATERIAL_RGB = {
    "wall":      (232, 232, 236),
    "floor":     (176, 118,  62),
    "ceiling":   (140, 152, 172),
    "door":      (198,  74,  60),
    "window":    ( 74, 158, 226),
    "furniture": (108, 176, 104),
}
# A blob smaller than this is a sliver of a reveal seen edge-on, not a doorway worth describing.
MIN_REGION_FRACTION = 0.004
# Below this a material is present but incidental -- a few pixels of ceiling in the top corner.
MIN_MENTION_FRACTION = 0.01


def _regions(mask: np.ndarray) -> List[Dict[str, Any]]:
    """Connected blobs of one material, largest first, with their screen position."""
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    total = mask.size
    out = []
    for i in range(1, count + 1):
        blob = labels == i
        fraction = float(blob.mean())
        if fraction < MIN_REGION_FRACTION:
            continue
        xs = np.flatnonzero(blob.any(axis=0))
        centre_x = float(xs.mean()) / mask.shape[1]
        out.append({"fraction": fraction, "centre_x": centre_x,
                    "side": "on the left" if centre_x < 0.38
                            else "on the right" if centre_x > 0.62 else "straight ahead"})
    return sorted(out, key=lambda r: -r["fraction"])


def _metric_facts(depth_npy: str, ceiling_m: float) -> List[str]:
    """
    The half of the description only a measured twin can write.

    Everything else here could be guessed from looking at the picture. Distances cannot: we know
    the far wall is 4.2 m away because the model is in metres and was checked against 278 doors
    of known width. If a truthful description is going to earn its place, this is the part that
    carries information the model has no other way to get.
    """
    grid = np.load(depth_npy)
    finite = np.isfinite(grid)
    if not finite.any():
        return []
    d = grid[finite]
    near, far = float(np.percentile(d, 1)), float(np.percentile(d, 99))
    facts = [f"The nearest surface is about {near:.1f} m from the camera and the furthest "
             f"about {far:.0f} m. The ceiling is {ceiling_m:.2f} m high."]

    # A narrow scene is the one models most often widen into a room, so say so explicitly.
    mid = grid[grid.shape[0] // 2, :]
    across = mid[np.isfinite(mid)]
    if across.size and float(np.median(across)) < 2.0:
        facts.append("This is a narrow space, not an open room.")
    return facts


def describe_frame(semantic_png: str, depth_npy: str = None,
                   ceiling_m: float = 2.70) -> Dict[str, Any]:
    from PIL import Image

    rgb = np.asarray(Image.open(semantic_png).convert("RGB")).astype(int)
    masks = {name: (np.abs(rgb - np.array(colour)).sum(axis=2) < 12)
             for name, colour in MATERIAL_RGB.items()}
    fractions = {name: float(m.mean()) for name, m in masks.items()}

    doors = _regions(masks["door"])
    windows = _regions(masks["window"])
    furniture = _regions(masks["furniture"])

    def count_phrase(items, singular, plural):
        if not items:
            return None
        if len(items) == 1:
            return f"one {singular} {items[0]['side']}"
        sides = ", ".join(i["side"] for i in items[:3])
        return f"{len(items)} {plural} ({sides})"

    present, absent = [], []
    for items, singular, plural in ((doors, "doorway", "doorways"),
                                    (windows, "window", "windows")):
        phrase = count_phrase(items, singular, plural)
        (present.append(phrase) if phrase else absent.append(plural))

    if furniture:
        present.append(f"{len(furniture)} piece{'s' if len(furniture) > 1 else ''} of furniture")
    else:
        absent.append("furniture")

    # Name whatever fills the frame. On the anchor used to develop this, HALF the picture was a
    # single tall furniture volume and only 15% was wall -- and every model painted a run of
    # cupboard doors onto it. That is not an unreasonable reading of a featureless 2.1 m box in a
    # corridor, but nothing told them which it was, so all three guessed and two guessed wrong.
    dominant = max(fractions, key=fractions.get)
    lead = None
    if fractions[dominant] > 0.30:
        noun = {"wall": "a plain plaster wall", "furniture": "a tall built-in cupboard",
                "floor": "the floor", "ceiling": "the ceiling",
                "door": "a doorway", "window": "a window"}[dominant]
        lead = f"{noun.capitalize()} fills most of the frame."

    # The load-bearing half. Stating what is NOT there is the only way to describe a blank wall,
    # and a blank wall is exactly what the models keep filling in.
    sentences = [lead] if lead else []
    if present:
        sentences.append("In view: " + "; ".join(present) + ".")
    if absent:
        sentences.append("There are no " + ", no ".join(absent) + " anywhere in this shot.")
    if fractions["wall"] + fractions["furniture"] > 0.30:
        sentences.append("The large flat surfaces are unbroken, with nothing mounted on them "
                         "and no openings other than those listed.")

    if depth_npy and os.path.exists(depth_npy):
        sentences.extend(_metric_facts(depth_npy, ceiling_m))

    # What each surface is made of, from the same pass that says where it is. Naming the
    # materials is not styling -- it is the identity a depth map cannot carry, and the reason a
    # door-shaped gap kept coming back as a run of oak panelling.
    seen = [n for n in ("floor", "wall", "ceiling", "door", "window", "furniture")
            if fractions[n] > MIN_MENTION_FRACTION]
    if seen:
        finish = {"floor": "pale oak plank floor", "wall": "matte white plaster walls",
                  "ceiling": "a flat white ceiling", "door": "painted door reveals",
                  "window": "a window opening with daylight beyond",
                  "furniture": "plain built-in joinery"}
        sentences.append("Surfaces: " + ", ".join(finish[n] for n in seen) + ".")

    return {
        "fractions": fractions,
        "doors": len(doors), "windows": len(windows), "furniture": len(furniture),
        "description": " ".join(sentences),
    }


def prompt_for_frame(semantic_png: str, style: str) -> str:
    """Style sentence first (models weight early tokens most), then the facts."""
    return f"{style} {describe_frame(semantic_png)['description']}"


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        info = describe_frame(path)
        print(os.path.basename(path))
        print(f"  doors {info['doors']}  windows {info['windows']}  "
              f"furniture {info['furniture']}")
        print(f"  {info['description']}")
