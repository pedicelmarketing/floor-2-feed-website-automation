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


def describe_frame(semantic_png: str) -> Dict[str, Any]:
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
