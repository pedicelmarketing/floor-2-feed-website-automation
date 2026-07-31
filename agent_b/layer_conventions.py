"""
Work out which layers in a drawing hold rooms, doors, windows and walls.

Every practice names its layers differently, so hardcoding one set makes a parser that reads
exactly one office's drawings. The real client file here proves the point: the parser was
written for the American AIA convention (A-ROOM / A-DOOR / A-GLAZ) and the file is Spanish
(0-AREAS / A-PUERTAS / A-VIDRIO). Nothing matched, which is why room boundaries ended up
hand-picked by DXF entity handle -- "the polyline whose internal id is 15BE1F" -- a selection
that would not survive the same file being re-saved, let alone a different drawing.

So: match layers to ROLES by pattern, score each candidate convention against the layers
actually present, and pick the best. Add a language or an office style by adding patterns
here rather than by editing the extractor.

What this deliberately does NOT do is guess. A role with no matching layer is reported
missing, and the extractor is expected to return nothing for it rather than invent geometry.
Silence about a missing room layer is how you end up with four rooms that came from
somewhere nobody can explain.
"""
import re
import unicodedata
from typing import Any, Dict, List, Optional

# The roles the geometry pipeline needs. Names are internal; layer names vary wildly.
ROLES = ("room_boundary", "room_label", "door", "window", "wall")

# Patterns are matched against a normalised layer name (accents stripped, upper-cased), so
# "A-PROYECCIÓN" and "A-PROYECCION" behave identically -- LibreDWG and ODA disagree about
# accent encoding in layer names, which would otherwise make detection converter-dependent.
CONVENTIONS: List[Dict[str, Any]] = [
    {
        "name": "AIA (US/UK English)",
        "patterns": {
            "room_boundary": [r"^A-ROOM$", r"^A-AREA$", r"^A-AREA-\w+$"],
            "room_label": [r"^A-ROOM-ID$", r"^A-AREA-IDEN$", r"^A-ANNO-TEXT$"],
            "door": [r"^A-DOOR$", r"^A-DOOR-\w+$"],
            "window": [r"^A-GLAZ$", r"^A-GLAZ-\w+$", r"^A-WIND(OW)?$"],
            "wall": [r"^A-WALL$", r"^A-WALL-\w+$"],
        },
    },
    {
        "name": "Spanish (ES/LatAm)",
        "patterns": {
            "room_boundary": [r"^\d*-?AREAS?$", r"^A-AREAS?$", r"^AMBIENTES?$", r"^RECINTOS?$"],
            "room_label": [r"^A-ANOT-TEXTO$", r"^A-ANOT-\w+$", r"^TEXTOS?$", r"^ROTULOS?$"],
            "door": [r"^A-PUERTAS?$", r"^PUERTAS?$"],
            "window": [r"^A-VIDRIOS?$", r"^VIDRIOS?$", r"^A-VENTANAS?$", r"^VENTANAS?$"],
            "wall": [r"^A-MUROS?$", r"^MUROS?$", r"^A-PARED(ES)?$"],
        },
    },
    {
        "name": "generic keyword",
        "patterns": {
            "room_boundary": [r"ROOM", r"AREA", r"SPACE"],
            "room_label": [r"LABEL", r"TEXT", r"ANNO", r"IDEN"],
            "door": [r"DOOR", r"PUERTA", r"PORTE", r"TUR"],
            "window": [r"GLAZ", r"WINDOW", r"VIDRIO", r"VENTANA", r"FENETRE"],
            "wall": [r"WALL", r"MURO", r"PARED", r"MUR"],
        },
    },
]


def normalise(layer_name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", layer_name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).upper().strip()


def _match_role(patterns: List[str], layers: Dict[str, int]) -> List[str]:
    """Layers matching any pattern, busiest first. Empty layers are ignored."""
    hits = [name for name, count in layers.items()
            if count > 0 and any(re.search(p, normalise(name)) for p in patterns)]
    return sorted(hits, key=lambda n: -layers[n])


def layer_counts(msp) -> Dict[str, int]:
    """Entity count per layer. Counts, not names, because empty layers are noise."""
    counts: Dict[str, int] = {}
    for entity in msp:
        if entity.dxf.hasattr("layer"):
            counts[entity.dxf.layer] = counts.get(entity.dxf.layer, 0) + 1
    return counts


def detect(msp=None, conventions: List[Dict[str, Any]] = None,
           counts: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Pick the convention that best explains this drawing's layers.

    Takes either a DXF modelspace or, via `counts`, a plain {layer_name: entity_count} map --
    the same detection serves a PDF, whose optional content groups carry the identical layer
    names because both come out of the same CAD file.

    Scoring weights the roles by how much the pipeline depends on them: without a room
    boundary there is nothing to build, whereas a missing wall layer costs little because the
    blockout extrudes room outlines rather than tracing wall centrelines. A convention that
    matches three decorative roles and no rooms should not beat one that finds the rooms.
    """
    conventions = conventions or CONVENTIONS
    if counts is None:
        if msp is None:
            raise ValueError("pass either a modelspace or a counts mapping")
        counts = layer_counts(msp)
    weights = {"room_boundary": 3.0, "door": 2.0, "window": 2.0, "room_label": 1.0, "wall": 0.5}

    scored = []
    for convention in conventions:
        matched = {role: _match_role(convention["patterns"].get(role, []), counts)
                   for role in ROLES}
        score = sum(weights[role] for role, hits in matched.items() if hits)
        scored.append({"name": convention["name"], "score": score, "layers": matched})

    scored.sort(key=lambda c: -c["score"])
    best = scored[0]
    missing = [role for role in ROLES if not best["layers"][role]]

    return {
        "convention": best["name"],
        "score": best["score"],
        "layers": {role: hits[0] if hits else None for role, hits in best["layers"].items()},
        "all_candidates": {role: hits for role, hits in best["layers"].items()},
        "missing_roles": missing,
        # A drawing whose rooms cannot be located is not one this pipeline can process; say
        # so here rather than returning an empty room list that reads like an empty building.
        "usable": "room_boundary" not in missing,
        "ranking": [(c["name"], c["score"]) for c in scored],
        "layer_counts": counts,
    }


def describe(detection: Dict[str, Any]) -> str:
    lines = [f"Layer convention: {detection['convention']} (score {detection['score']:.1f})"]
    for role in ROLES:
        chosen = detection["layers"][role]
        alternatives = detection["all_candidates"][role]
        extra = f"  (+{len(alternatives) - 1} more)" if len(alternatives) > 1 else ""
        lines.append(f"  {role:15s} -> {chosen or 'NOT FOUND'}{extra}")
    if detection["missing_roles"]:
        lines.append(f"  missing: {', '.join(detection['missing_roles'])}")
    return "\n".join(lines)
