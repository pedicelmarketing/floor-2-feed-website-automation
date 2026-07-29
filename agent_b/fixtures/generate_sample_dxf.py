"""
Generates a synthetic single-room DXF fixture for testing the geometry-extraction
pipeline (dwg_parser.py -> 3d_room_builder.py) without needing a real client CAD file.

Layer / annotation convention this fixture follows (documented once here, consumed by
dwg_parser.py):

  A-ROOM        closed LWPOLYLINE  -> room footprint boundary (meters, world XY)
  A-ROOM-ID     TEXT inside the room polygon -> room name
  A-ROOM-DATA   TEXT inside the room polygon -> "CH=<ceiling_height_m>"
  A-DOOR        LINE on a room-boundary edge -> door opening (endpoints = opening extents)
  A-DOOR-DATA   TEXT near the door's midpoint -> "HEAD=<m>" (sill assumed 0.0 = floor)
  A-GLAZ        LINE on a room-boundary edge -> window opening (endpoints = opening extents)
  A-GLAZ-DATA   TEXT near the window's midpoint -> "SILL=<m>;HEAD=<m>"

"Near" = within 1.0m of the opening segment's midpoint.
"""
import ezdxf


def build(path: str) -> None:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    for layer in ["A-ROOM", "A-ROOM-ID", "A-ROOM-DATA", "A-DOOR", "A-DOOR-DATA", "A-GLAZ", "A-GLAZ-DATA"]:
        doc.layers.add(layer)

    # 6m x 4m room, 3m ceiling
    footprint = [(0, 0), (6, 0), (6, 4), (0, 4)]
    msp.add_lwpolyline(footprint, close=True, dxfattribs={"layer": "A-ROOM"})

    msp.add_text("Living Room", dxfattribs={"layer": "A-ROOM-ID", "height": 0.2}).set_placement((2.5, 2.0))
    msp.add_text("CH=3.0", dxfattribs={"layer": "A-ROOM-DATA", "height": 0.15}).set_placement((2.5, 1.6))

    # Door centered on south wall (0,0)-(6,0), width 0.9m
    msp.add_line((2.55, 0), (3.45, 0), dxfattribs={"layer": "A-DOOR"})
    msp.add_text("HEAD=2.10", dxfattribs={"layer": "A-DOOR-DATA", "height": 0.12}).set_placement((2.55, -0.3))

    # Window centered on north wall (6,4)-(0,4), width 3.0m
    msp.add_line((4.5, 4), (1.5, 4), dxfattribs={"layer": "A-GLAZ"})
    msp.add_text("SILL=0.90;HEAD=2.10", dxfattribs={"layer": "A-GLAZ-DATA", "height": 0.12}).set_placement((3.0, 4.2))

    doc.saveas(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    import os
    build(os.path.join(os.path.dirname(__file__), "sample_room.dxf"))
