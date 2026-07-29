import os
import math
import numpy as np
import trimesh
import shapely.geometry
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional


def _edge_outward_normal(a: np.ndarray, b: np.ndarray, polygon_is_ccw: bool) -> np.ndarray:
    direction = b - a
    direction = direction / np.linalg.norm(direction)
    normal = np.array([direction[1], -direction[0]])
    return normal if polygon_is_ccw else -normal


def _is_ccw(polygon: List[List[float]]) -> bool:
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area > 0


def _wall_world_transform(wall_a: np.ndarray, direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Maps a wall panel authored in local (u=along wall, v=height, w=thickness) coords into
    world space: local x -> world XY along the wall, local y -> world Z (up), local z ->
    world XY along the wall's outward normal (thickness direction).
    """
    t = np.zeros((4, 4))
    t[0:2, 0] = direction   # local x (u) -> world XY along the wall
    t[2, 1] = 1.0           # local y (v) -> world Z
    t[0:2, 2] = normal      # local z (w) -> world XY thickness direction
    t[0:2, 3] = wall_a      # translation
    t[3, 3] = 1.0
    return t


class Room3DBuilder:
    def __init__(self, room_data: Dict[str, Any]):
        self.room_data = room_data
        self.mesh: Optional[trimesh.Trimesh] = None

    def build_mesh(self, wall_thickness_m: float = 0.12, floor_thickness_m: float = 0.1,
                    ceiling_thickness_m: float = 0.1) -> bool:
        """
        Extrudes the room's 2D footprint into an untextured gray-box "blockout": floor slab,
        ceiling slab, and one wall panel per polygon edge with rectangular holes cut for any
        door/window opening recorded on that edge.

        This mesh exists to be measured (for depth/edge rendering), never to be looked at
        directly -- no materials or lighting are applied here by design.
        """
        room_name = self.room_data.get("room_name", "Unknown Room")
        print(f"Building 3D blockout for {room_name}...")

        polygon_2d = self.room_data.get("polygon")
        ceiling_height = self.room_data.get("ceiling_height_m")

        if not polygon_2d or len(polygon_2d) < 3:
            print("No polygon data provided.")
            return False

        if ceiling_height is None:
            # Refuse to guess. A wrong ceiling height is the single most visible error in a
            # render (see docs/specs architectural-visualization-pipeline.md, PDF rule 1) --
            # rooms missing this must be routed to human review, not defaulted.
            print(f"Room '{room_name}' has no extracted ceiling height. Refusing to blockout "
                  f"with a guessed value -- route to human review instead.")
            return False

        polygon_is_ccw = _is_ccw(polygon_2d)
        openings_by_edge: Dict[int, List[Dict[str, Any]]] = {}
        for opening in self.room_data.get("doors", []) + self.room_data.get("windows", []):
            openings_by_edge.setdefault(opening["wall_edge_index"], []).append(opening)

        parts = []
        n = len(polygon_2d)
        for i in range(n):
            a = np.array(polygon_2d[i], dtype=float)
            b = np.array(polygon_2d[(i + 1) % n], dtype=float)
            wall_len = float(np.linalg.norm(b - a))
            if wall_len < 1e-6:
                continue
            direction = (b - a) / wall_len
            normal = _edge_outward_normal(a, b, polygon_is_ccw)

            wall_poly = shapely.geometry.box(0, 0, wall_len, ceiling_height)
            for opening in openings_by_edge.get(i, []):
                u0 = opening["offset_along_wall_m"]
                u1 = u0 + opening["width_m"]
                v0 = opening["sill_m"]
                v1 = opening["head_m"]
                # difference() (rather than an interior ring) correctly handles both a
                # fully-enclosed window hole and a door notch that touches the floor edge.
                wall_poly = wall_poly.difference(shapely.geometry.box(u0, v0, u1, v1))

            if not wall_poly.is_valid or wall_poly.area <= 0:
                print(f"Skipping degenerate wall panel on edge {i} of '{room_name}'.")
                continue

            wall_mesh = trimesh.creation.extrude_polygon(wall_poly, height=wall_thickness_m)
            wall_mesh.apply_transform(_wall_world_transform(a, direction, normal))
            parts.append(wall_mesh)

        floor_poly = shapely.geometry.Polygon(polygon_2d)
        floor_mesh = trimesh.creation.extrude_polygon(floor_poly, height=floor_thickness_m)
        floor_mesh.apply_translation([0, 0, -floor_thickness_m])
        parts.append(floor_mesh)

        ceiling_mesh = trimesh.creation.extrude_polygon(floor_poly, height=ceiling_thickness_m)
        ceiling_mesh.apply_translation([0, 0, ceiling_height])
        parts.append(ceiling_mesh)

        self.mesh = trimesh.util.concatenate(parts)
        print(f"Blockout built: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces.")
        return True

    def render_control_maps(self, output_dir: str, camera_position=None, camera_target=None,
                             width: int = 480, height: int = 360, fov_deg: float = 70.0) -> bool:
        """
        Places a camera in the blockout and renders a true (ray-cast, not estimated) Z-depth
        map and an edge map, using trimesh's mesh-ray intersector -- no OpenGL/GPU context
        needed, which sidesteps the headless-rendering risk this file used to flag.
        """
        if self.mesh is None:
            print("No mesh built yet. Call build_mesh() first.")
            return False

        polygon_2d = self.room_data.get("polygon")
        ceiling_height = self.room_data["ceiling_height_m"]
        centroid = np.mean(np.array(polygon_2d), axis=0)

        if camera_position is None:
            # Default: pulled in from a corner, eye height 1.6m (spec's standard eye-level
            # camera). Inset scales with room size so small rooms don't put the camera nose
            # against the wall and large rooms don't leave it stranded near the corner.
            corner = np.array(polygon_2d[0], dtype=float)
            inward = (centroid - corner)
            corner_to_centroid_dist = np.linalg.norm(inward)
            inward = inward / corner_to_centroid_dist
            inset = min(1.5, corner_to_centroid_dist * 0.4)
            camera_position = np.array([corner[0] + inward[0] * inset, corner[1] + inward[1] * inset, 1.6])
        else:
            camera_position = np.array(camera_position, dtype=float)

        if camera_target is None:
            camera_target = np.array([centroid[0], centroid[1], min(1.6, ceiling_height / 2)])
        else:
            camera_target = np.array(camera_target, dtype=float)

        forward = camera_target - camera_position
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0, 0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.999:
            # Looking (nearly) straight up/down: world_up is parallel to forward, so it can't
            # define "right". Fall back to world X as the reference axis instead.
            world_up = np.array([1.0, 0, 0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        aspect = width / height
        fov_rad = math.radians(fov_deg)
        half_h = math.tan(fov_rad / 2)
        half_w = half_h * aspect

        xs = np.linspace(-half_w, half_w, width)
        ys = np.linspace(half_h, -half_h, height)
        grid_x, grid_y = np.meshgrid(xs, ys)

        directions = (
            forward[None, None, :]
            + grid_x[:, :, None] * right[None, None, :]
            + grid_y[:, :, None] * cam_up[None, None, :]
        )
        directions = directions.reshape(-1, 3)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        origins = np.tile(camera_position, (directions.shape[0], 1))

        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        locations, index_ray, index_tri = intersector.intersects_location(
            origins, directions, multiple_hits=False
        )

        depth = np.full(directions.shape[0], np.inf)
        normals = np.zeros((directions.shape[0], 3))
        if len(index_ray) > 0:
            hit_vec = locations - origins[index_ray]
            z_depth = hit_vec @ forward  # true z-depth, not ray length
            depth[index_ray] = z_depth
            normals[index_ray] = self.mesh.face_normals[index_tri]

        depth_grid = depth.reshape(height, width)
        normal_grid = normals.reshape(height, width, 3)

        os.makedirs(output_dir, exist_ok=True)
        self._save_depth_map(depth_grid, os.path.join(output_dir, "depth.png"))
        self._save_edge_map(depth_grid, normal_grid, os.path.join(output_dir, "edges.png"))
        print(f"Control maps written to {output_dir}")
        return True

    @staticmethod
    def _save_depth_map(depth_grid: np.ndarray, path: str) -> None:
        finite = depth_grid[np.isfinite(depth_grid)]
        if finite.size == 0:
            print("WARNING: camera hit nothing -- check camera_position/camera_target.")
            Image.fromarray(np.zeros(depth_grid.shape, dtype=np.uint8)).save(path)
            return
        near, far = float(finite.min()), float(finite.max())
        span = max(far - near, 1e-6)
        # SD-style depth ControlNet convention: near = bright, far = dark, misses = black
        normalized = np.where(np.isfinite(depth_grid), 255.0 * (1.0 - (depth_grid - near) / span), 0.0)
        Image.fromarray(np.clip(normalized, 0, 255).astype(np.uint8)).save(path)

    @staticmethod
    def _save_edge_map(depth_grid: np.ndarray, normal_grid: np.ndarray, path: str) -> None:
        finite_mask = np.isfinite(depth_grid)
        safe_depth = np.where(finite_mask, depth_grid, 0.0)

        # Depth discontinuities -> silhouette + occlusion edges
        gx = np.abs(np.diff(safe_depth, axis=1, prepend=safe_depth[:, :1]))
        gy = np.abs(np.diff(safe_depth, axis=0, prepend=safe_depth[:1, :]))
        depth_edges = (gx + gy) > 0.15

        # Normal discontinuities -> wall/ceiling/floor corner lines (missed by depth alone)
        ndx = np.abs(np.diff(normal_grid, axis=1, prepend=normal_grid[:, :1, :])).sum(axis=2)
        ndy = np.abs(np.diff(normal_grid, axis=0, prepend=normal_grid[:1, :, :])).sum(axis=2)
        normal_edges = (ndx + ndy) > 0.3

        edges = (depth_edges | normal_edges) & finite_mask
        image = np.where(edges, 255, 0).astype(np.uint8)
        Image.fromarray(image).save(path)


if __name__ == "__main__":
    import json
    import os as _os
    sample_room = {
        "room_name": "Living Room",
        "polygon": [[0, 0], [6, 0], [6, 4], [0, 4]],
        "ceiling_height_m": 3.0,
        "doors": [{"wall_edge_index": 0, "offset_along_wall_m": 2.55, "width_m": 0.9,
                    "sill_m": 0.0, "head_m": 2.1}],
        "windows": [{"wall_edge_index": 2, "offset_along_wall_m": 1.5, "width_m": 3.0,
                      "sill_m": 0.9, "head_m": 2.1}],
    }
    builder = Room3DBuilder(sample_room)
    if builder.build_mesh():
        out_dir = _os.path.join(_os.path.dirname(__file__), "fixtures", "output")
        builder.render_control_maps(out_dir)
        builder.mesh.export(_os.path.join(out_dir, "blockout.glb"))
