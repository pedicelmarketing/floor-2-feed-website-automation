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

    def _default_camera(self, camera_position, camera_target):
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

        return camera_position, camera_target

    def _render_frame(self, camera_position: np.ndarray, camera_target: np.ndarray,
                       width: int, height: int, fov_deg: float,
                       ray_batch: int = 16384):
        """Casts one frame's worth of rays and returns (depth_grid, normal_grid) -- the
        shared core behind both the single-shot render_control_maps() and the multi-frame
        render_camera_path()."""
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
        n_rays = directions.shape[0]

        # self.mesh.ray auto-selects the fastest available backend: Embree (via embreex) if
        # installed, else trimesh's pure-Python ray_triangle. The difference is not marginal
        # -- on the 744-face four-room mesh the pure-Python path takes ~115 s/frame, which is
        # ~94 min for a 49-frame clip and unusable inside a retry loop.
        intersector = self.mesh.ray
        depth = np.full(n_rays, np.inf)
        normals = np.zeros((n_rays, 3))

        # Rays are cast in batches, not all at once. The pure-Python fallback allocates
        # ray x candidate-triangle arrays, so peak memory grows with (rays * triangles): a
        # 480x832 frame is ~400k rays, which survives a 288-face single room but OOM-kills
        # the process on the 744-face four-room mesh -- and a whole floor would be far
        # larger. Batching bounds peak memory to (batch * triangles). Harmless on Embree,
        # which is not memory-bound this way.
        for start in range(0, n_rays, ray_batch):
            end = min(start + ray_batch, n_rays)
            batch_dirs = directions[start:end]
            batch_origins = np.tile(camera_position, (batch_dirs.shape[0], 1))

            locations, index_ray, index_tri = intersector.intersects_location(
                batch_origins, batch_dirs, multiple_hits=False
            )
            if len(index_ray) == 0:
                continue

            hit_vec = locations - batch_origins[index_ray]
            z_depth = hit_vec @ forward  # true z-depth, not ray length
            depth[start + index_ray] = z_depth
            normals[start + index_ray] = self.mesh.face_normals[index_tri]

        return depth.reshape(height, width), normals.reshape(height, width, 3)

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

        camera_position, camera_target = self._default_camera(camera_position, camera_target)
        depth_grid, normal_grid = self._render_frame(camera_position, camera_target, width, height, fov_deg)

        os.makedirs(output_dir, exist_ok=True)
        self._save_depth_map(depth_grid, os.path.join(output_dir, "depth.png"))
        self._save_edge_map(depth_grid, normal_grid, os.path.join(output_dir, "edges.png"))
        print(f"Control maps written to {output_dir}")
        return True

    def render_camera_path(self, camera_path: List[Tuple[np.ndarray, np.ndarray]], output_dir: str,
                            width: int = 480, height: int = 832, fov_deg: float = 70.0,
                            save_raw: bool = True) -> bool:
        """
        Renders one depth + edge frame per (position, target) pair in camera_path, as
        depth_XXXX.png / edges_XXXX.png -- the frame-sequence input a video control signal
        (e.g. Wan VACE) needs, built from the exact same ray-caster already validated on
        single stills.

        The depth PNGs are normalised ONCE for the whole sequence, not per frame. This is
        load-bearing for video and was originally got wrong. Normalising each frame against
        its own nearest and farthest surface makes the grey value mean "how far, relative to
        whatever else is in shot right now", so any change in the nearest or farthest
        surface rescales every pixel in the frame. Measured on the tour: as the camera left
        the doorway the nearest surface went 0.11 m -> 1.61 m in one step, and although the
        camera had moved 6.6 cm the control frame changed by 89.6/255 on average -- a hard
        cut, which the video model then faithfully reproduced. Normalising over the sequence
        makes a surface 3 m away the same grey in every frame, which is what temporal
        coherence requires.

        With save_raw (default), also writes per frame:
          depth_XXXX.npy  raw float metric depth, pre-normalisation (inf where the ray hit
                          nothing), and
          void_XXXX.png   an explicit 0/255 mask of those misses.
        Both exist for QA: in the 8-bit PNG the farthest surface collapses to 0, the same
        value as a miss, so void and far wall are indistinguishable there. Any metric
        comparing generated geometry against ground truth needs the raw arrays instead.
        """
        if self.mesh is None:
            print("No mesh built yet. Call build_mesh() first.")
            return False

        os.makedirs(output_dir, exist_ok=True)

        # Pass 1: cast rays, keep the raw depth, and note the sequence-wide depth range.
        # Raw arrays go to disk rather than memory -- 121 frames at 480x832 is ~390 MB held.
        grids: List[str] = []
        lo_samples, hi_samples = [], []
        for i, (camera_position, camera_target) in enumerate(camera_path):
            depth_grid, normal_grid = self._render_frame(
                np.array(camera_position, dtype=float), np.array(camera_target, dtype=float),
                width, height, fov_deg
            )
            raw_path = os.path.join(output_dir, f"depth_{i:04d}.npy")
            np.save(raw_path, depth_grid)
            grids.append(raw_path)

            self._save_edge_map(depth_grid, normal_grid, os.path.join(output_dir, f"edges_{i:04d}.png"))
            if save_raw:
                void = (~np.isfinite(depth_grid)).astype(np.uint8) * 255
                Image.fromarray(void).save(os.path.join(output_dir, f"void_{i:04d}.png"))

            finite = depth_grid[np.isfinite(depth_grid)]
            if finite.size:
                # Percentiles, not min/max: a handful of grazing-angle pixels at 1 cm would
                # otherwise set the near plane for the entire sequence and flatten everything.
                lo_samples.append(float(np.percentile(finite, 0.5)))
                hi_samples.append(float(np.percentile(finite, 99.5)))

        near = min(lo_samples) if lo_samples else 0.0
        far = max(hi_samples) if hi_samples else 1.0

        # Pass 2: encode every frame against that one shared range.
        for i, raw_path in enumerate(grids):
            depth_grid = np.load(raw_path)
            self._save_depth_map(depth_grid, os.path.join(output_dir, f"depth_{i:04d}.png"),
                                 near=near, far=far)
            if not save_raw:
                os.remove(raw_path)

        print(f"Rendered {len(camera_path)} frames to {output_dir}"
              f"{' (+ raw .npy depth and void masks)' if save_raw else ''}")
        print(f"Depth encoded against a shared range of {near:.2f}..{far:.2f} m "
              f"(one scale for the whole sequence, so a given distance is a given grey)")
        return True

    @staticmethod
    def _save_depth_map(depth_grid: np.ndarray, path: str,
                        near: float = None, far: float = None) -> None:
        """
        Write the 8-bit control frame. Pass near/far to encode against a fixed range; omit
        them only for a one-off still, where per-frame auto-ranging has nothing to be
        inconsistent with.
        """
        finite = depth_grid[np.isfinite(depth_grid)]
        if finite.size == 0 and near is None:
            print("WARNING: camera hit nothing -- check camera_position/camera_target.")
            Image.fromarray(np.zeros(depth_grid.shape, dtype=np.uint8)).save(path)
            return
        if near is None or far is None:
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
