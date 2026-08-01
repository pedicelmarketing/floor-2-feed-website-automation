import os
import math
import numpy as np
import trimesh
import shapely.geometry
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional


# Brightness multiplier per material. Kept well apart so the classes are distinguishable after
# the model's own contrast handling, and all below 1.0 so nothing clips to white and loses its
# shading. Ordinary wall is the 1.0 reference.
MATERIAL_TINT = {
    "wall": 1.00,
    "floor": 0.62,
    "ceiling": 0.80,
    "door": 0.42,        # darkest: the opening the model kept turning into panelling
    "window": 1.00,
    "furniture": 0.52,
}

# Wider spread, for a render meant to be READ -- by a person deciding whether the extraction is
# right, or by a model being handed the scene as reference rather than as a per-frame control.
# MATERIAL_TINT above is deliberately left alone: it is the control every measured run so far was
# generated against, and changing it would silently make those numbers incomparable.
#
# Windows are the brightest thing in the frame, which is both legible and what a window
# physically is. Six values one step apart.
#
# The steps are 0.10, not the 0.14 first tried. Tint MULTIPLIES the lit value rather than
# replacing it, so spreading the materials downward to separate them darkens every surface at
# once: the first attempt put walls at 0.80 and furniture at 0.38 and rendered a black clip.
# Separation has to come from a narrow spread near the top plus CLAY_EXPOSURE below, not from
# pushing the dim materials further down.
MATERIAL_TINT_LEGIBLE = {
    "window": 1.00,      # brightest: daylight
    "wall": 0.90,
    "ceiling": 0.80,
    "floor": 0.70,
    "furniture": 0.60,
    "door": 0.50,        # darkest
}

# Flat colour key, for the pass whose only job is to say WHAT each surface is. Hues are chosen to
# stay distinguishable in greyscale too (their luminances differ), so the pass survives being
# converted, compressed or viewed by something colour-blind.
MATERIAL_RGB = {
    "wall":      (232, 232, 236),
    "floor":     (176, 118,  62),
    "ceiling":   (140, 152, 172),
    "door":      (198,  74,  60),
    "window":    ( 74, 158, 226),
    "furniture": (108, 176, 104),
}

# Plausible surface colour, for the shaded colour pass. These are ALBEDO -- how much light each
# surface reflects -- not final pixel values, because they get multiplied by the lighting. White
# plaster really does sit near 0.75 of white rather than at white itself, so painting it 255 and
# then lighting it produces a wall that clips to flat white and loses every corner.
#
# Chosen to be believable AND separable: an oak floor and a plaster wall differ by hue, so no
# amount of shading can make one read as the other. That is the failure the grey passes have --
# a well-lit door and a shadowed wall land on the same grey.
MATERIAL_BASE_RGB = {
    "wall":      (196, 192, 186),
    "ceiling":   (208, 208, 212),
    "floor":     (150, 112,  72),
    "door":      (172, 148, 120),
    "window":    (222, 224, 226),   # reveals, backlit by whatever is outside
    "furniture": (132, 126, 118),
}

# What a ray that hits nothing becomes in the colour pass. In the grey passes a miss is pure
# white -- identical to a brightly lit wall -- and in this drawing the windows ARE misses,
# because the opening is cut clean through between sill and head. So a window and a hole in the
# model were literally the same pixel value. Daylight blue says "this is outside".
SKY_RGB = (206, 226, 244)

# The colour pass needs less gain than the grey one. Clay multiplies a tint of at most 1.0 into
# the lit value, so it starts dark and needs lifting; colour starts from an albedo near 200 and
# would clip to flat white at the same exposure, losing exactly the corner shading that makes
# the render readable.
MATERIAL_EXPOSURE_SCALE = 1.15

# How much light the floor throws back at the ceiling in the colour pass. Applies to
# downward-facing surfaces only; see _shade_from_normals.
CEILING_BOUNCE = 0.34

# Sunlight and skylight, for the cinematic pass. Sun is warm and strong; the fill is cool and
# weak. Keeping them far apart in both colour and intensity is what produces contrast -- the
# earlier passes were deliberately flat so a person could READ them, and a model steered by a
# flat evenly-lit render returns a flat evenly-lit photograph. Legibility and beauty pull in
# opposite directions here, and the reference clip should be optimised for the second.
SUN_RGB = (255, 241, 216)        # ~5000 K direct sun
# Near-neutral, only just cool. The first attempt used a proper sky blue (176,199,232), which is
# what physically comes off the sky -- and rendered every corridor blue, because almost nothing
# in a flat is in direct sun so the fill IS the picture. Indoors most bounce light has already
# hit warm plaster and a wood floor by the time it arrives, so it lands close to neutral. The
# blue belongs only in shadows right beside a window, which this is too crude to model.
SKY_FILL_RGB = (226, 229, 234)
#
# The first values tried were 1.55 / 0.42, chosen to be physically sensible. They rendered the
# corridors at a mean pixel value of 38 out of 255, because most of a flat is not in direct sun
# and 0.42 of ambient is not enough to see by. Physically defensible, visually useless -- and a
# model copying a near-black reference returns a near-black photograph. Real interior
# photography lifts the fill hard and keeps the contrast in the ratio, which is what these do:
# the sun is 2.4x the fill, so window patches still read as window patches.
SUN_STRENGTH = 2.40
SKY_STRENGTH = 1.00


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
        # Which triangle each ray hit. Carried out so a caller can say what a surface IS, not
        # only where it is -- see _save_clay_render's material tinting.
        faces = np.full(n_rays, -1, dtype=np.int64)

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
            faces[start + index_ray] = index_tri

        self._last_face_index = faces.reshape(height, width)
        # Where each pixel's surface actually is in the world. Reconstructed rather than
        # collected in the loop because the batches only write hits, and this way a miss stays
        # at the camera rather than at some stale location. Needed for cast shadows: you cannot
        # ask "does the sun reach this point" without knowing which point.
        ray_len = depth / np.maximum(directions @ forward, 1e-6)
        points = camera_position[None, :] + directions * np.where(
            np.isfinite(ray_len), ray_len, 0.0)[:, None]
        self._last_hit_points = points.reshape(height, width, 3)
        return depth.reshape(height, width), normals.reshape(height, width, 3)

    def sun_visibility(self, points: np.ndarray, normals: np.ndarray, finite: np.ndarray,
                       sun_dir: np.ndarray) -> np.ndarray:
        """
        1.0 where the sun reaches a surface, 0.0 where something blocks it.

        This is the whole difference between a render that reads as a lit room and one that
        reads as a diagram. Ambient shading tells you a surface faces the light; only a shadow
        ray tells you a WINDOW is casting a bright patch onto that particular piece of floor,
        and the shape of that patch is the single most recognisable thing in architectural
        photography. It costs one extra ray per pixel against a mesh Embree is already holding.

        Origins are pushed off the surface along its own normal, or every ray instantly
        re-hits the triangle it started from and the whole frame renders as shadow.
        """
        height, width = finite.shape
        lit = np.zeros(height * width, dtype=bool)
        flat_pts = points.reshape(-1, 3)
        flat_nrm = normals.reshape(-1, 3)
        mask = finite.reshape(-1)
        if not mask.any():
            return lit.reshape(height, width).astype(float)

        origins = flat_pts[mask] + flat_nrm[mask] * 1e-3
        towards = np.tile(-sun_dir, (origins.shape[0], 1))
        blocked = np.zeros(origins.shape[0], dtype=bool)
        for start in range(0, origins.shape[0], 65536):
            end = min(start + 65536, origins.shape[0])
            blocked[start:end] = self.mesh.ray.intersects_any(
                origins[start:end], towards[start:end])
        lit[mask] = ~blocked
        return lit.reshape(height, width).astype(float)

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
        self._save_clay_render(depth_grid, normal_grid, os.path.join(output_dir, "clay.png"),
                               view_dir=camera_target - camera_position)
        print(f"Control maps written to {output_dir}")
        return True

    def render_camera_path(self, camera_path: List[Tuple[np.ndarray, np.ndarray]], output_dir: str,
                            width: int = 480, height: int = 832, fov_deg: float = 70.0,
                            save_raw: bool = True,
                            face_materials: np.ndarray = None,
                            tint_map: dict = None, semantic: bool = False,
                            exposure: float = 1.0, sun_dir: np.ndarray = None) -> bool:
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
            materials = None
            if face_materials is not None:
                idx = self._last_face_index
                # -1 marks a ray that hit nothing; index it to a harmless slot and let the
                # finite mask discard it, rather than letting -1 wrap to the last face.
                materials = np.where(idx >= 0, face_materials[np.clip(idx, 0, None)], "")
            self._save_clay_render(depth_grid, normal_grid,
                                   os.path.join(output_dir, f"clay_{i:04d}.png"),
                                   view_dir=np.array(camera_target, dtype=float)
                                   - np.array(camera_position, dtype=float),
                                   materials=materials, tint_map=tint_map, exposure=exposure)
            if materials is not None and semantic:
                self._save_semantic_map(depth_grid, materials,
                                        os.path.join(output_dir, f"semantic_{i:04d}.png"))
                sun_visible = None
                if sun_dir is not None:
                    sun_visible = self.sun_visibility(
                        self._last_hit_points, normal_grid, np.isfinite(depth_grid), sun_dir)
                self._save_material_render(
                    depth_grid, normal_grid,
                    os.path.join(output_dir, f"material_{i:04d}.png"),
                    view_dir=np.array(camera_target, dtype=float)
                    - np.array(camera_position, dtype=float),
                    materials=materials, exposure=exposure * MATERIAL_EXPOSURE_SCALE,
                    sun_dir=sun_dir, sun_visible=sun_visible)
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
    def _shade_from_normals(normal_grid, depth_grid, finite, view_dir=None, up_fill=0.0):
        """
        Lambert key + fill + distance falloff + screen-space ambient occlusion.

        Extracted so the grey clay pass and the colour material pass light the scene
        identically. Kept as one function rather than copied because the two would drift:
        the camera-relative key light and the occlusion radius were both got wrong once
        already, and a second copy is a second place to get them wrong again.
        """

        # The key light must be tied to the CAMERA, not to world space. A fixed world light
        # pointing mostly upward lights floors and leaves every vertical wall on ambient alone,
        # which renders a room as one flat dark plane -- measured, and it is what the first
        # attempt produced. An over-the-shoulder key means whatever the camera faces is lit.
        if view_dir is None:
            key = np.array([0.35, 0.35, 0.87], dtype=float)
        else:
            view = np.asarray(view_dir, dtype=float)
            view = view / (np.linalg.norm(view) + 1e-9)
            up = np.array([0.0, 0.0, 1.0])
            side = np.cross(view, up)
            side = side / (np.linalg.norm(side) + 1e-9)
            key = -view + 0.45 * up + 0.30 * side     # behind the camera, high and to one side
        key /= np.linalg.norm(key)
        # Fill from the opposite side so shadowed faces keep some shape instead of going black.
        fill = -key + np.array([0.0, 0.0, 0.55])
        fill /= np.linalg.norm(fill)

        lam = np.clip((normal_grid * key).sum(axis=2), 0.0, 1.0)
        bounce = np.clip((normal_grid * fill).sum(axis=2), 0.0, 1.0)
        shade = 0.18 + 0.66 * lam + 0.16 * bounce            # ambient + key + fill

        # Light bounced up off the floor, for DOWNWARD-facing surfaces only. Both the key and the
        # fill point upward to some degree, so a ceiling -- whose normal points straight down --
        # catches neither and renders on ambient alone, at 0.18 of white. That is nearly black,
        # and it is wrong: a real ceiling is one of the brightest surfaces in a room precisely
        # because the floor throws light back at it. Defaults to zero so the control track every
        # measured run used is bit-for-bit unchanged.
        if up_fill:
            shade = shade + up_fill * np.clip(-normal_grid[:, :, 2], 0.0, 1.0)

        # Fade very distant surfaces slightly, which reads as depth without faking fog.
        if finite.any():
            far = float(np.percentile(depth_grid[finite], 99.0))
            if far > 1e-6:
                falloff = 1.0 - 0.25 * np.clip(np.where(finite, depth_grid, 0.0) / far, 0, 1)
                shade = shade * falloff

        # AMBIENT OCCLUSION. Flat Lambert leaves a corner looking like flat wall: the two faces
        # meeting there differ only by their shading angle, and where that angle barely changes
        # the corner vanishes entirely. Every real CG render darkens contact edges, and models
        # trained on CG footage read that darkening as geometry. Without it the render is
        # missing the cue that says "these two surfaces meet here".
        #
        # Screen-space, from the depth and normal grids already in hand -- no extra ray casting.
        # A pixel is occluded to the extent its neighbours sit in FRONT of the plane through it,
        # which is what happens on the inside of a corner and not on a flat wall or a convex edge.
        safe = np.where(finite, depth_grid, 0.0)
        occlusion = np.zeros_like(safe)
        samples = 0
        for dy, dx in ((0, 3), (0, -3), (3, 0), (-3, 0), (2, 2), (2, -2), (-2, 2), (-2, -2)):
            shifted = np.roll(np.roll(safe, dy, axis=0), dx, axis=1)
            valid = np.roll(np.roll(finite, dy, axis=0), dx, axis=1) & finite
            # Positive where the neighbour is nearer; clamped so a distant background does not
            # read as a bright halo, and so a doorway edge is not mistaken for a corner.
            delta = np.clip(safe - shifted, 0.0, 0.35)
            occlusion += np.where(valid, delta, 0.0)
            samples += 1
        occlusion = np.clip(occlusion / (samples * 0.35), 0.0, 1.0)
        shade = shade * (1.0 - 0.55 * occlusion)
        return shade

    @staticmethod
    def _save_clay_render(depth_grid: np.ndarray, normal_grid: np.ndarray, path: str,
                          view_dir: np.ndarray = None, materials: np.ndarray = None,
                          tint_map: dict = None, exposure: float = 1.0) -> None:
        """
        A plain shaded grey render of the blockout -- what a CG viewport shows before materials.

        Needed because "render to real" models expect a RENDER, not a depth map. A depth map
        encodes distance: bright means near, and a wall two metres away is the same grey
        whichever way it faces. A render encodes light: surfaces facing the light are bright,
        surfaces facing away are dark, and that is what tells a viewer -- or a model trained on
        CG footage -- where a corner is. Handing a distance map to a model expecting shading
        gets the two confused, and every surface at one distance reads as one flat plane.

        The normals were already being computed for the edge map, so this costs one extra pass
        over data we have, not another ray cast.

        Two-light setup: a key light over the camera's shoulder and a weak fill from below, so
        that no surface goes fully black and the model still has structure to work with in the
        shadows.
        """
        finite = np.isfinite(depth_grid)
        shade = Room3DBuilder._shade_from_normals(normal_grid, depth_grid, finite, view_dir)

        # MATERIAL TINT. Distance tells the model where a surface is, never what it is. Fed a
        # uniformly grey render the model has no way to know a door-shaped gap is a door, and
        # it does not guess: on this apartment it rendered the walls, the doorway and the
        # returns as one continuous run of oak panelling, because the prompt said oak. Depth
        # correlation 0.975 and edge recall 0.958 -- every surface in the right place, several
        # of them the wrong object.
        #
        # Shifting each material's base brightness carries that missing information in the one
        # channel the model already reads. Shading is preserved: the tint scales the lit value
        # rather than replacing it, so corners still read as corners.
        if materials is not None:
            tint = np.ones_like(shade)
            for value, factor in (tint_map or MATERIAL_TINT).items():
                tint = np.where(materials == value, factor, tint)
            shade = shade * tint

        # A FIXED gain, never a per-frame auto-level. Normalising each frame to its own brightest
        # pixel is the same mistake the depth encoding made and was fixed for: it makes a given
        # surface a different grey depending on what else is in shot, so walking past a doorway
        # rescales the whole image and the model reads a hard cut. A constant multiplier lifts the
        # midtones without ever making one frame's grey mean something different from the next's.
        shade = shade * exposure

        image = np.where(finite, np.clip(shade, 0, 1) * 255.0, 255.0)   # misses -> white void
        Image.fromarray(image.astype(np.uint8)).save(path)

    @staticmethod
    def _save_material_render(depth_grid: np.ndarray, normal_grid: np.ndarray, path: str,
                              view_dir: np.ndarray = None, materials: np.ndarray = None,
                              exposure: float = 1.0, sun_dir: np.ndarray = None,
                              sun_visible: np.ndarray = None) -> None:
        """
        Shaded render in plausible colour: the lighting of the clay pass, the identity of the
        semantic pass, in one image.

        The two existing passes each throw away what the other keeps. Clay has light and shadow
        but every surface is the same grey, so a door and a wall differ only by a tint step that
        good lighting can wipe out. The semantic pass has unambiguous identity but no shading at
        all, so it reads as a diagram rather than a room.

        This matters specifically for a model that is shown the clip as a REFERENCE rather than
        wired into a control branch. A control branch consumes a depth map happily because it was
        trained to. A reference is understood the way a person understands a photograph, and the
        strongest signal that a floor is wood and a wall is plaster is that one of them is brown.

        Rays that hit nothing become sky rather than white. In the grey passes a miss is pure
        white, which is also what a brightly lit wall looks like -- so a window and a hole in the
        model are indistinguishable, and windows in this drawing ARE holes between sill and head.
        A cool bright blue says "outside" in a way no grey value can.
        """
        finite = np.isfinite(depth_grid)
        shade = Room3DBuilder._shade_from_normals(normal_grid, depth_grid, finite, view_dir,
                                                  up_fill=CEILING_BOUNCE)

        height, width = depth_grid.shape
        base = np.zeros((height, width, 3), dtype=float)
        for value, rgb in MATERIAL_BASE_RGB.items():
            mask = materials == value if materials is not None else np.zeros_like(finite)
            for channel in range(3):
                base[:, :, channel] = np.where(mask, rgb[channel], base[:, :, channel])
        # Anything unlabelled still gets a surface colour rather than black.
        unlabelled = finite & (base.sum(axis=2) == 0)
        for channel in range(3):
            base[:, :, channel] = np.where(unlabelled, MATERIAL_BASE_RGB["wall"][channel],
                                           base[:, :, channel])

        if sun_dir is None or sun_visible is None:
            lit = base * (shade * exposure)[:, :, None]
        else:
            # Warm sun, cool sky. The split is the whole archviz signature: a room lit by one
            # white lamp reads as a diagram, the same room lit warm-from-the-window and
            # cool-in-shadow reads as a photograph. It is also physically what happens --
            # direct sunlight is around 5000K and the shadows are filled by blue sky.
            facing = np.clip((normal_grid * -sun_dir).sum(axis=2), 0.0, 1.0)
            direct = facing * sun_visible                      # lit only where nothing blocks
            sky = shade                                        # the existing soft ambient

            warm = np.array(SUN_RGB, dtype=float) / 255.0
            cool = np.array(SKY_FILL_RGB, dtype=float) / 255.0
            light = (SUN_STRENGTH * direct[:, :, None] * warm
                     + SKY_STRENGTH * sky[:, :, None] * cool)
            lit = base * light * exposure

        image = np.where(finite[:, :, None], np.clip(lit, 0, 255), np.array(SKY_RGB, dtype=float))
        Image.fromarray(image.astype(np.uint8)).save(path)

    @staticmethod
    def _save_semantic_map(depth_grid: np.ndarray, materials: np.ndarray, path: str) -> None:
        """
        Flat colour by material -- no shading at all. The one pass that answers "what is this
        surface" without the answer being confounded by how brightly it happens to be lit.

        Shading and identity fight each other in a single grey channel: a well-lit door and a
        shadowed wall can land on the same value, which is exactly the confusion the tinting was
        introduced to fix and only partly does. Colour separates them because hue survives
        lighting. Rays that hit nothing stay black here rather than white, so a genuine hole in
        the model is never mistaken for a bright surface.
        """
        finite = np.isfinite(depth_grid)
        height, width = depth_grid.shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for value, rgb in MATERIAL_RGB.items():
            mask = finite & (materials == value)
            for channel in range(3):
                image[:, :, channel] = np.where(mask, rgb[channel], image[:, :, channel])
        Image.fromarray(image).save(path)

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
