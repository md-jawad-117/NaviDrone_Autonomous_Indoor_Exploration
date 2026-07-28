"""
One-time conversion: a house GLB -> OBJ meshes PyBullet can load directly.

PyBullet's mesh loader (createVisualShape/createCollisionShape with GEOM_MESH)
does not read GLB, so we split the house into two OBJ exports via trimesh:

  - house_visual.obj:    everything except the ceiling/roof, so the interior
                          is actually visible from above/inside in the GUI/camera.
  - house_collision.obj: everything, including the ceiling, so the drone still
                          can't fly out through where it should be.

Both are static (mass=0) concave meshes, which PyBullet handles fine for
non-dynamic level geometry.

PyBullet's OBJ loader ignores flat `Kd` diffuse colors with no texture map
(verified: a solid-Kd triangle renders mid-grey, not its Kd color) -- it only
picks up color from an actual texture image. Since this house's materials are
flat colors (no texture), _bake_material_colors_as_textures generates a tiny
solid-color PNG per material and rewires the visual OBJ to reference it via a
single dummy UV coordinate (fine since the texture is uniform, so any UV
samples the same color).

Run:
    python scripts/convert_house_glb.py [path/to/house.glb]

If no path is given, defaults to "3d Model/room.glb" (this project's original
house). The ceiling is auto-detected (by name first, falling back to a
geometric heuristic: a thin, high, wide slab) -- see detect_ceiling_nodes --
so this should work unmodified on a different house file.
"""
import os
import re
import sys

import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC_GLB = os.path.join(ROOT, "3d Model", "room.glb")
OUT_DIR = os.path.join(ROOT, "env", "assets", "house")

# glTF's scene graph is Y-up (confirmed: scene.bounds shows a ~2.5 m span on Y,
# matching ceiling height); PyBullet is Z-up. +90 deg rotation about X maps
# Y-up -> Z-up: (x, y, z) -> (x, -z, y).
_YUP_TO_ZUP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])


def detect_ceiling_nodes(scene: trimesh.Scene) -> set:
    """
    Auto-detects which node(s) represent the ceiling/roof, to exclude from the
    visual mesh (so the interior is visible) while keeping them in the
    collision mesh (so the drone can't fly through where it should be).

    Tries name matching first (covers this project's "roof" node and common
    naming elsewhere). Falls back to geometry for houses that don't name it
    that way: a ceiling is thin (small Z extent), high (near the top of the
    whole scene), and wide (spans most of the building's footprint) --
    distinguishing it from small thin objects like shelves or trim.
    """
    name_matches = {
        node_name for node_name in scene.graph.nodes_geometry
        if any(kw in node_name.lower() for kw in ("roof", "ceiling"))
    }
    if name_matches:
        return name_matches

    node_bounds = {}
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        corners = trimesh.bounds.corners(geom.bounds)
        world_corners = trimesh.transform_points(corners, _YUP_TO_ZUP @ transform)
        node_bounds[node_name] = (world_corners.min(axis=0), world_corners.max(axis=0))

    if not node_bounds:
        return set()

    overall_min = np.min([b[0] for b in node_bounds.values()], axis=0)
    overall_max = np.max([b[1] for b in node_bounds.values()], axis=0)
    footprint_area = (overall_max[0] - overall_min[0]) * (overall_max[1] - overall_min[1])
    height_range = overall_max[2] - overall_min[2]

    candidates = set()
    for node_name, (lo, hi) in node_bounds.items():
        z_thickness = hi[2] - lo[2]
        z_center = (hi[2] + lo[2]) / 2
        node_footprint_area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        is_thin = z_thickness < max(0.15, 0.05 * height_range)
        is_high = z_center > overall_min[2] + 0.85 * height_range
        is_wide = node_footprint_area > 0.3 * footprint_area
        if is_thin and is_high and is_wide:
            candidates.add(node_name)

    if candidates:
        print(f"No node named 'roof'/'ceiling' found; geometric fallback "
              f"identified {candidates} as the ceiling.")
    else:
        print("WARNING: could not identify a ceiling node by name or geometry -- "
              "the visual mesh will include it (interior may be hard to see from above).")
    return candidates


def build_scene(scene: trimesh.Scene, exclude_nodes: set) -> trimesh.Scene:
    """Returns a new Scene containing only nodes not in exclude_nodes, with
    each geometry's world transform baked in (since multiple nodes can share
    the same geometry, e.g. repeated chair/bed meshes), converted to Z-up."""
    out = trimesh.Scene()
    for node_name in scene.graph.nodes_geometry:
        if node_name in exclude_nodes:
            continue
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name].copy()
        geom.apply_transform(_YUP_TO_ZUP @ transform)
        out.add_geometry(geom, node_name=node_name)
    return out


def _assign_materials_to_colorvisuals_objects(obj_path: str, mtl_path: str, out_dir: str):
    """
    Some objects in this house (floor, walls, window_frames, baseboard, top_trim)
    never had a named material in the source file -- trimesh exports them as
    bare per-vertex "v x y z r g b" color lines instead, with no "usemtl" of
    their own. Two problems result: (1) that "v ... r g b" extension isn't
    something PyBullet's loader reliably honors, and (2) an "o" block with no
    usemtl inherits whatever material was last declared, so these ended up
    rendering as a leftover blue chair-adjacent material instead of their own
    (verified: file's own vertex colors are tan walls / grey floor, but
    PyBullet actually rendered both blue). Fix: give each such object its own
    real material, baked from its own vertex color, and strip the non-standard
    color columns back to plain "v x y z" so every vertex line in the file has
    a consistent, standard format.
    """
    with open(obj_path, "r") as f:
        obj_lines = f.readlines()

    new_obj_lines = []
    new_materials = []  # (name, (r, g, b))
    i, n = 0, len(obj_lines)

    while i < n:
        line = obj_lines[i]
        if not line.startswith("o "):
            new_obj_lines.append(line)
            i += 1
            continue

        obj_name = line.strip().split(maxsplit=1)[1]
        new_obj_lines.append(line)
        i += 1

        if i < n and obj_lines[i].strip().startswith("usemtl"):
            # Already has a real material -- leave this object's block untouched.
            continue

        block_start = i
        color = None
        while i < n and not obj_lines[i].startswith("o "):
            if color is None and obj_lines[i].startswith("v "):
                parts = obj_lines[i].split()
                if len(parts) >= 7:
                    color = tuple(float(x) for x in parts[4:7])
            i += 1
        block_end = i

        mtl_name = f"AutoColor_{obj_name}"
        new_obj_lines.append(f"usemtl {mtl_name}\n")
        new_materials.append((mtl_name, color or (0.5, 0.5, 0.5)))

        for bl in obj_lines[block_start:block_end]:
            if bl.startswith("v "):
                parts = bl.split()
                new_obj_lines.append(f"v {parts[1]} {parts[2]} {parts[3]}\n")
            else:
                new_obj_lines.append(bl)

    if new_materials:
        # Kd only -- _bake_material_colors_as_textures (called right after
        # this) bakes every material's Kd into the shared atlas texture and
        # writes the map_Kd line itself, so no per-material texture needed here.
        with open(mtl_path, "a") as f:
            for name, (r, g, b) in new_materials:
                f.write(f"\nnewmtl {name}\n")
                f.write("Ka 0.40000000 0.40000000 0.40000000\n")
                f.write(f"Kd {r:.8f} {g:.8f} {b:.8f}\n")
                f.write("Ks 0.40000000 0.40000000 0.40000000\n")
                f.write("Ns 1.00000000\n")

    with open(obj_path, "w") as f:
        f.writelines(new_obj_lines)

    print(f"Assigned {len(new_materials)} new materials to previously vertex-color-only objects: "
          f"{[name for name, _ in new_materials]}")


def _bake_material_colors_as_textures(obj_path: str, mtl_path: str, out_dir: str):
    """Rewrites obj_path/mtl_path so every material's Kd color renders correctly
    in PyBullet (see module docstring for why this is needed).

    All materials share ONE small texture atlas (one swatch per material)
    rather than one PNG per material. Originally each material got its own
    tiny solid-color PNG, which worked fine with a handful of materials, but
    once this house had 13 distinct per-material textures, PyBullet's OBJ
    loader stopped respecting per-face materials and rendered the *entire*
    mesh with just one material's texture (confirmed: the whole house came
    out as flat navy "Chair" blue) -- some limit on distinct textures per
    visual shape. A shared atlas sidesteps that limit entirely: there's only
    ever one texture reference, no matter how many material colors exist;
    each material just gets its own UV coordinate pointing at its own patch.
    """
    _assign_materials_to_colorvisuals_objects(obj_path, mtl_path, out_dir)

    with open(mtl_path, "r") as f:
        mtl_lines = f.readlines()

    materials = []
    kd_by_name = {}
    cleaned_mtl_lines = []
    current_mtl = None
    for line in mtl_lines:
        stripped = line.strip()
        if stripped.startswith("newmtl "):
            current_mtl = stripped.split(maxsplit=1)[1]
            materials.append(current_mtl)
            cleaned_mtl_lines.append(line)
        elif stripped.startswith("Kd ") and current_mtl is not None:
            kd_by_name[current_mtl] = tuple(float(x) for x in stripped.split()[1:4])
            cleaned_mtl_lines.append(line)
        elif stripped.startswith("map_Kd"):
            continue  # drop any prior per-material texture reference
        else:
            cleaned_mtl_lines.append(line)

    swatch_px = 8
    atlas = np.zeros((swatch_px, swatch_px * len(materials), 3), dtype=np.uint8)
    material_uv = {}
    for i, name in enumerate(materials):
        r, g, b = kd_by_name.get(name, (0.5, 0.5, 0.5))
        atlas[:, i * swatch_px:(i + 1) * swatch_px] = [int(round(r * 255)),
                                                         int(round(g * 255)),
                                                         int(round(b * 255))]
        material_uv[name] = ((i + 0.5) / len(materials), 0.5)

    atlas_name = "material_atlas.png"
    Image.fromarray(atlas).save(os.path.join(out_dir, atlas_name))

    final_mtl_lines = []
    current_mtl = None
    for line in cleaned_mtl_lines:
        stripped = line.strip()
        final_mtl_lines.append(line)
        if stripped.startswith("newmtl "):
            current_mtl = stripped.split(maxsplit=1)[1]
        elif stripped.startswith("Kd ") and current_mtl is not None:
            final_mtl_lines.append(f"map_Kd {atlas_name}\n")

    with open(mtl_path, "w") as f:
        f.writelines(final_mtl_lines)

    # Rewrite the obj: one "vt" per material (right after mtllib), then
    # retarget every face's texture-coordinate index to whichever material
    # is currently active (tracked via "usemtl"), replacing any old vt index
    # (e.g. the single shared dummy point a prior version of this script used).
    with open(obj_path, "r") as f:
        obj_lines = f.readlines()

    vt_index_by_name = {name: i + 1 for i, name in enumerate(materials)}  # OBJ indices are 1-based
    vt_lines = [f"vt {material_uv[name][0]:.6f} {material_uv[name][1]:.6f}\n" for name in materials]
    face_token_re = re.compile(r"^(\d+)(?:/(\d*))?(?:/(\d*))?$")

    new_obj_lines = []
    current_mtl = None
    inserted_vt = False
    for line in obj_lines:
        stripped = line.strip()
        if stripped.startswith("vt "):
            continue  # drop old vt lines -- replaced wholesale below
        if stripped.startswith("mtllib") and not inserted_vt:
            new_obj_lines.append(line)
            new_obj_lines.extend(vt_lines)
            inserted_vt = True
            continue
        if stripped.startswith("usemtl "):
            current_mtl = stripped.split(maxsplit=1)[1]
            new_obj_lines.append(line)
            continue
        if stripped.startswith("f ") and current_mtl in vt_index_by_name:
            vt_idx = vt_index_by_name[current_mtl]
            new_tokens = []
            for token in stripped.split()[1:]:
                m = face_token_re.match(token)
                v_idx, _, vn_idx = m.groups()
                new_tokens.append(f"{v_idx}/{vt_idx}/{vn_idx}" if vn_idx else f"{v_idx}/{vt_idx}")
            new_obj_lines.append("f " + " ".join(new_tokens) + "\n")
        else:
            new_obj_lines.append(line)

    with open(obj_path, "w") as f:
        f.writelines(new_obj_lines)


def main():
    src_glb = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC_GLB
    if not os.path.exists(src_glb):
        raise FileNotFoundError(f"House model not found: {src_glb}")
    os.makedirs(OUT_DIR, exist_ok=True)

    scene = trimesh.load(src_glb)
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)

    all_nodes = set(scene.graph.nodes_geometry)
    print(f"Loaded {len(all_nodes)} nodes from {src_glb}")

    exclude_from_visual = detect_ceiling_nodes(scene)
    print(f"Excluding from visual mesh: {exclude_from_visual}")

    visual_scene = build_scene(scene, exclude_from_visual)
    collision_scene = build_scene(scene, exclude_nodes=set())

    visual_path = os.path.join(OUT_DIR, "house_visual.obj")
    collision_path = os.path.join(OUT_DIR, "house_collision.obj")

    visual_scene.export(visual_path)
    collision_scene.export(collision_path)

    mtl_path = os.path.join(OUT_DIR, "material.mtl")
    _bake_material_colors_as_textures(visual_path, mtl_path, OUT_DIR)
    print(f"Baked material Kd colors into textures for PyBullet compatibility -> {mtl_path}")

    print(f"Wrote visual mesh   ({len(visual_scene.geometry)} parts) -> {visual_path}")
    print(f"Wrote collision mesh ({len(collision_scene.geometry)} parts) -> {collision_path}")
    print(f"Overall bounds (collision mesh): {collision_scene.bounds.tolist()}")


if __name__ == "__main__":
    main()
