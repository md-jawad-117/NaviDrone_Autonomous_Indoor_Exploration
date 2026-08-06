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
import gc
import os
import re
import sys
import tempfile

import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC_GLB = os.path.join(ROOT, "3d Model", "room.glb")
OUT_DIR = os.path.join(ROOT, "env", "assets", "house")

# See decimate_scene_geometry's docstring: PyBullet's visual mesh loader can
# fail to render at all well before a mesh gets anywhere near this dense --
# a ~1.45M-face house rendered as entirely blank with no error. Two known
# empirical data points on this project: this project's original house
# (542,085 faces) rendered perfectly fine with NO decimation involved at all;
# a heavily furnished variant at ~1.45M faces rendered entirely blank.
#
# REVERTED to 800k on 2026-08-06 after briefly trying 1.5M: that let a
# 1,462,925-face house skip decimation entirely (since it was just under the
# new threshold), and the undecimated mesh then ran out of memory inside
# build_scene()'s own geometry-copying step -- a different crash than the
# original blank-render bug, but the same root cause (too much raw geometry
# held/copied in memory at once).
#
# EXPERIMENT (2026-08-06): trying 1.0M -- still comfortably below the
# ~1.2M ceiling noted above and well below the known-bad ~1.45M blank-render
# point, but higher than the confirmed-working 800k, to see if furniture
# keeps a bit more detail without tripping either failure mode. Revert to
# 800k if this house goes blank or runs out of memory again.
TARGET_TOTAL_FACES = 1_200_000

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


_STRUCTURAL_NAME_KEYWORDS = ("wall", "floor", "ceiling", "roof", "trim", "baseboard", "window", "door")


def _is_structural_name(node_name: str) -> bool:
    lname = node_name.lower()
    return lname.startswith("room__") or any(kw in lname for kw in _STRUCTURAL_NAME_KEYWORDS)


def _decimate_preserving_color(geom: "trimesh.Trimesh", target_faces: int) -> "trimesh.Trimesh":
    """Quadric-decimates geom to target_faces, then recovers each new vertex's
    color via nearest-neighbor lookup back to the pre-decimation mesh.

    trimesh.simplify_quadric_decimation silently resets vertex_colors to a
    flat default grey (confirmed empirically -- not documented behavior),
    which is why a first attempt at this rendered every decimated object as
    uniform grey instead of its baked material color."""
    try:
        orig_colors = geom.visual.to_color().vertex_colors.copy()
    except Exception:
        orig_colors = None
    orig_vertices = geom.vertices.copy()

    simplified = geom.simplify_quadric_decimation(face_count=target_faces)

    if orig_colors is not None and len(orig_vertices) > 0 and len(simplified.vertices) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(orig_vertices)
        _, idx = tree.query(simplified.vertices)
        simplified.visual = trimesh.visual.ColorVisuals(
            mesh=simplified, vertex_colors=orig_colors[idx])

    return simplified


def decimate_scene_geometry(scene: trimesh.Scene, target_total_faces: int, min_faces_per_mesh: int = 20,
                             min_retention_fraction: float = 0.35):
    """
    Reduces total face count via quadric decimation, applied ONLY to
    furniture/prop geometry -- architectural pieces (walls, floor, trim,
    baseboard, window frames; detected by name, see _is_structural_name) are
    left at full resolution. Mutates scene.geometry in place.

    Why this exists: PyBullet's createVisualShape doesn't just get slow with
    a very dense combined mesh, it can fail to render at all -- confirmed on
    this project: a furnished house GLB exported at ~1.45M faces for the
    visual mesh alone, and the result was a completely blank main viewport,
    blank synthetic-camera RGB panel, and depth reading as pure far-plane for
    every pixel (so 0 valid points ever reached the SLAM/mapping pipeline),
    with no error raised anywhere -- collision mesh loading worked fine at an
    even larger face count, since Bullet's physics-mesh path handles large
    meshes differently than the visual-shape path.

    A first version of this decimated every mesh uniformly (structural
    geometry included) and broke things worse than the original problem:
    flat grey rendering (see _decimate_preserving_color) and a badly holed
    floor/walls that made the occupancy-grid raycasts used for room
    detection find almost nothing. Architectural geometry is comparatively
    cheap face-count-wise anyway (a few hundred thousand faces here) and is
    exactly the geometry raycasting depends on being topologically intact --
    the actual bloat is almost always in furniture/prop meshes, which are
    visually forgiving to simplify.

    A second issue showed up even with structural geometry excluded: at this
    house's actual ratio (~20% of original faces per furniture item to hit a
    400k budget), thin double-walled objects -- an appliance's outer shell,
    where the inner and outer surface sit very close together -- got holes
    punched clean through them. Quadric decimation optimizes for geometric
    error, not "stay watertight," and doesn't know it's collapsing a shell
    rather than a solid. min_retention_fraction is a hard floor (default:
    never remove more than 65% of any single mesh's faces, i.e. always keep
    >= 35%) applied on top of the global ratio, so a handful of especially
    detailed objects don't get sacrificed just to hit the nominal total --
    the actual post-decimation total is allowed to land above
    target_total_faces as a result, and is printed so you can see by how much.

    Simplifying is applied once, right after loading, so both the visual and
    collision exports derive from the same (mostly) reduced geometry.
    """
    geom_node_names = {}
    for node_name in scene.graph.nodes_geometry:
        _, geom_name = scene.graph[node_name]
        geom_node_names.setdefault(geom_name, []).append(node_name)

    structural_faces = 0
    decimatable = {}  # geom_name -> n_faces
    for geom_name, geom in scene.geometry.items():
        n_faces = len(geom.faces)
        names = geom_node_names.get(geom_name, [geom_name])
        if any(_is_structural_name(n) for n in names):
            structural_faces += n_faces
        else:
            decimatable[geom_name] = n_faces

    decimatable_total = sum(decimatable.values())
    total = structural_faces + decimatable_total
    print(f"Scene: {total} faces total ({structural_faces} structural -- kept at full resolution; "
          f"{decimatable_total} across {len(decimatable)} furniture/prop meshes -- decimatable)")

    if total <= target_total_faces:
        print(f"Already within the {target_total_faces} target; skipping decimation.")
        return

    budget_for_decimatable = max(0, target_total_faces - structural_faces)
    if decimatable_total == 0 or budget_for_decimatable <= 0:
        print(f"WARNING: structural geometry alone ({structural_faces} faces) already meets/exceeds the "
              f"{target_total_faces} target; nothing decimatable can bring this under budget without "
              f"touching structural geometry, which this intentionally avoids. Consider raising "
              f"target_total_faces instead.")
        return

    ratio = budget_for_decimatable / decimatable_total
    print(f"Decimating furniture/prop meshes: {decimatable_total} faces -> ~{budget_for_decimatable} "
          f"target (ratio {ratio:.4f}, floored at {min_retention_fraction:.0%} retention per mesh)")
    for geom_name, n_faces in decimatable.items():
        if n_faces <= min_faces_per_mesh:
            continue
        target = max(min_faces_per_mesh, int(n_faces * ratio), int(n_faces * min_retention_fraction))
        if target >= n_faces:
            continue
        try:
            scene.geometry[geom_name] = _decimate_preserving_color(scene.geometry[geom_name], target)
        except Exception as e:
            print(f"  WARNING: could not simplify '{geom_name}' ({n_faces} faces): {e}")

    new_total = sum(len(g.faces) for g in scene.geometry.values())
    print(f"Decimation done: {new_total} faces total.")

    new_total = sum(len(g.faces) for g in scene.geometry.values())
    print(f"Decimation done: {new_total} faces total.")


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
    def _strip_color(bl):
        if bl.startswith("v "):
            parts = bl.split()
            return f"v {parts[1]} {parts[2]} {parts[3]}\n"
        return bl

    # Streams the file line-by-line instead of loading it wholesale (as an
    # earlier version did via f.readlines()) -- fine for small houses, but a
    # multi-hundred-MB / multi-million-line OBJ (dense source meshes) blew
    # past available memory just holding that many Python line objects twice
    # over (once as the read list, once as the rewritten list). The only
    # lookahead needed per vertex-color-only object block is up to its first
    # "v " line (to read the color) -- typically the very first line in the
    # block -- so buffering stays tiny even if the block itself has millions
    # of vertices after that point.
    new_materials = []  # (name, (r, g, b))
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".obj.tmp")
    os.close(tmp_fd)

    with open(obj_path, "r") as fin, open(tmp_path, "w") as fout:
        line = fin.readline()
        while line:
            if not line.startswith("o "):
                fout.write(line)
                line = fin.readline()
                continue

            obj_name = line.strip().split(maxsplit=1)[1]
            fout.write(line)
            line = fin.readline()

            if line.startswith("usemtl"):
                # Already has a real material -- leave this object's block untouched.
                fout.write(line)
                line = fin.readline()
                continue

            pending = []
            color = None
            while line and not line.startswith("o "):
                pending.append(line)
                if color is None and line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 7:
                        color = tuple(float(x) for x in parts[4:7])
                    line = fin.readline()
                    break
                line = fin.readline()

            mtl_name = f"AutoColor_{obj_name}"
            fout.write(f"usemtl {mtl_name}\n")
            new_materials.append((mtl_name, color or (0.5, 0.5, 0.5)))

            for bl in pending:
                fout.write(_strip_color(bl))
            # Rest of the block (after the line the color was read from)
            # streams straight through, no further buffering needed.
            while line and not line.startswith("o "):
                fout.write(_strip_color(line))
                line = fin.readline()

    os.replace(tmp_path, obj_path)

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
    # Streamed line-by-line for the same reason as the pass above -- no
    # lookahead needed here at all, just O(1) running state, so there's no
    # reason to ever hold the full file in memory.
    vt_index_by_name = {name: i + 1 for i, name in enumerate(materials)}  # OBJ indices are 1-based
    vt_lines = [f"vt {material_uv[name][0]:.6f} {material_uv[name][1]:.6f}\n" for name in materials]
    face_token_re = re.compile(r"^(\d+)(?:/(\d*))?(?:/(\d*))?$")

    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".obj.tmp2")
    os.close(tmp_fd)

    current_mtl = None
    inserted_vt = False
    with open(obj_path, "r") as fin, open(tmp_path, "w") as fout:
        for line in fin:
            stripped = line.strip()
            if stripped.startswith("vt "):
                continue  # drop old vt lines -- replaced wholesale below
            if stripped.startswith("mtllib") and not inserted_vt:
                fout.write(line)
                fout.writelines(vt_lines)
                inserted_vt = True
                continue
            if stripped.startswith("usemtl "):
                current_mtl = stripped.split(maxsplit=1)[1]
                fout.write(line)
                continue
            if stripped.startswith("f ") and current_mtl in vt_index_by_name:
                vt_idx = vt_index_by_name[current_mtl]
                new_tokens = []
                for token in stripped.split()[1:]:
                    m = face_token_re.match(token)
                    v_idx, _, vn_idx = m.groups()
                    new_tokens.append(f"{v_idx}/{vt_idx}/{vn_idx}" if vn_idx else f"{v_idx}/{vt_idx}")
                fout.write("f " + " ".join(new_tokens) + "\n")
            else:
                fout.write(line)

    os.replace(tmp_path, obj_path)


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

    decimate_scene_geometry(scene, target_total_faces=TARGET_TOTAL_FACES)

    exclude_from_visual = detect_ceiling_nodes(scene)
    print(f"Excluding from visual mesh: {exclude_from_visual}")

    visual_path = os.path.join(OUT_DIR, "house_visual.obj")
    collision_path = os.path.join(OUT_DIR, "house_collision.obj")

    # Build + export + free one derived scene at a time instead of holding
    # both visual_scene and collision_scene (each its own full baked-geometry
    # copy) in memory simultaneously -- collision includes nearly everything
    # visual does plus more, so building both upfront nearly doubles peak
    # memory right before the heaviest step (trimesh's own OBJ export), which
    # is exactly where a dense/heavy source mesh ran out of memory.
    visual_scene = build_scene(scene, exclude_from_visual)
    visual_parts = len(visual_scene.geometry)
    visual_scene.export(visual_path)
    del visual_scene
    gc.collect()

    collision_scene = build_scene(scene, exclude_nodes=set())
    collision_parts = len(collision_scene.geometry)
    collision_bounds = collision_scene.bounds.tolist()
    collision_scene.export(collision_path)
    del collision_scene
    gc.collect()

    mtl_path = os.path.join(OUT_DIR, "material.mtl")
    _bake_material_colors_as_textures(visual_path, mtl_path, OUT_DIR)
    print(f"Baked material Kd colors into textures for PyBullet compatibility -> {mtl_path}")

    print(f"Wrote visual mesh   ({visual_parts} parts) -> {visual_path}")
    print(f"Wrote collision mesh ({collision_parts} parts) -> {collision_path}")
    print(f"Overall bounds (collision mesh): {collision_bounds}")


if __name__ == "__main__":
    main()
