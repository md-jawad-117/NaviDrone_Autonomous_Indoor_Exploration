"""Loads the converted house mesh (see scripts/convert_house_glb.py) as a static
PyBullet body: a visual mesh with the roof stripped out (so the interior is
actually visible) plus a separate collision mesh that keeps the roof (so the
drone still can't fly out through the ceiling)."""
import os

import pybullet as p

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "house")
_VISUAL_OBJ = os.path.join(_ASSET_DIR, "house_visual.obj")
_COLLISION_OBJ = os.path.join(_ASSET_DIR, "house_collision.obj")


class HouseWorld:
    """Static mesh-based world, alternative to the procedural SimWorld."""

    def __init__(self, config: dict, client_id: int):
        """
        Args:
            config: parsed contents of config/house_config.yaml under 'world'.
            client_id: PyBullet physics client id.
        """
        self.config = config
        self.client_id = client_id
        self.body_ids = []

        for path in (_VISUAL_OBJ, _COLLISION_OBJ):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found. Run `python scripts/convert_house_glb.py` "
                    "first to generate the house OBJ meshes from the GLB source."
                )

        # GEOM_FORCE_CONCAVE_TRIMESH is required here: without it, PyBullet builds a
        # convex hull of the whole mesh, which for an enclosed room turns the entire
        # interior into one solid collision blob. Only valid for static (mass=0) bodies.
        collision_shape = p.createCollisionShape(
            p.GEOM_MESH, fileName=_COLLISION_OBJ, flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=self.client_id,
        )
        visual_shape = p.createVisualShape(
            p.GEOM_MESH, fileName=_VISUAL_OBJ, physicsClientId=self.client_id,
        )
        house_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 0],
            physicsClientId=self.client_id,
        )
        self.body_ids.append(house_body)

    def get_bounds(self):
        """
        Returns (x_min, x_max, y_min, y_max) of the house's actual collision
        geometry, via PyBullet's AABB query -- so occupancy-grid/planning
        bounds can be derived automatically instead of hardcoded per house
        (this project's original bounds were hand-measured for one specific
        house file and would be wrong for a different one).
        """
        aabb_min, aabb_max = p.getAABB(self.body_ids[0], physicsClientId=self.client_id)
        return aabb_min[0], aabb_max[0], aabb_min[1], aabb_max[1]
