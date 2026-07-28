"""PyBullet world builder: room, ground, walls and obstacles for the drone-nav sim."""
import os
import random
from typing import Optional

import pybullet as p
import pybullet_data


DIFFICULTY_OBSTACLE_COUNT = {
    "sparse": 6,
    "moderate": 18,
    "dense": 35,
}

# Varied colors so ORB/feature-based VO has something to lock onto later.
_OBSTACLE_COLORS = [
    (0.85, 0.25, 0.2, 1.0),
    (0.2, 0.5, 0.85, 1.0),
    (0.9, 0.75, 0.15, 1.0),
    (0.3, 0.7, 0.3, 1.0),
    (0.6, 0.3, 0.7, 1.0),
    (0.9, 0.5, 0.2, 1.0),
]


class SimWorld:
    """Builds and owns the static PyBullet scene (ground, walls, obstacles)."""

    def __init__(self, config: dict, client_id: int):
        """
        Args:
            config: parsed contents of config/env_config.yaml under the 'world' key.
            client_id: PyBullet physics client id (from p.connect(...)).
        """
        self.config = config
        self.client_id = client_id
        self.room_size = config.get("room_size", [10.0, 10.0])
        self.wall_height = config.get("wall_height", 3.0)
        self.difficulty = config.get("difficulty", "sparse")
        self.seed = config.get("seed", 42)
        self.ground_texture = config.get("ground_texture", "checker")

        self.body_ids = []
        self._build()

    def _build(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._add_ground()
        self._add_walls()
        self._add_obstacles()

    def _add_ground(self):
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        if self.ground_texture == "checker":
            tex_path = os.path.join(pybullet_data.getDataPath(), "checker_blue.png")
            if os.path.exists(tex_path):
                tex_id = p.loadTexture(tex_path)
                p.changeVisualShape(plane_id, -1, textureUniqueId=tex_id,
                                     physicsClientId=self.client_id)
        self.body_ids.append(plane_id)

    def _add_walls(self):
        sx, sy = self.room_size
        h = self.wall_height
        thickness = 0.1
        wall_color = (0.75, 0.75, 0.78, 1.0)

        # (center_xyz, half_extents)
        specs = [
            ((0, sy / 2, h / 2), (sx / 2, thickness / 2, h / 2)),
            ((0, -sy / 2, h / 2), (sx / 2, thickness / 2, h / 2)),
            ((sx / 2, 0, h / 2), (thickness / 2, sy / 2, h / 2)),
            ((-sx / 2, 0, h / 2), (thickness / 2, sy / 2, h / 2)),
        ]
        for center, half_extents in specs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                          physicsClientId=self.client_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                       rgbaColor=wall_color, physicsClientId=self.client_id)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis, basePosition=center,
                                      physicsClientId=self.client_id)
            self.body_ids.append(body)

    def _add_obstacles(self):
        rng = random.Random(self.seed)
        count = DIFFICULTY_OBSTACLE_COUNT.get(self.difficulty, 6)
        sx, sy = self.room_size
        margin = 0.8
        min_h, max_h = 0.4, 2.2

        for i in range(count):
            x = rng.uniform(-sx / 2 + margin, sx / 2 - margin)
            y = rng.uniform(-sy / 2 + margin, sy / 2 - margin)
            # keep the spawn area near the origin clear
            if abs(x) < 1.2 and abs(y) < 1.2:
                continue

            shape_kind = rng.choice(["box", "cylinder"])
            color = _OBSTACLE_COLORS[i % len(_OBSTACLE_COLORS)]
            height = rng.uniform(min_h, max_h)

            if shape_kind == "box":
                half_extents = (rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.4), height / 2)
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                              physicsClientId=self.client_id)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                           rgbaColor=color, physicsClientId=self.client_id)
            else:
                radius = rng.uniform(0.15, 0.35)
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height,
                                              physicsClientId=self.client_id)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                           rgbaColor=color, physicsClientId=self.client_id)

            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis,
                                      basePosition=(x, y, height / 2),
                                      physicsClientId=self.client_id)
            self.body_ids.append(body)
