"""Quadrotor wrapper: velocity-level flight control + onboard RGB-D camera."""
import os
import math
from typing import Tuple

import numpy as np
import pybullet as p

_URDF_PATH = os.path.join(os.path.dirname(__file__), "assets", "quadrotor.urdf")


class DroneRobot:
    """
    A kinematically-flown quadrotor: teleop commands set body-frame velocities
    directly (via resetBaseVelocity) rather than driving individual rotor thrusts.
    This keeps flight trivially stable so early work can focus on perception/SLAM,
    not attitude control. Collision shapes are still active, so it can't fly
    through obstacles or walls.
    """

    def __init__(self, config: dict, client_id: int):
        """
        Args:
            config: parsed contents of config/env_config.yaml under the 'drone' key.
            client_id: PyBullet physics client id.
        """
        self.config = config
        self.client_id = client_id

        self.max_linear_speed = config.get("max_linear_speed", 1.5)
        self.max_vertical_speed = config.get("max_vertical_speed", 1.0)
        self.max_yaw_rate = math.radians(config.get("max_yaw_rate_deg", 60.0))

        start_pos = config.get("start_position", [0.0, 0.0, 1.0])
        start_yaw = math.radians(config.get("start_yaw_deg", 0.0))
        start_orn = p.getQuaternionFromEuler([0, 0, start_yaw])

        if not os.path.exists(_URDF_PATH):
            raise FileNotFoundError(
                f"Quadrotor URDF not found at {_URDF_PATH}. "
                "Check env/assets/quadrotor.urdf exists."
            )

        self.body_id = p.loadURDF(
            _URDF_PATH, basePosition=start_pos, baseOrientation=start_orn,
            physicsClientId=self.client_id,
        )
        # Gravity would otherwise make this fall; teleop control fully owns velocity.
        p.changeDynamics(self.body_id, -1, linearDamping=0.0, angularDamping=0.0,
                          physicsClientId=self.client_id)

    def set_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """
        Args:
            vx, vy: body-frame forward/lateral speed in [-1, 1], scaled by max_linear_speed.
            vz: body-frame vertical speed in [-1, 1], scaled by max_vertical_speed.
            yaw_rate: normalized yaw rate in [-1, 1], scaled by max_yaw_rate.
        """
        vx = max(-1.0, min(1.0, vx)) * self.max_linear_speed
        vy = max(-1.0, min(1.0, vy)) * self.max_linear_speed
        vz = max(-1.0, min(1.0, vz)) * self.max_vertical_speed
        yaw_rate = max(-1.0, min(1.0, yaw_rate)) * self.max_yaw_rate

        pos, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)

        # Force level orientation (yaw only, zero roll/pitch) every step. This
        # kinematic drone only ever *commands* yaw -- but a wall collision can
        # still tip it via physics response, and since we only ever zero the
        # angular *velocity* below (stopping further tilting) and never correct
        # the orientation itself, an untreated tilt from one bad collision
        # would otherwise persist for the rest of the flight (confirmed in
        # practice: the drone tilted after a hard collision and never
        # recovered). Re-leveling every frame is a no-op when already level,
        # and snaps back immediately when a collision tips it.
        _, _, yaw = p.getEulerFromQuaternion(orn)
        level_orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.body_id, pos, level_orn,
                                           physicsClientId=self.client_id)

        rot = np.array(p.getMatrixFromQuaternion(level_orn)).reshape(3, 3)
        world_linear = rot @ np.array([vx, vy, vz])

        p.resetBaseVelocity(
            self.body_id,
            linearVelocity=world_linear.tolist(),
            angularVelocity=[0, 0, yaw_rate],
            physicsClientId=self.client_id,
        )

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (position_xyz, quaternion_xyzw, euler_rpy) ground truth."""
        pos, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        euler = p.getEulerFromQuaternion(orn)
        return np.array(pos), np.array(orn), np.array(euler)

    def _camera_extrinsics(self, cam_config: dict):
        """Shared math for get_camera_image / get_camera_pose: camera position + world-frame
        forward/up unit vectors, derived from the current drone pose and mount config."""
        mount_offset = cam_config.get("mount_offset", [0.11, 0.0, 0.0])
        tilt = math.radians(cam_config.get("tilt_deg", 0.0))

        pos, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        cam_pos = np.array(pos) + rot @ np.array(mount_offset)
        forward_body = np.array([math.cos(tilt), 0, -math.sin(tilt)])
        up_body = np.array([math.sin(tilt), 0, math.cos(tilt)])
        forward_world = rot @ forward_body
        up_world = rot @ up_body
        return cam_pos, forward_world, up_world

    def get_camera_pose(self, cam_config: dict) -> np.ndarray:
        """
        Returns the 4x4 world_T_cam pose of the onboard camera, in OpenCV camera-frame
        convention (x-right, y-down, z-forward) -- the same convention used when
        back-projecting depth pixels in slam/visual_odometry.py. Used to seed VO's
        trajectory in the same frame as ground truth (a one-time bootstrap; VO does
        not otherwise consult ground truth).
        """
        cam_pos, forward_world, up_world = self._camera_extrinsics(cam_config)
        down_world = -up_world
        right_world = np.cross(forward_world, up_world)

        T = np.eye(4)
        T[:3, 0] = right_world
        T[:3, 1] = down_world
        T[:3, 2] = forward_world
        T[:3, 3] = cam_pos
        return T

    def get_camera_image(self, cam_config: dict):
        """
        Renders the onboard forward-facing camera.

        Returns:
            rgb: (H, W, 3) uint8
            depth: (H, W) float32, meters (linearized from the OpenGL z-buffer)
        """
        width = cam_config.get("width", 320)
        height = cam_config.get("height", 240)
        fov = cam_config.get("fov_deg", 90.0)
        near = cam_config.get("near", 0.05)
        far = cam_config.get("far", 20.0)

        cam_pos, forward_world, up_world = self._camera_extrinsics(cam_config)
        target = cam_pos + forward_world

        view_matrix = p.computeViewMatrix(cam_pos.tolist(), target.tolist(), up_world.tolist())
        proj_matrix = p.computeProjectionMatrixFOV(fov, width / height, near, far)

        img = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id,
        )
        rgb = np.reshape(img[2], (height, width, 4))[:, :, :3].astype(np.uint8)
        depth_buffer = np.reshape(img[3], (height, width))

        # OpenGL depth buffer -> linear metric depth.
        depth = far * near / (far - (far - near) * depth_buffer)

        return rgb, depth.astype(np.float32)
