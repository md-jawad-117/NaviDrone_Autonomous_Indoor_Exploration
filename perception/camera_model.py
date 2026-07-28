"""Derives pinhole camera intrinsics from the PyBullet FOV projection used by DroneRobot."""
import math

import numpy as np


def get_intrinsics(cam_config: dict) -> np.ndarray:
    """
    Returns the 3x3 intrinsics matrix K for the onboard camera.

    PyBullet's computeProjectionMatrixFOV takes a *vertical* FOV; since the same
    aspect ratio is used for both the projection matrix and the pixel grid here,
    fx == fy (square pixels).
    """
    width = cam_config["width"]
    height = cam_config["height"]
    vfov_rad = math.radians(cam_config["fov_deg"])

    fy = height / (2.0 * math.tan(vfov_rad / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0

    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def backproject_points(points_2d: np.ndarray, depth: np.ndarray, K: np.ndarray,
                        min_depth: float, max_depth: float, patch_radius: int = 2):
    """Back-projects 2D pixel coords to 3D camera-frame points (OpenCV convention:
    x-right, y-down, z-forward) using a depth image. Shared by VisualOdometry
    (frame-to-frame) and LoopClosureDetector (non-sequential keyframe matches).

    Uses the median depth over a small (2*patch_radius+1) window around each
    point rather than a single pixel: ORB corners disproportionately land on
    object silhouettes/edges, exactly where a lone depth sample is most likely
    to be a near/far edge-mixing artifact. The median over a small neighborhood
    is robust to a minority of such contaminated pixels.

    Returns (points_3d, valid_mask); invalid entries in points_3d are zero.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pts3d = np.zeros((len(points_2d), 3), dtype=np.float64)
    valid = np.zeros(len(points_2d), dtype=bool)

    h, w = depth.shape
    for i, (u, v) in enumerate(points_2d):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        y0, y1 = max(0, vi - patch_radius), min(h, vi + patch_radius + 1)
        x0, x1 = max(0, ui - patch_radius), min(w, ui + patch_radius + 1)
        patch = depth[y0:y1, x0:x1]
        valid_patch = patch[np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)]
        if len(valid_patch) < 3:
            continue
        z = np.median(valid_patch)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts3d[i] = [x, y, z]
        valid[i] = True

    return pts3d, valid
