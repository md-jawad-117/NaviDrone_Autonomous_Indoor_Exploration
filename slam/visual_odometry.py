"""
Feature-based visual odometry front-end.

Frame-to-frame tracking: ORB features are matched between consecutive frames,
matched keypoints from the *previous* frame are back-projected to 3D using its
depth image, and solvePnPRansac recovers the relative camera motion. Poses are
chained to build a running (drifting) world-frame trajectory estimate.

Keyframes are recorded separately (on translation/rotation thresholds) for the
mapping and pose-graph modules built in later stages.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from perception.camera_model import backproject_points


@dataclass
class Keyframe:
    pose: np.ndarray            # 4x4, cam-to-world
    rgb: np.ndarray
    depth: np.ndarray
    keypoints: list
    descriptors: np.ndarray


@dataclass
class VOResult:
    pose: np.ndarray             # 4x4, cam-to-world, current frame estimate
    num_matches: int
    num_inliers: int
    tracking_ok: bool
    is_keyframe: bool


def _pose_to_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def _scale_pose(T: np.ndarray, factor: float) -> np.ndarray:
    """Scales a relative pose's translation and rotation angle by `factor`
    (0 -> identity/no motion, 1 -> T unchanged). Used to decay the
    constant-velocity fallback toward "assume stationary" the longer
    tracking stays lost."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    R_scaled, _ = cv2.Rodrigues(rvec * factor)
    return _pose_to_4x4(R_scaled, T[:3, 3] * factor)


class VisualOdometry:
    def __init__(self, K: np.ndarray, config: dict, initial_pose: Optional[np.ndarray] = None):
        """
        Args:
            K: 3x3 camera intrinsics (see perception/camera_model.py).
            config: parsed config/slam_config.yaml (both 'vo' and 'keyframe' keys).
            initial_pose: 4x4 cam-to-world pose to seed the trajectory with
                (use the drone's ground-truth start pose so trajectories are
                comparable in the same world frame). Defaults to identity.
        """
        self.K = K
        self.vo_cfg = config["vo"]
        self.kf_cfg = config["keyframe"]

        self.orb = cv2.ORB_create(
            nfeatures=self.vo_cfg["orb_features"],
            fastThreshold=self.vo_cfg["fast_threshold"],
            edgeThreshold=self.vo_cfg["edge_threshold"],
            nlevels=self.vo_cfg["nlevels"],
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.current_pose = initial_pose.copy() if initial_pose is not None else np.eye(4)
        self.trajectory: List[np.ndarray] = [self.current_pose.copy()]
        self.keyframes: List[Keyframe] = []

        self._prev_gray = None
        self._prev_kp = None
        self._prev_des = None
        self._prev_depth = None
        self._last_good_relative = np.eye(4)  # constant-velocity fallback base
        self._consecutive_fallbacks = 0

        self._frames_processed = 0

    def _backproject(self, points_2d: np.ndarray, depth: np.ndarray):
        """Back-projects 2D pixel coords to 3D camera-frame points using a depth image."""
        return backproject_points(
            points_2d, depth, self.K,
            self.vo_cfg["min_valid_depth"], self.vo_cfg["max_valid_depth"],
        )

    def _should_add_keyframe(self, current_pose: np.ndarray) -> bool:
        if not self.keyframes:
            return True
        since_last_kf = np.linalg.inv(self.keyframes[-1].pose) @ current_pose
        t_norm = np.linalg.norm(since_last_kf[:3, 3])
        rot_deg = _rotation_angle_deg(since_last_kf[:3, :3])
        return (t_norm >= self.kf_cfg["translation_thresh"] or
                rot_deg >= self.kf_cfg["rotation_thresh_deg"])

    def process_frame(self, rgb: np.ndarray, depth: np.ndarray) -> VOResult:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        self._frames_processed += 1

        if self._prev_gray is None or self._prev_des is None or des is None:
            self._store_prev(gray, kp, des, depth)
            self.keyframes.append(Keyframe(self.current_pose.copy(), rgb, depth, kp, des))
            return VOResult(self.current_pose.copy(), 0, 0, True, True)

        num_matches, num_inliers, tracking_ok = 0, 0, False
        candidate_pose = None

        matches = self.matcher.knnMatch(self._prev_des, des, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.vo_cfg["ratio_test_threshold"] * n.distance:
                good.append(m)
        num_matches = len(good)

        if num_matches >= self.vo_cfg["min_matches"]:
            prev_pts = np.array([self._prev_kp[m.queryIdx].pt for m in good])
            cur_pts = np.array([kp[m.trainIdx].pt for m in good])

            obj_pts, valid = self._backproject(prev_pts, self._prev_depth)
            obj_pts = obj_pts[valid]
            img_pts = cur_pts[valid]

            if len(obj_pts) >= self.vo_cfg["min_pnp_points"]:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts.astype(np.float64), img_pts.astype(np.float64), self.K, None,
                    reprojectionError=self.vo_cfg["ransac_reproj_threshold"],
                    confidence=self.vo_cfg["ransac_confidence"],
                    iterationsCount=self.vo_cfg["ransac_iterations"],
                )
                if ok and inliers is not None and len(inliers) >= self.vo_cfg["min_pnp_points"]:
                    R, _ = cv2.Rodrigues(rvec)
                    candidate_pose = _pose_to_4x4(R, tvec)
                    num_inliers = len(inliers)

                    t_norm = np.linalg.norm(candidate_pose[:3, 3])
                    rot_deg = _rotation_angle_deg(candidate_pose[:3, :3])
                    if (t_norm <= self.vo_cfg["max_translation_per_frame"] and
                            rot_deg <= self.vo_cfg["max_rotation_deg_per_frame"]):
                        tracking_ok = True
                    else:
                        print(f"[VO] frame {self._frames_processed}: rejected implausible "
                              f"PnP solve (t={t_norm:.2f}m, rot={rot_deg:.1f}deg)")

        if tracking_ok:
            relative_pose = candidate_pose
            self._last_good_relative = candidate_pose
            self._consecutive_fallbacks = 0
        else:
            self._consecutive_fallbacks += 1
            frames_past_hold = max(0, self._consecutive_fallbacks - self.vo_cfg["hold_frames_before_decay"])
            decay = self.vo_cfg["fallback_decay"] ** frames_past_hold
            relative_pose = _scale_pose(self._last_good_relative, decay)
            print(f"[VO] frame {self._frames_processed}: weak tracking "
                  f"({num_matches} matches) - constant-velocity fallback "
                  f"(decay={decay:.2f}, {self._consecutive_fallbacks} consecutive)")

        # p_world = world_T_prev @ relative_pose^-1 @ p_cur_cam  (see module derivation)
        self.current_pose = self.current_pose @ np.linalg.inv(relative_pose)
        self.trajectory.append(self.current_pose.copy())

        is_keyframe = self._should_add_keyframe(self.current_pose)
        if is_keyframe:
            self.keyframes.append(Keyframe(self.current_pose.copy(), rgb, depth, kp, des))

        self._store_prev(gray, kp, des, depth)

        return VOResult(self.current_pose.copy(), num_matches, num_inliers,
                         tracking_ok, is_keyframe)

    def _store_prev(self, gray, kp, des, depth):
        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_des = des
        self._prev_depth = depth
