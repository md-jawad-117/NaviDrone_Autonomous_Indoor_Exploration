"""
Loop-closure detection: matches a keyframe against *non-adjacent* past keyframes
(unlike VisualOdometry, which only ever compares consecutive frames) to find
places the drone has revisited. Uses the same ORB + depth-based PnP approach as
VisualOdometry, just applied to arbitrary keyframe pairs instead of adjacent
frames, and with stricter thresholds since a wrong loop edge can badly corrupt
the pose-graph optimization.
"""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from perception.camera_model import backproject_points
from slam.visual_odometry import Keyframe, _pose_to_4x4, _rotation_angle_deg


@dataclass
class LoopClosure:
    from_idx: int                # index of the earlier (matched-against) keyframe
    to_idx: int                  # index of the later (new) keyframe
    relative_pose: np.ndarray    # 4x4, cam-to-cam: p_to = relative_pose @ p_from
    num_inliers: int


class LoopClosureDetector:
    def __init__(self, K: np.ndarray, config: dict):
        """
        Args:
            K: 3x3 camera intrinsics.
            config: parsed config/slam_config.yaml under the 'loop_closure' key.
        """
        self.K = K
        self.cfg = config
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def find_closures(self, keyframes: List[Keyframe], new_idx: int) -> List[LoopClosure]:
        """Tries to match keyframes[new_idx] against every sufficiently-old earlier
        keyframe. Returns only the single best (highest-inlier) match, if any:
        keeping every match that clears the threshold tends to add a handful of
        marginal, mutually-inconsistent edges that can badly warp unrelated
        parts of the graph even under a robust loss (verified: with all matches
        kept, one bad edge among several distorted a keyframe by ~9.7m even
        though the trajectory endpoint improved 20x)."""
        new_kf = keyframes[new_idx]
        if new_kf.descriptors is None:
            return []

        best = None
        cutoff = new_idx - self.cfg["skip_recent_keyframes"]
        for i in range(max(0, cutoff)):
            candidate = keyframes[i]
            closure = self._try_match(candidate, new_kf, i, new_idx)
            if closure is None:
                continue
            if not self._agrees_with_vo(keyframes, closure):
                continue
            if best is None or closure.num_inliers > best.num_inliers:
                best = closure
        return [best] if best is not None else []

    def _agrees_with_vo(self, keyframes: List[Keyframe], closure: "LoopClosure") -> bool:
        """Rejects a geometrically-plausible-looking match that nonetheless implies
        the drone teleported somewhere VO never went near -- a strong sign the
        match is a coincidental false positive (e.g. two visually similar but
        physically distant walls), not a real revisit. VO drifts, but not by
        the tens of meters a bad match can imply, so a generous tolerance here
        still catches genuine loop closures while rejecting nonsense ones."""
        vo_relative = np.linalg.inv(keyframes[closure.from_idx].pose) @ keyframes[closure.to_idx].pose
        diff = np.linalg.inv(closure.relative_pose) @ vo_relative
        trans_diff = np.linalg.norm(diff[:3, 3])
        rot_diff = _rotation_angle_deg(diff[:3, :3])
        return (trans_diff <= self.cfg["max_vo_disagreement_m"] and
                rot_diff <= self.cfg["max_vo_disagreement_deg"])

    def _try_match(self, from_kf: Keyframe, to_kf: Keyframe,
                    from_idx: int, to_idx: int) -> LoopClosure:
        if from_kf.descriptors is None or to_kf.descriptors is None:
            return None

        matches = self.matcher.knnMatch(from_kf.descriptors, to_kf.descriptors, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.cfg["ratio_test_threshold"] * n.distance:
                good.append(m)
        if len(good) < self.cfg["min_matches"]:
            return None

        from_pts = np.array([from_kf.keypoints[m.queryIdx].pt for m in good])
        to_pts = np.array([to_kf.keypoints[m.trainIdx].pt for m in good])

        obj_pts, valid = backproject_points(
            from_pts, from_kf.depth, self.K,
            self.cfg["min_valid_depth"], self.cfg["max_valid_depth"],
        )
        obj_pts = obj_pts[valid]
        img_pts = to_pts[valid]
        if len(obj_pts) < self.cfg["min_pnp_points"]:
            return None

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts.astype(np.float64), img_pts.astype(np.float64), self.K, None,
            reprojectionError=self.cfg["ransac_reproj_threshold"],
            confidence=self.cfg["ransac_confidence"],
            iterationsCount=self.cfg["ransac_iterations"],
        )
        if not ok or inliers is None or len(inliers) < self.cfg["min_pnp_points"]:
            return None

        R, _ = cv2.Rodrigues(rvec)
        relative_pose = _pose_to_4x4(R, tvec)
        return LoopClosure(from_idx, to_idx, relative_pose, len(inliers))
