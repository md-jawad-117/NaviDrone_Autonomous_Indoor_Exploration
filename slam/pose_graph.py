"""
Pose-graph optimization: corrects VO's drifting keyframe trajectory using
loop-closure constraints. No g2o/GTSAM dependency (neither is installed and
both are painful to build on Windows) -- this is a small, from-scratch
Levenberg-Marquardt optimizer over SE(3) node poses via scipy, which is more
than adequate for a room-sized graph with a few dozen keyframes.

Each pose is optimized as a 6-vector [rotvec(3), translation(3)] (decoupled
parameterization, not a true Lie-algebra exponential map -- simpler, and fine
for LM with reasonable initial guesses from VO). Node 0 is held fixed as the
world-frame anchor.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class PoseGraphEdge:
    i: int
    j: int
    relative_pose: np.ndarray   # 4x4; want inv(T_i) @ T_j ~= relative_pose
    weight: float


class PoseGraph:
    def __init__(self, config: dict):
        """config: parsed config/slam_config.yaml under the 'pose_graph' key."""
        self.cfg = config
        self.nodes: List[np.ndarray] = []       # 4x4 poses
        self.edges: List[PoseGraphEdge] = []

    def add_node(self, pose: np.ndarray) -> int:
        self.nodes.append(pose.copy())
        return len(self.nodes) - 1

    def add_sequential_edge(self, i: int, j: int, relative_pose: np.ndarray):
        self.edges.append(PoseGraphEdge(i, j, relative_pose, self.cfg["sequential_weight"]))

    def add_loop_edge(self, i: int, j: int, relative_pose: np.ndarray):
        self.edges.append(PoseGraphEdge(i, j, relative_pose, self.cfg["loop_weight"]))

    def _pack(self) -> np.ndarray:
        """Nodes 1..N-1 as a flat [rotvec,trans]*... vector; node 0 is fixed."""
        params = []
        for T in self.nodes[1:]:
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            params.append(rvec.ravel())
            params.append(T[:3, 3])
        return np.concatenate(params) if params else np.array([])

    def _unpack(self, x: np.ndarray) -> List[np.ndarray]:
        poses = [self.nodes[0]]
        for k in range(1, len(self.nodes)):
            offset = (k - 1) * 6
            rvec = x[offset:offset + 3]
            t = x[offset + 3:offset + 6]
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(T)
        return poses

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        poses = self._unpack(x)
        res = np.zeros(len(self.edges) * 6)
        for k, edge in enumerate(self.edges):
            T_i, T_j = poses[edge.i], poses[edge.j]
            predicted = np.linalg.inv(T_i) @ T_j
            error = np.linalg.inv(edge.relative_pose) @ predicted
            rvec_err, _ = cv2.Rodrigues(error[:3, :3])
            res[k * 6:k * 6 + 3] = rvec_err.ravel() * edge.weight
            res[k * 6 + 3:k * 6 + 6] = error[:3, 3] * edge.weight
        return res

    def optimize(self, max_iterations: int = 100) -> List[np.ndarray]:
        """Runs the optimization and updates self.nodes in place. Returns the
        optimized node poses (also available afterward as self.nodes).

        Uses a robust ('huber') loss rather than plain least-squares: a handful
        of marginal loop-closure edges (low inlier count, imprecise geometry)
        would otherwise get weighted the same as solid ones and can visibly
        warp unrelated parts of the trajectory to accommodate them. Huber loss
        down-weights residuals past f_scale instead of letting them dominate.
        Requires method='trf' -- scipy's 'lm' only supports the plain loss.
        """
        if len(self.nodes) < 2 or not self.edges:
            return self.nodes

        x0 = self._pack()
        result = least_squares(
            self._residuals, x0, method="trf", loss="huber", f_scale=self.cfg["f_scale"],
            max_nfev=max_iterations * max(len(x0), 1),
        )
        self.nodes = self._unpack(result.x)
        return self.nodes
