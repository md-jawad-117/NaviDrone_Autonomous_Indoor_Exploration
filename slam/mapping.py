"""
Accumulates a 3D point cloud map from RGB-D frames + the poses estimated by
VisualOdometry (slam/visual_odometry.py). Points are backprojected in the
camera's OpenCV frame (x-right, y-down, z-forward) and transformed to world
frame using each frame's cam-to-world pose -- the same convention VO uses
internally, so no axis conversion is needed here.
"""
import numpy as np
import open3d as o3d


class PointCloudMapper:
    def __init__(self, K: np.ndarray, config: dict):
        """
        Args:
            K: 3x3 camera intrinsics (see perception/camera_model.py).
            config: parsed config/slam_config.yaml under the 'mapping' key.
        """
        self.K = K
        self.cfg = config

        self.map_pcd = o3d.geometry.PointCloud()
        self._buffer_points = []
        self._buffer_colors = []
        self._frames_since_flush = 0

    def frame_to_world_points(self, rgb: np.ndarray, depth: np.ndarray, pose_cam_to_world: np.ndarray):
        """
        Backprojects one RGB-D frame to world-frame points, without touching the
        accumulated map. Exposed separately from integrate_frame so callers that
        need per-frame world points for something else (e.g. an occupancy grid)
        don't have to prematurely flush/duplicate the accumulated point buffer.

        Returns (points_world (N,3), colors (N,3) in [0,1]) -- both empty if no
        valid depth in this frame.
        """
        stride = self.cfg["pixel_stride"]
        min_d = self.cfg["min_depth"]
        max_d = self.cfg["max_depth"]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        h, w = depth.shape
        us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
        us = us.ravel()
        vs = vs.ravel()
        zs = depth[vs, us]

        valid = np.isfinite(zs) & (zs >= min_d) & (zs <= max_d)
        us, vs, zs = us[valid], vs[valid], zs[valid]
        if len(zs) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        xs = (us - cx) * zs / fx
        ys = (vs - cy) * zs / fy
        pts_cam = np.stack([xs, ys, zs], axis=1)

        R = pose_cam_to_world[:3, :3]
        t = pose_cam_to_world[:3, 3]
        pts_world = pts_cam @ R.T + t

        colors = rgb[vs, us].astype(np.float64) / 255.0
        return pts_world, colors

    def integrate_frame(self, rgb: np.ndarray, depth: np.ndarray, pose_cam_to_world: np.ndarray):
        """
        Args:
            rgb: (H, W, 3) uint8, from DroneRobot.get_camera_image.
            depth: (H, W) float32 meters, from DroneRobot.get_camera_image.
            pose_cam_to_world: 4x4, e.g. VOResult.pose from VisualOdometry.
        """
        pts_world, colors = self.frame_to_world_points(rgb, depth, pose_cam_to_world)
        if len(pts_world) == 0:
            return

        self._buffer_points.append(pts_world)
        self._buffer_colors.append(colors)
        self._frames_since_flush += 1

        if self._frames_since_flush >= self.cfg["flush_every_n_frames"]:
            self._flush()

    def _flush(self):
        if not self._buffer_points:
            return
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(np.concatenate(self._buffer_points, axis=0))
        new_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(self._buffer_colors, axis=0))

        self.map_pcd += new_pcd
        self.map_pcd = self.map_pcd.voxel_down_sample(self.cfg["voxel_size"])

        self._buffer_points = []
        self._buffer_colors = []
        self._frames_since_flush = 0

    def get_map(self) -> o3d.geometry.PointCloud:
        """Flushes any pending frames and returns the current accumulated map."""
        self._flush()
        return self.map_pcd

    def save(self, path: str):
        o3d.io.write_point_cloud(path, self.get_map())


def show_point_cloud(pcd: o3d.geometry.PointCloud, point_size: float = 3.5,
                      window_name: str = "Point Cloud"):
    """Opens an Open3D viewer with a smaller/finer point size than the default
    (draw_geometries' default point_size tends to look chunky for a dense map)."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()
