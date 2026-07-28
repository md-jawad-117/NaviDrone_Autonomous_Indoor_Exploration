"""
2D top-down occupancy grid: a much easier map representation to visually sanity
-check than a dense 3D point cloud ("does this look like the house floor plan?"
vs. squinting at a rotated blob of dots). Built from the same accumulated 3D
points as PointCloudMapper, just flattened onto the XY plane within a height
band that excludes floor/ceiling.
"""
from typing import Tuple

import numpy as np
import pybullet as p
from scipy.ndimage import binary_dilation


class OccupancyGrid:
    def __init__(self, config: dict):
        """config: parsed config/slam_config.yaml under the 'occupancy_grid' key."""
        self.cfg = config
        self.resolution = config["resolution"]
        self.x_min, self.x_max = config["x_min"], config["x_max"]
        self.y_min, self.y_max = config["y_min"], config["y_max"]

        self.width = int(np.ceil((self.x_max - self.x_min) / self.resolution))
        self.height = int(np.ceil((self.y_max - self.y_min) / self.resolution))
        self.hit_count = np.zeros((self.height, self.width), dtype=np.int32)
        self.visited = np.zeros((self.height, self.width), dtype=bool)
        self.known_occupied = None  # populated by build_from_known_geometry, if used

    def build_from_known_geometry(self, client_id: int, z_range=None):
        """
        Populates the grid directly from the world's actual collision geometry
        via raycasts, instead of from accumulated sensor points -- this is the
        "I already have the floor plan" case: one vertical ray per cell (from
        z_range[0] to z_range[1]) tells us definitively whether that cell is
        occupied, no flying required. Used for path planning, where we want the
        ground truth map, not a sensor-built approximation of it.

        z_range defaults to config['known_geometry_z_range'] -- deliberately a
        narrow band around the drone's actual cruise altitude, not the whole
        room height, since low furniture (e.g. a bed frame) the drone would
        fly straight over shouldn't count as an obstacle for planning.
        """
        if z_range is None:
            z_range = self.cfg["known_geometry_z_range"]
        xs = self.x_min + (np.arange(self.width) + 0.5) * self.resolution
        ys = self.y_min + (np.arange(self.height) + 0.5) * self.resolution
        grid_x, grid_y = np.meshgrid(xs, ys)  # both (height, width)
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()

        occupied = np.zeros(len(flat_x), dtype=bool)
        # A hair under MAX_RAY_INTERSECTION_BATCH_SIZE: pybullet sometimes returns one
        # fewer result than requested at the exact limit.
        batch_size = p.MAX_RAY_INTERSECTION_BATCH_SIZE - 1
        for start in range(0, len(flat_x), batch_size):
            end = min(start + batch_size, len(flat_x))
            froms = [(x, y, z_range[0]) for x, y in zip(flat_x[start:end], flat_y[start:end])]
            tos = [(x, y, z_range[1]) for x, y in zip(flat_x[start:end], flat_y[start:end])]
            results = p.rayTestBatch(froms, tos, physicsClientId=client_id)
            occupied[start:start + len(results)] = [r[0] != -1 for r in results]

        self.known_occupied = occupied.reshape(self.height, self.width)

    def traversable_mask(self, inflate_radius_m: float = 0.3) -> np.ndarray:
        """
        Returns a (H, W) boolean mask, True = safe to fly through, for path
        planning. Obstacles are inflated by inflate_radius_m (an approximate
        drone/safety radius) so planned paths don't hug walls. Uses the known-
        geometry grid if build_from_known_geometry was called, otherwise falls
        back to the sensor-accumulated occupied cells.
        """
        occupied = (self.known_occupied if self.known_occupied is not None
                    else self.hit_count >= self.cfg["occupied_point_threshold"])
        inflate_cells = max(1, int(round(inflate_radius_m / self.resolution)))
        structure = np.ones((2 * inflate_cells + 1, 2 * inflate_cells + 1), dtype=bool)
        inflated_occupied = binary_dilation(occupied, structure=structure)
        return ~inflated_occupied

    def _world_to_cell(self, xy: np.ndarray):
        col = ((xy[:, 0] - self.x_min) / self.resolution).astype(np.int64)
        row = ((xy[:, 1] - self.y_min) / self.resolution).astype(np.int64)
        valid = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        return row[valid], col[valid]

    def world_to_cell_xy(self, x: float, y: float) -> Tuple[int, int]:
        """Single-point version, returns (row, col); may fall outside grid bounds."""
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        return row, col

    def cell_to_world_xy(self, row: int, col: int) -> Tuple[float, float]:
        x = self.x_min + (col + 0.5) * self.resolution
        y = self.y_min + (row + 0.5) * self.resolution
        return x, y

    def integrate_points(self, points_world: np.ndarray):
        """points_world: (N, 3) array of accumulated map points, in world frame."""
        if len(points_world) == 0:
            return
        in_band = ((points_world[:, 2] >= self.cfg["min_height"]) &
                   (points_world[:, 2] <= self.cfg["max_height"]))
        pts = points_world[in_band]
        if len(pts) == 0:
            return
        rows, cols = self._world_to_cell(pts[:, :2])
        np.add.at(self.hit_count, (rows, cols), 1)

    def integrate_visited(self, xy: np.ndarray):
        """Marks the drone's own (x, y) position as visited/known-free space."""
        rows, cols = self._world_to_cell(np.atleast_2d(xy))
        self.visited[rows, cols] = True

    def to_image(self) -> np.ndarray:
        """
        Returns a (H, W) uint8 image: 127 = unknown, 255 = free (visited, no
        obstacle), 0 = occupied. Row 0 is the max-y edge (so it displays with
        the conventional top-down orientation via imshow/matplotlib).
        """
        img = np.full((self.height, self.width), 127, dtype=np.uint8)
        img[self.visited] = 255
        img[self.hit_count >= self.cfg["occupied_point_threshold"]] = 0
        return np.flipud(img)

    def world_extent(self):
        """(x_min, x_max, y_min, y_max) -- for matplotlib's imshow(extent=...)."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)
