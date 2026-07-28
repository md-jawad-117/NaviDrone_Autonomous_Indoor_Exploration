"""
Grid-based A* path planning over an OccupancyGrid's traversable mask. Replaces
the manual "raycast every candidate leg by hand" approach used earlier: given
any two points, this finds a route around obstacles and through doorways
automatically, then simplifies the raw grid path down to the fewest straight
segments (so a simple straight-line-pursuit controller, like
control/scripted_flight.py's WaypointController, can follow it).
"""
import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

from slam.occupancy_grid import OccupancyGrid

_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class AStarPlanner:
    def __init__(self, occupancy_grid: OccupancyGrid, inflate_radius_m: float = 0.3):
        """
        Args:
            occupancy_grid: an OccupancyGrid, ideally built via
                build_from_known_geometry (ground-truth map).
            inflate_radius_m: obstacles are grown by this much so planned paths
                keep a safety margin from walls/furniture.
        """
        self.grid = occupancy_grid
        self.free = occupancy_grid.traversable_mask(inflate_radius_m)
        self.height, self.width = self.free.shape

    def _in_bounds(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def _snap_to_free(self, cell: Tuple[int, int], max_radius: int = 40) -> Optional[Tuple[int, int]]:
        """BFS outward from `cell` to the nearest free cell, in case the exact
        requested point lands inside inflated obstacle margin."""
        if self._in_bounds(cell) and self.free[cell]:
            return cell
        visited = {cell}
        frontier = [cell]
        for _ in range(max_radius):
            next_frontier = []
            for r, c in frontier:
                for dr, dc in _NEIGHBORS_8:
                    nb = (r + dr, c + dc)
                    if nb in visited or not self._in_bounds(nb):
                        continue
                    visited.add(nb)
                    if self.free[nb]:
                        return nb
                    next_frontier.append(nb)
            frontier = next_frontier
            if not frontier:
                break
        return None

    def _has_line_of_sight(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Bresenham line check: True if every cell on the straight line a->b is free."""
        r0, c0 = a
        r1, c1 = b
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r1 >= r0 else -1
        sc = 1 if c1 >= c0 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            if not self.free[r, c]:
                return False
            if (r, c) == (r1, c1):
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_heap = [(heuristic(start, goal), start)]
        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            visited.add(current)

            for dr, dc in _NEIGHBORS_8:
                nb = (current[0] + dr, current[1] + dc)
                if not self._in_bounds(nb) or not self.free[nb] or nb in visited:
                    continue
                if dr != 0 and dc != 0:
                    # Disallow cutting diagonally between two blocked orthogonal cells.
                    if not self.free[current[0] + dr, current[1]] or not self.free[current[0], current[1] + dc]:
                        continue
                step_cost = math.hypot(dr, dc)
                tentative = g_score[current] + step_cost
                if tentative < g_score.get(nb, math.inf):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    heapq.heappush(open_heap, (tentative + heuristic(nb, goal), nb))

        return None

    def _simplify(self, path_cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Greedy string-pulling: from each point, jump to the farthest point
        with a clear straight line, instead of keeping every grid step."""
        if len(path_cells) <= 2:
            return path_cells
        simplified = [path_cells[0]]
        i = 0
        while i < len(path_cells) - 1:
            j = len(path_cells) - 1
            while j > i + 1 and not self._has_line_of_sight(path_cells[i], path_cells[j]):
                j -= 1
            simplified.append(path_cells[j])
            i = j
        return simplified

    def plan(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float],
             simplify: bool = True) -> Optional[List[Tuple[float, float]]]:
        """Returns a list of (x, y) world waypoints from start_xy to goal_xy,
        or None if no path exists."""
        start_cell = self._snap_to_free(self.grid.world_to_cell_xy(*start_xy))
        goal_cell = self._snap_to_free(self.grid.world_to_cell_xy(*goal_xy))
        if start_cell is None or goal_cell is None:
            return None

        path_cells = self._astar(start_cell, goal_cell)
        if path_cells is None:
            return None

        if simplify:
            path_cells = self._simplify(path_cells)

        return [self.grid.cell_to_world_xy(r, c) for r, c in path_cells]
