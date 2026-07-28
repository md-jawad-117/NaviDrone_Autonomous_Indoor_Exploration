"""
Automatic room-goal detection: finds "roomy" open-floor points on an
OccupancyGrid without any hand-picked coordinates. Uses a distance transform
(every free cell's distance to the nearest wall/obstacle) plus iterative
farthest-point suppression -- pick the most "interior" point remaining, clear
a radius around it, repeat. Simpler than full watershed room segmentation,
but good enough for picking one visit-worthy point per room.
"""
from typing import List, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from slam.occupancy_grid import OccupancyGrid


def _border_mask(shape, margin_cells: int) -> np.ndarray:
    """True for cells within margin_cells of the grid's outer edge."""
    mask = np.zeros(shape, dtype=bool)
    if margin_cells <= 0:
        return mask
    mask[:margin_cells, :] = True
    mask[-margin_cells:, :] = True
    mask[:, :margin_cells] = True
    mask[:, -margin_cells:] = True
    return mask


def _exclude_border_touching_components(free: np.ndarray) -> np.ndarray:
    """Drops every connected free-space component that touches the grid's
    outer edge at all, not just a fixed-width ring near it.

    By construction (OccupancyGrid is built with a safety margin past the
    house's actual footprint), real interior room space should never reach
    that padded boundary. If it does, it means the exterior wall raycast
    didn't fully seal the building off from its own margin somewhere (missed
    geometry, an odd door/window gap) -- and for an irregular floor plan,
    the resulting leaked "outside" region can extend deep into the bounding
    box (e.g. a real notch/cutout in the building's silhouette), too deep for
    a fixed-width margin exclusion to catch (confirmed: on one house, a
    corner notch connected all the way through, and excluding just a 0.5m
    border ring still left the pick sitting right at the edge of it). Since
    any such leak is, by definition, connected to the border, this only
    needs to check which whole components touch it -- no width parameter,
    works for any leak shape/depth.
    """
    labeled, num = label(free, structure=np.ones((3, 3)))
    if num == 0:
        return free
    border_labels = (set(np.unique(labeled[0, :])) | set(np.unique(labeled[-1, :]))
                     | set(np.unique(labeled[:, 0])) | set(np.unique(labeled[:, -1])))
    border_labels.discard(0)
    if not border_labels:
        return free
    return free & ~np.isin(labeled, list(border_labels))


def find_room_goals(grid: OccupancyGrid, start: Tuple[float, float],
                     inflate_radius_m: float = 0.15,
                     min_room_radius_m: float = 0.5,
                     suppression_radius_m: float = 1.2,
                     exclude_margin_m: float = 0.0) -> List[Tuple[float, float]]:
    """
    Returns a list of (x, y) world points, one per detected "roomy" area.

    Args:
        grid: an OccupancyGrid with build_from_known_geometry already called.
        start: the drone's starting (x, y) -- room search is restricted to
            whichever connected free-space region contains this point. This
            also throws out anything outside the building's exterior walls:
            the occupancy grid is built with a small margin past the house
            footprint for safety, and that outside area has nothing to block
            a raycast, so it registers as "free" too -- without this restriction,
            those wide-open outside cells win the "roomiest" search every time.
        inflate_radius_m: safety margin for the underlying traversable mask
            (same meaning as AStarPlanner's inflate_radius_m).
        min_room_radius_m: a candidate point must be at least this far from
            any wall/obstacle to count as a "room" (not just a corridor) --
            stops the search once no area is roomy enough left.
        suppression_radius_m: after picking a room center, clear a disk of
            this radius around it so the next pick lands in a different room.
        exclude_margin_m: never pick a point within this distance of the
            grid's outer edge, regardless of how "free" it looks. The
            same-component restriction above assumes the exterior wall
            raycast fully seals the building off from its own safety
            margin -- but a wall the raycast doesn't fully catch (missed
            geometry, an odd door/window gap) lets the two merge into one
            connected region, and the wide-open margin then wins the
            "roomiest" search every time (confirmed: on one house, every
            goal landed exactly at a grid corner). This is a hard backstop
            independent of that connectivity assumption.
    """
    free = grid.traversable_mask(inflate_radius_m)
    free = _exclude_border_touching_components(free)

    labeled, _ = label(free, structure=np.ones((3, 3)))
    start_label = labeled[grid.world_to_cell_xy(*start)]
    free = free & (labeled == start_label)

    dist_cells = distance_transform_edt(free)

    min_room_radius_cells = min_room_radius_m / grid.resolution
    suppression_radius_cells = suppression_radius_m / grid.resolution
    margin_cells = int(round(exclude_margin_m / grid.resolution))

    working = dist_cells.copy()
    working[_border_mask(working.shape, margin_cells)] = -1
    goals = []
    rows, cols = np.ogrid[:working.shape[0], :working.shape[1]]

    while True:
        idx = np.unravel_index(np.argmax(working), working.shape)
        if working[idx] < min_room_radius_cells:
            break
        row, col = idx
        goals.append(grid.cell_to_world_xy(row, col))

        disk = (rows - row) ** 2 + (cols - col) ** 2 <= suppression_radius_cells ** 2
        working[disk] = -1

    return goals


def find_start_position(grid: OccupancyGrid, inflate_radius_m: float = 0.15,
                         exclude_margin_m: float = 0.0) -> Tuple[float, float]:
    """
    Picks a safe default drone starting position with no prior knowledge of
    the house layout: the point deepest inside the largest connected free-space
    region (most clearance from any wall/obstacle). Restricting to the largest
    connected region -- rather than just the single global farthest-from-a-wall
    point -- avoids landing in a small disconnected pocket or the thin margin
    outside the building's exterior walls (see find_room_goals for why that
    margin registers as "free" too).

    exclude_margin_m: see find_room_goals -- same hard backstop against
    landing in the grid's outer safety margin if wall detection leaks.

    Raises RuntimeError if the grid has no free space at all (e.g. the house
    mesh failed to load, or inflate_radius_m is unreasonably large for it).
    """
    free = grid.traversable_mask(inflate_radius_m)
    free = _exclude_border_touching_components(free)
    labeled, num_components = label(free, structure=np.ones((3, 3)))
    if num_components == 0:
        raise RuntimeError("No free space found in the occupancy grid -- check the house "
                            "mesh loaded correctly, the exterior walls fully seal it off from "
                            "its safety margin, and inflate_radius_m isn't too large.")

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background/occupied label, not a real component
    largest_label = int(np.argmax(sizes))

    dist_cells = distance_transform_edt(free & (labeled == largest_label))
    margin_cells = int(round(exclude_margin_m / grid.resolution))
    dist_cells[_border_mask(dist_cells.shape, margin_cells)] = -1
    row, col = np.unravel_index(np.argmax(dist_cells), dist_cells.shape)
    return grid.cell_to_world_xy(row, col)


def order_goals_nearest_neighbor(start: Tuple[float, float],
                                  goals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Greedy nearest-neighbor visiting order -- good enough for a handful of
    room goals; not a true shortest-tour (TSP) solve."""
    remaining = list(goals)
    ordered = []
    current = start
    while remaining:
        nearest = min(remaining, key=lambda g: (g[0] - current[0]) ** 2 + (g[1] - current[1]) ** 2)
        ordered.append(nearest)
        remaining.remove(nearest)
        current = nearest
    return ordered
