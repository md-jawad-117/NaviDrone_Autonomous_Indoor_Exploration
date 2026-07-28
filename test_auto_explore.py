"""
Fully automatic house exploration: zero hand-picked room coordinates. Room
goals are found automatically (slam/room_detection.py) from the known house
geometry via a distance-transform "roomiest point" search, ordered with a
nearest-neighbor visiting order, then A*-planned (slam/path_planning.py) and
flown, with a 360 scan at each room and reactive collision-avoidance braking
(control/collision_avoidance.py) along the way. Point this at a different
house GLB (converted via scripts/convert_house_glb.py) and it should find and
visit that house's rooms with no changes needed.

Run:
    python test_auto_explore.py
"""
import os
import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import yaml

from env.house_world import HouseWorld
from env.drone_robot import DroneRobot
from control.scripted_flight import WaypointController
from control.collision_avoidance import apply_braking
from perception.camera_model import get_intrinsics
from slam.visual_odometry import VisualOdometry
from slam.mapping import PointCloudMapper, show_point_cloud
from slam.occupancy_grid import OccupancyGrid
from slam.path_planning import AStarPlanner
from slam.room_detection import find_room_goals, find_start_position, order_goals_nearest_neighbor

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")

CRUISE_Z = 1.0
GRID_BOUNDS_MARGIN = 0.5  # meters past the house's actual exterior walls

# Clip range for our own colorized depth view, tighter than the camera's
# actual near/far (0.05m/20m) -- room-scale distances (a few meters) would
# otherwise all bunch up at one end of the colormap and look nearly flat.
DEPTH_VIS_MIN_M = 0.1
DEPTH_VIS_MAX_M = 6.0


def load_config(name):
    with open(os.path.join(ROOT, "config", name), "r") as f:
        return yaml.safe_load(f)


def plan_full_route(planner: AStarPlanner, goals) -> tuple:
    """Chains A*-planned legs between consecutive goals into one waypoint list,
    dropping duplicate boundary points between legs. Also returns the set of
    indices in that list which are actual goal arrivals (as opposed to
    intermediate A*-path transit points) -- these are where a scan-in-place
    should happen, not every waypoint along the way."""
    full_route = [goals[0]]
    goal_indices = set()
    for i in range(len(goals) - 1):
        leg = planner.plan(goals[i], goals[i + 1])
        if leg is None:
            raise RuntimeError(f"No path found from {goals[i]} to {goals[i + 1]}")
        full_route.extend(leg[1:])
        goal_indices.add(len(full_route) - 1)
    return full_route, goal_indices


def main():
    env_config = load_config("house_config.yaml")
    slam_config = load_config("slam_config.yaml")
    flight_config = load_config("flight_controller.yaml")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    gui_mode = env_config["sim"].get("gui", True)
    client_id = p.connect(p.GUI if gui_mode else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    # The built-in depth preview panel shows the raw, non-linear OpenGL
    # z-buffer through a fixed blue-only shader -- not recolorable from here,
    # and low-sensitivity at room scale. Disable it in favor of our own
    # colorized view (below) built from the actual linear metric depth.
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.setGravity(0, 0, 0)
    p.setTimeStep(env_config["sim"]["timestep"])

    house = HouseWorld(env_config, client_id)
    cam_config = env_config["camera"]

    # Derive the occupancy grid's world bounds from the house's actual geometry
    # instead of a hardcoded per-house box -- makes this whole script portable
    # to a different house file with no manual re-measuring.
    x_min, x_max, y_min, y_max = house.get_bounds()
    grid_cfg = slam_config["occupancy_grid"]
    grid_cfg["x_min"] = x_min - GRID_BOUNDS_MARGIN
    grid_cfg["x_max"] = x_max + GRID_BOUNDS_MARGIN
    grid_cfg["y_min"] = y_min - GRID_BOUNDS_MARGIN
    grid_cfg["y_max"] = y_max + GRID_BOUNDS_MARGIN
    print(f"House bounds: x=[{x_min:.2f},{x_max:.2f}] y=[{y_min:.2f},{y_max:.2f}]")
    p.resetDebugVisualizerCamera(cameraDistance=max(x_max - x_min, y_max - y_min),
                                  cameraYaw=45, cameraPitch=-50,
                                  cameraTargetPosition=[(x_min + x_max) / 2, (y_min + y_max) / 2, 1])

    # Build the planning grid *before* spawning the drone: the drone's own body
    # would otherwise sit exactly at the start position and block that cell's
    # raycast, corrupting the "start is free" assumption find_room_goals and
    # find_start_position rely on (confirmed: with the drone already loaded,
    # room detection silently found zero rooms).
    print("Detecting rooms automatically from known house geometry...")
    grid_for_planning = OccupancyGrid(grid_cfg)
    grid_for_planning.build_from_known_geometry(client_id)

    room_cfg = slam_config["room_detection"]
    start_xy = find_start_position(grid_for_planning, inflate_radius_m=room_cfg["inflate_radius_m"],
                                    exclude_margin_m=GRID_BOUNDS_MARGIN)
    print(f"Auto-picked start position: {tuple(round(v, 2) for v in start_xy)}")
    env_config["drone"]["start_position"] = [start_xy[0], start_xy[1], CRUISE_Z]

    drone = DroneRobot(env_config["drone"], client_id)

    # Live-tunable sliders in the GUI's "Params" panel (otherwise empty) --
    # nudge cruise speed / braking clearance during a run without editing
    # yaml and restarting. GUI-only: readUserDebugParameter errors in DIRECT mode.
    if gui_mode:
        speed_param = p.addUserDebugParameter("Cruise speed (m/s)", 0.3, 3.0, drone.max_linear_speed)
        braking_dist_param = p.addUserDebugParameter("Braking distance (m)", 0.3, 2.5, 1.2)
        min_safe_dist_param = p.addUserDebugParameter("Min safe distance (m)", 0.1, 1.0, 0.45)

    room_goals = find_room_goals(
        grid_for_planning, start_xy,
        inflate_radius_m=room_cfg["inflate_radius_m"],
        min_room_radius_m=room_cfg["min_room_radius_m"],
        suppression_radius_m=room_cfg["suppression_radius_m"],
        exclude_margin_m=GRID_BOUNDS_MARGIN,
    )
    print(f"Found {len(room_goals)} room goals: "
          f"{[tuple(round(v, 2) for v in g) for g in room_goals]}")

    ordered_goals = order_goals_nearest_neighbor(start_xy, room_goals)
    full_goal_sequence = [start_xy] + ordered_goals + [start_xy]

    planner = AStarPlanner(grid_for_planning, inflate_radius_m=room_cfg["inflate_radius_m"])
    route_xy, scan_indices = plan_full_route(planner, full_goal_sequence)
    waypoints = [(x, y, CRUISE_Z) for x, y in route_xy]
    print(f"Planned {len(waypoints)} waypoints across {len(full_goal_sequence) - 1} legs "
          f"({len(scan_indices)} will pause for a 360 scan).\n")

    controller = WaypointController(waypoints, flight_config, scan_indices=scan_indices)

    K = get_intrinsics(cam_config)
    initial_pose = drone.get_camera_pose(cam_config)
    vo = VisualOdometry(K, slam_config, initial_pose=initial_pose)

    gt_mapper = PointCloudMapper(K, slam_config["mapping"])
    sensed_grid = OccupancyGrid(slam_config["occupancy_grid"])

    print("Flying auto-planned route (no manual control needed).")
    print("Close the PyBullet window or press ESC to stop early.\n")

    gt_trajectory = []
    step = 0
    camera_every_n_steps = 6
    max_steps = 50000

    # See config/house_config.yaml's sim.speed_multiplier comment: this only
    # paces wall-clock playback, it doesn't change simulated physics. A
    # multiplier <= 0 removes the sleep entirely (fastest possible, bounded
    # only by actual per-step compute -- the GUI will look sped way up).
    speed_multiplier = env_config["sim"].get("speed_multiplier", 1.0)
    step_sleep = (env_config["sim"]["timestep"] / speed_multiplier) if speed_multiplier > 0 else 0.0

    # Safety monitor: if the drone ever ends up somewhere physically impossible
    # (outside the house footprint, or well off its cruise altitude), snap it
    # back to the last known-good position instead of silently continuing --
    # seen once in practice (reportedly right as the drone tried to leave a
    # room through a tight doorway) with the drone ending up outside/above the
    # building; root cause wasn't reproducible headlessly, so this is a
    # recovery net rather than a fix for a specific diagnosed cause.
    grid_cfg = slam_config["occupancy_grid"]
    bounds_margin = 0.3
    x_bounds = (grid_cfg["x_min"] - bounds_margin, grid_cfg["x_max"] + bounds_margin)
    y_bounds = (grid_cfg["y_min"] - bounds_margin, grid_cfg["y_max"] + bounds_margin)
    z_bounds = (CRUISE_Z - 1.0, CRUISE_Z + 1.0)
    # Recover to a position from a bit further back, not just the immediately
    # preceding frame: if the drone was already wedged tight against something
    # right before the anomaly, the "last" position may itself be too close
    # to whatever caused it.
    good_pos_history = deque([list(drone.get_pose()[0])], maxlen=120)

    try:
        while step < max_steps:
            if not p.isConnected(client_id):
                print("\nPyBullet window was closed, stopping.")
                break

            pos, quat, euler = drone.get_pose()

            if not (x_bounds[0] <= pos[0] <= x_bounds[1] and
                    y_bounds[0] <= pos[1] <= y_bounds[1] and
                    z_bounds[0] <= pos[2] <= z_bounds[1]):
                recovery_pos = good_pos_history[0]  # furthest back = most clearance
                print(f"\nstep {step}: drone position {pos} is outside the house -- "
                      f"recovering to {recovery_pos} and abandoning the current waypoint.")
                p.resetBasePositionAndOrientation(drone.body_id, recovery_pos, quat, client_id)
                p.resetBaseVelocity(drone.body_id, [0, 0, 0], [0, 0, 0], client_id)
                pos = np.array(recovery_pos)
                # Don't just reset position and let the controller retry the
                # identical approach that caused this -- that repeated and even
                # compounded in practice. Move on to the next waypoint instead.
                controller.force_skip_current_waypoint()
                good_pos_history.clear()
                good_pos_history.append(recovery_pos)
            else:
                good_pos_history.append(list(pos))

            vx, vy, vz, yaw_rate, done = controller.get_command(pos, euler[2])
            if done:
                print(f"\nAuto-explore route complete at step {step} ({step / 120:.1f}s sim time).")
                break

            if gui_mode:
                drone.max_linear_speed = p.readUserDebugParameter(speed_param)
                braking_distance = p.readUserDebugParameter(braking_dist_param)
                min_safe_distance = p.readUserDebugParameter(min_safe_dist_param)
            else:
                braking_distance, min_safe_distance = 1.2, 0.45

            # Reactive braking: slow down before reaching an obstacle instead of
            # only reacting (via the escape maneuver) after clipping one.
            vx, vy = apply_braking(client_id, pos, quat, vx, vy,
                                    braking_distance=braking_distance,
                                    min_safe_distance=min_safe_distance)

            drone.set_velocity_command(vx, vy, vz, yaw_rate)
            # p.stepSimulation() redraws the full textured 3D scene every call
            # in GUI mode -- that redraw, not the wall-clock sleep above, is
            # the actual per-step cost at this house's texture/geometry
            # complexity. Only pay for it on steps where we're already
            # capturing a camera frame (physics still steps every frame,
            # only the *visual* redraw is skipped in between).
            if gui_mode:
                p.configureDebugVisualizer(
                    p.COV_ENABLE_RENDERING, 1 if step % camera_every_n_steps == 0 else 0)
            p.stepSimulation()

            if step % camera_every_n_steps == 0:
                rgb, depth = drone.get_camera_image(cam_config)

                gt_cam_pose = drone.get_camera_pose(cam_config)
                frame_points, _ = gt_mapper.frame_to_world_points(rgb, depth, gt_cam_pose)
                gt_mapper.integrate_frame(rgb, depth, gt_cam_pose)
                sensed_grid.integrate_points(frame_points)
                sensed_grid.integrate_visited(gt_cam_pose[:3, 3])
                gt_trajectory.append(gt_cam_pose[:3, 3].copy())

                vo.process_frame(rgb, depth)

                if gui_mode:
                    depth_clipped = np.clip(depth, DEPTH_VIS_MIN_M, DEPTH_VIS_MAX_M)
                    depth_norm = ((depth_clipped - DEPTH_VIS_MIN_M)
                                  / (DEPTH_VIS_MAX_M - DEPTH_VIS_MIN_M) * 255).astype(np.uint8)
                    # Invert so near = warm/red, far = cool/blue (usual depth-map
                    # convention) -- JET maps low values to blue, high to red.
                    depth_color = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_JET)
                    cv2.imshow("Drone Depth (colorized)", depth_color)
                    cv2.waitKey(1)

                print(f"\rstep {step:5d} | waypoint {controller.idx}/{len(waypoints)} "
                      f"| map points ~{len(gt_mapper.map_pcd.points):6d}", end="")

            if step_sleep > 0:
                time.sleep(step_sleep)
            step += 1

    except p.error:
        print("\nPyBullet connection lost (window closed mid-step), stopping.")
    finally:
        cv2.destroyAllWindows()
        if p.isConnected(client_id):
            p.disconnect(client_id)
        print()

    if len(gt_trajectory) < 2:
        print("Not enough frames captured -- did the route run at all?")
        return

    gt = np.array(gt_trajectory)

    grid_img = sensed_grid.to_image()
    x_min, x_max, y_min, y_max = sensed_grid.world_extent()
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(grid_img, cmap="gray", vmin=0, vmax=255, extent=(x_min, x_max, y_min, y_max))
    ax.plot(gt[:, 0], gt[:, 1], color="tab:red", linewidth=1.5, label="flight path")
    goals_arr = np.array(room_goals)
    ax.scatter(goals_arr[:, 0], goals_arr[:, 1], color="tab:cyan", marker="*", s=150,
               zorder=5, label="auto-detected room goals")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Occupancy grid (fully automatic room detection + A*)")
    ax.legend()
    fig.tight_layout()
    grid_path = os.path.join(RESULTS_DIR, "occupancy_grid_auto.png")
    fig.savefig(grid_path, dpi=150)
    print(f"Saved occupancy grid to {grid_path}")

    map_path = os.path.join(RESULTS_DIR, "house_map_auto.ply")
    import open3d as o3d
    o3d.io.write_point_cloud(map_path, gt_mapper.get_map())
    print(f"Saved map ({len(gt_mapper.map_pcd.points)} points) to {map_path}")

    plt.show()
    print("Opening point cloud (close window to end)...")
    show_point_cloud(gt_mapper.get_map())


if __name__ == "__main__":
    main()
