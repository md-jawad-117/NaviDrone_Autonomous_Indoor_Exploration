"""
Reactive braking: casts a short ray in the drone's intended direction of
travel each step and scales down forward/lateral speed as it nears an
obstacle, instead of committing to full speed and only reacting after a
collision (which is what the stuck-detection + escape maneuver in
control/scripted_flight.py does -- a recovery net, not prevention). This is
meant to run alongside that recovery net, not replace it: it should make wall
bumps rare, but the recovery net still catches whatever this misses (e.g. a
sharp turn where the ray doesn't point at the actual obstacle).
"""
import numpy as np
import pybullet as p


def apply_braking(client_id: int, drone_position: np.ndarray, drone_orientation,
                   vx: float, vy: float, braking_distance: float = 1.2,
                   min_safe_distance: float = 0.45, ray_start_offset: float = 0.2):
    """
    Args:
        drone_position: current (x, y, z) world position.
        drone_orientation: current orientation quaternion.
        vx, vy: the controller's intended body-frame forward/lateral command,
            normalized to [-1, 1] (same convention as DroneRobot.set_velocity_command).
        braking_distance: start slowing down when an obstacle is within this range.
        min_safe_distance: full stop (of the forward/lateral component) once
            an obstacle is this close.
        ray_start_offset: cast the ray from this far in front of the drone's
            center, not the center itself -- otherwise it immediately self-
            intersects the drone's own collision shape at ~zero distance.

    Returns:
        (vx_scaled, vy_scaled) -- same direction, reduced magnitude if an
        obstacle is ahead. vz and yaw_rate are left untouched by the caller;
        this only brakes horizontal translation.
    """
    horizontal_cmd = np.array([vx, vy])
    cmd_norm = np.linalg.norm(horizontal_cmd)
    if cmd_norm < 1e-6:
        return vx, vy

    rot = np.array(p.getMatrixFromQuaternion(drone_orientation)).reshape(3, 3)
    world_dir = rot @ np.array([vx, vy, 0.0])
    world_dir_2d = world_dir[:2]
    world_dir_norm = np.linalg.norm(world_dir_2d)
    if world_dir_norm < 1e-6:
        return vx, vy
    world_dir_2d = world_dir_2d / world_dir_norm

    start = drone_position[:2] + world_dir_2d * ray_start_offset
    end = start + world_dir_2d * braking_distance
    result = p.rayTest([start[0], start[1], drone_position[2]],
                        [end[0], end[1], drone_position[2]],
                        physicsClientId=client_id)[0]

    if result[0] == -1:
        return vx, vy  # nothing within braking_distance

    clearance = result[2] * braking_distance + ray_start_offset
    if clearance <= min_safe_distance:
        scale = 0.0
    else:
        scale = (clearance - min_safe_distance) / (braking_distance - min_safe_distance)
        scale = min(1.0, max(0.0, scale))

    return vx * scale, vy * scale
