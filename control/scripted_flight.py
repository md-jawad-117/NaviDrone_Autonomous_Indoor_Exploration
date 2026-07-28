"""
Waypoint-following flight controller: replaces manual keyboard teleop with a
deterministic, repeatable patrol route. Deliberately always moves forward while
turning (never stops to spin in place) -- pure in-place rotation was measured
to be VO's worst case in this house, so this controller sidesteps that failure
mode by construction rather than trying to make VO robust to it.
"""
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# (vx, vy, yaw_rate) for each successive escape attempt. Varying the direction
# matters: repeating the identical push on every retry can just drive the
# drone back into the same obstacle corner it's already wedged against.
# Magnitudes are re-scaled whenever house_config's max_linear_speed changes,
# to keep the *absolute* escape speed roughly constant (~0.55-0.65 m/s)
# regardless of normal cruise speed -- backing off from a collision should
# stay gentle even as cruise flight gets faster.
_ESCAPE_MANEUVERS = [
    (-0.25, 0.2, 0.5),
    (-0.25, -0.2, -0.5),
    (-0.29, 0.0, 0.6),
    (-0.29, 0.0, -0.6),
]


class WaypointController:
    def __init__(self, waypoints: Sequence[Sequence[float]], config: dict,
                 scan_indices: Optional[set] = None):
        """
        Args:
            waypoints: list of (x, y) or (x, y, z) target points in world frame.
            config: parsed config/flight_controller.yaml.
            scan_indices: waypoint indices (typically room-goal arrival points,
                not transit points along the way) where the drone should pause
                and rotate in place before moving on. A straight-line dash into
                a room and back out again only ever points the camera at
                whatever's directly ahead, missing walls to the sides -- a scan
                sweeps the whole room instead. Safe here specifically because
                this route's map is built from ground-truth pose, not VO (pure
                rotation is VO's hardest case elsewhere in this project).
        """
        self.waypoints = [np.array(wp, dtype=np.float64) for wp in waypoints]
        self.cfg = config
        self.idx = 0
        self.done = len(self.waypoints) == 0

        self.scan_indices = scan_indices or set()
        self._scanned_indices_done = set()
        self._scan_steps_remaining = 0

        # Stuck detection: this controller has no real obstacle avoidance (it's
        # a straight-line pursuit), so a planned path that's geometrically fine
        # in the discretized grid can still clip a doorway edge in continuous
        # space and get physically wedged. Rather than hand-tune every route,
        # detect near-zero progress and back out, retrying a few times before
        # giving up on that waypoint and moving on. Tracks distance-to-target,
        # not raw displacement: a drone oscillating near an obstacle can move
        # a real distance each check interval without ever getting closer to
        # where it's actually trying to go, which raw-displacement checking
        # doesn't catch (confirmed in practice: stuck 100k+ steps at one
        # waypoint with the old raw-displacement check never triggering).
        self._call_count = 0
        self._idx_at_last_check = None
        self._dist_at_last_check = None
        self._stuck_intervals = 0
        self._escape_steps_remaining = 0
        self._escape_attempts = 0

    def get_command(self, position: np.ndarray, yaw: float) -> Tuple[float, float, float, float, bool]:
        """
        Args:
            position: current (x, y, z) world position.
            yaw: current yaw in radians.

        Returns:
            (vx, vy, vz, yaw_rate, done) -- vx/vy/vz/yaw_rate normalized to
            [-1, 1] as expected by DroneRobot.set_velocity_command.
        """
        if self.done:
            return 0.0, 0.0, 0.0, 0.0, True

        if self._scan_steps_remaining > 0:
            self._scan_steps_remaining -= 1
            target = self.waypoints[self.idx]
            vz_cmd = 0.0
            if len(target) > 2:
                vz_cmd = np.clip(self.cfg["kp_vertical"] * (target[2] - position[2]), -1.0, 1.0)
            return 0.0, 0.0, vz_cmd, self.cfg.get("scan_yaw_rate", 0.5), False

        target_for_stuck_check = self.waypoints[self.idx]
        dist_to_target = np.linalg.norm(target_for_stuck_check[:2] - position[:2])
        stuck_event = self._update_stuck_tracking(self.idx, dist_to_target)
        if stuck_event == "escalate":
            self._escape_attempts += 1
            if self._escape_attempts > self.cfg.get("max_escape_attempts", 3):
                self._escape_attempts = 0
                self.idx += 1
                if self.idx >= len(self.waypoints):
                    self.done = True
                    return 0.0, 0.0, 0.0, 0.0, True
            else:
                self._escape_steps_remaining = self.cfg.get("escape_duration_steps", 120)

        if self._escape_steps_remaining > 0:
            self._escape_steps_remaining -= 1
            vx, vy, yaw_rate = _ESCAPE_MANEUVERS[(self._escape_attempts - 1) % len(_ESCAPE_MANEUVERS)]
            # Still correct altitude during an escape, not just during normal
            # pursuit: a collision nudge during the escape (the whole reason
            # an escape is happening) can shift altitude, and with escape vz
            # hardcoded to 0 there was nothing to pull it back afterward --
            # over a long flight with several escapes, that drift accumulated
            # (observed: cruise altitude climbing from 1.0m to 2.0m over ~400s).
            target = self.waypoints[self.idx]
            vz_cmd = 0.0
            if len(target) > 2:
                vz_cmd = np.clip(self.cfg["kp_vertical"] * (target[2] - position[2]), -1.0, 1.0)
            return vx, vy, vz_cmd, yaw_rate, False

        target = self.waypoints[self.idx]
        delta_xy = target[:2] - position[:2]
        dist = np.linalg.norm(delta_xy)

        if dist < self.cfg["arrival_radius"]:
            if self.idx in self.scan_indices and self.idx not in self._scanned_indices_done:
                self._scanned_indices_done.add(self.idx)
                self._scan_steps_remaining = self.cfg.get("scan_duration_steps", 1500)
                return 0.0, 0.0, 0.0, self.cfg.get("scan_yaw_rate", 0.5), False

            self.idx += 1
            if self.idx >= len(self.waypoints):
                self.done = True
                return 0.0, 0.0, 0.0, 0.0, True
            target = self.waypoints[self.idx]
            delta_xy = target[:2] - position[:2]
            dist = np.linalg.norm(delta_xy)

        desired_heading = math.atan2(delta_xy[1], delta_xy[0])
        yaw_error = _normalize_angle(desired_heading - yaw)

        yaw_rate_cmd = np.clip(self.cfg["kp_yaw"] * yaw_error, -1.0, 1.0)
        # Turn tightly (down to a near-stop) before committing to forward motion --
        # this controller has no obstacle avoidance, so precisely tracking the
        # straight line to each waypoint matters more here than avoiding an
        # in-place turn (which only matters for VO's sake, and this route's
        # primary map is built from ground-truth pose, not VO).
        forward_cmd = np.clip(self.cfg["kp_linear"] * dist, 0.0, 1.0) * max(0.0, math.cos(yaw_error)) ** 3

        vz_cmd = 0.0
        if len(target) > 2:
            vz_cmd = np.clip(self.cfg["kp_vertical"] * (target[2] - position[2]), -1.0, 1.0)

        return forward_cmd, 0.0, vz_cmd, yaw_rate_cmd, False

    def force_skip_current_waypoint(self):
        """Abandons the current waypoint and resets escape/stuck state. Meant
        to be called externally (e.g. by a caller's own safety monitor) after
        recovering from a physically-impossible position: just resetting the
        drone's position/velocity without also changing strategy means the
        controller immediately retries the exact same approach that caused the
        problem, which can repeat or even compound the failure step by step."""
        self._escape_steps_remaining = 0
        self._escape_attempts = 0
        self._stuck_intervals = 0
        self._idx_at_last_check = None
        self._dist_at_last_check = None
        self._scan_steps_remaining = 0
        self.idx += 1
        if self.idx >= len(self.waypoints):
            self.done = True

    def _update_stuck_tracking(self, idx: int, dist_to_target: float) -> Optional[str]:
        """Returns 'escalate' if enough consecutive low-progress checks have
        elapsed to warrant an escape maneuver (or giving up on the waypoint),
        else None. Progress is checked every stuck_check_interval calls, as
        distance-to-target getting closer (not raw movement -- see __init__)."""
        self._call_count += 1
        interval = self.cfg.get("stuck_check_interval", 240)
        if self._call_count % interval != 0:
            return None

        if self._idx_at_last_check == idx and self._dist_at_last_check is not None:
            progress = self._dist_at_last_check - dist_to_target
            if progress < self.cfg.get("stuck_progress_threshold", 0.15):
                self._stuck_intervals += 1
            else:
                self._stuck_intervals = 0
                self._escape_attempts = 0
        else:
            # Target changed since the last check -- that's real progress.
            self._stuck_intervals = 0
            self._escape_attempts = 0

        self._idx_at_last_check = idx
        self._dist_at_last_check = dist_to_target

        if (self._stuck_intervals >= self.cfg.get("max_stuck_intervals", 1) and
                self._escape_steps_remaining <= 0):
            self._stuck_intervals = 0
            return "escalate"
        return None
