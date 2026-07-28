"""Keyboard teleoperation for DroneRobot using PyBullet's keyboard event API."""
import pybullet as p


CONTROLS_HELP = """
Drone teleop controls:
  W / S       forward / backward
  A / D       strafe left / right
  ARROW-UP    ascend
  ARROW-DOWN  descend
  Q / E       yaw left / right
  ESC         quit
"""


def read_teleop_command(client_id: int):
    """
    Polls the PyBullet GUI keyboard state and returns (vx, vy, vz, yaw_rate, quit),
    each velocity component normalized to [-1, 1].
    """
    keys = p.getKeyboardEvents(physicsClientId=client_id)

    vx = vy = vz = yaw_rate = 0.0
    quit_requested = False

    def held(key):
        return key in keys and keys[key] & p.KEY_IS_DOWN

    if held(ord('w')):
        vx += 1.0
    if held(ord('s')):
        vx -= 1.0
    if held(ord('d')):
        vy -= 1.0
    if held(ord('a')):
        vy += 1.0
    if held(p.B3G_UP_ARROW):
        vz += 1.0
    if held(p.B3G_DOWN_ARROW):
        vz -= 1.0
    if held(ord('e')):
        yaw_rate -= 1.0
    if held(ord('q')):
        yaw_rate += 1.0
    if held(ord('\x1b')):  # ESC
        quit_requested = True

    return vx, vy, vz, yaw_rate, quit_requested
