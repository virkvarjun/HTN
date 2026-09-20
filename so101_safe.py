"""Send actions to the SO-101 follower only if the pose is inside the safe range.

    from so101_safe import send
    send(robot, {"shoulder_pan.pos": 10.0, ..., "gripper.pos": 30.0})

send() checks the five arm joints with so101_ik.check_pose (SAFE_LIMITS box plus
table clearance). A pose past the limit is not sent: the message says which joint
or link is out of range and UnsafePoseError is raised, because driving the arm past
that position risks breaking the robot. Actions use LeRobot's "<joint>.pos" keys in
degrees (SO101FollowerConfig.use_degrees=True), where 0 is the calibrated midpoint.
"""
import glob
import os

from so101_ik import JOINTS, UnsafePoseError, check_pose


def default_port():
    """The arm's serial port: $SO101_PORT if set, else the one USB serial device present. The SO-101's
    adapter shows up as /dev/tty.usbmodem* on macOS and /dev/ttyACM* on Linux, and its number changes."""
    if os.environ.get("SO101_PORT"):
        return os.environ["SO101_PORT"]
    for pattern in ("/dev/tty.usbmodem*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


def send(robot, action):
    """robot.send_action(action) after the safety check. Raises UnsafePoseError instead of sending."""
    joints = {j: float(action[f"{j}.pos"]) for j in JOINTS}
    msg = check_pose(joints)
    if msg:
        print(msg)
        raise UnsafePoseError(msg)
    return robot.send_action(action)


def arm_config(port):
    """SO101FollowerConfig for this arm: the repo's calibration/so_follower/follower.json when present
    (the motor calibration of this particular arm, so any laptop can drive it), else LeRobot's cache."""
    from pathlib import Path
    from lerobot.robots.so_follower import SO101FollowerConfig
    here = Path(__file__).resolve().parent / "calibration" / "so_follower"
    if (here / "follower.json").exists():
        return SO101FollowerConfig(port=port, id="follower", calibration_dir=here)
    return SO101FollowerConfig(port=port, id="follower")


def connect_with_retries(robot, tries=6, pause_s=0.4):
    """robot.connect(calibrate=False), retried: the Feetech bus now and then returns a corrupted or missing
    status packet during the first writes after connect ("Incorrect status packet", "no status packet")."""
    import time
    last = None
    for attempt in range(tries):
        try:
            robot.connect(calibrate=False)
            return robot
        except ConnectionError as e:
            last = e
            print(f"arm bus glitch on connect ({str(e).split('[')[-1].strip(']')}), retrying ({attempt + 1}/{tries})")
            try:
                robot.bus.port_handler.closePort()
            except Exception:
                pass
            time.sleep(pause_s)
    raise last
