"""Record a grasp demonstration on the SO-101 by guiding the arm by hand.

Run: .venv/bin/python record_demo.py NAME [--port PORT] [--tracker HOST] [--rate 20] [--fake]

Torque is switched off so the arm goes limp: hold it before it drops. Then move the gripper through the
grasp by hand. The script samples all six joints continuously and stores each sample with the gripper
frame's forward kinematics (position in metres in base_link, approach pitch, jaw heading), so the demo can
later be anchored to the quadruped and replayed through the IK at a different position.

Keys while recording:
  space   mark a keyframe (an approach point, the moment the jaws are around the body, the top of the lift)
  g       mark "grasp": the gripper is closed on the robot
  r       mark "release"
  q       finish and save to demos/NAME.json
  x       abort without saving

--tracker HOST (default: pi/host, else qnxpi78.local) reads the Pi tracker while recording and stores the
quadruped's tag pose with every sample and mark. grasp_robot.py needs that to move the grasp to wherever
the quadruped is later. Without a tracker the demo only replays exactly where it was recorded.
"""
import argparse
import json
import os
import select
import sys
import termios
import time
import tty

from so101_safe import default_port, arm_config, connect_with_retries
from so101_ik import JOINTS, fk
from sesame_tracker import Tracker, Poller

LABELS = {" ": "keyframe", "g": "grasp", "r": "release"}


class Arm:
    def __init__(self, port):
        if not port:
            raise SystemExit("no arm found: plug in the SO-101 (a USB serial device), or set SO101_PORT / --port")
        from lerobot.robots.so_follower import SO101Follower
        self.robot = SO101Follower(arm_config(port))
        connect_with_retries(self.robot)
        # The Feetech bus occasionally drops a status packet right after connect; retry the torque-off.
        for attempt in range(5):
            try:
                self.robot.bus.disable_torque(num_retry=3)
                break
            except ConnectionError as e:
                if attempt == 4:
                    raise
                print(f"bus glitch ({str(e).split('[')[0].strip()}), retrying torque-off")
                time.sleep(0.2)

    def read(self):
        return {k[:-4]: float(v) for k, v in self.robot.get_observation().items() if k.endswith(".pos")}

    def close(self):
        self.robot.config.disable_torque_on_disconnect = True
        self.robot.disconnect()


class FakeArm:
    """A slowly moving pose, for trying the recorder without the arm."""
    def __init__(self):
        self.t0 = time.time()

    def read(self):
        import math
        t = time.time() - self.t0
        return {"shoulder_pan": 20 * math.sin(t), "shoulder_lift": 10 + 5 * t, "elbow_flex": 10, "wrist_flex": 60,
                "wrist_roll": 5, "gripper": 40 if t < 2 else 5}

    def close(self):
        pass


class Keys:
    """Non-blocking single keypresses from the terminal (or from a pipe, for tests)."""
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.tty = sys.stdin.isatty()
        if self.tty:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def get(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore(self):
        if self.tty:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--port", default=default_port(), help="arm serial port (default: the USB serial device found, or $SO101_PORT)")
    ap.add_argument("--tracker", help="Pi tracker host; the quadruped's tag pose is recorded alongside the arm")
    ap.add_argument("--no-tracker", action="store_true", help="record without the tracker (replay only at the recorded spot)")
    ap.add_argument("--rate", type=float, default=20.0, help="samples per second")
    ap.add_argument("--fake", action="store_true", help="no hardware, a moving fake pose")
    args = ap.parse_args()

    os.makedirs("demos", exist_ok=True)
    path = os.path.join("demos", f"{args.name}.json")
    if os.path.exists(path):
        print(f"{path} exists, pick another name")
        return 1
    poller = None
    if not args.no_tracker:
        tracker = Tracker(args.tracker)
        first = tracker.wait_for_robot(40.0)
        if first is None:
            print(f"no tracker reported the quadruped at {tracker.host} (units {tracker.units}) in 40 s.")
            print("Are the trackers running (sh run.sh trackers) and is the tag in view? Or record with --no-tracker.")
            return 1
        print(f"tracker: camera {first['unit']} sees the quadruped at ({first['robot']['x']:.1f}, {first['robot']['y']:.1f}) cm heading {first['robot']['heading']:.0f}")
        poller = Poller(tracker, period=0.3)
    arm = FakeArm() if args.fake else Arm(args.port)
    print("torque is OFF, the arm is limp. Guide the gripper by hand.")
    print("space = keyframe   g = grasp   r = release   q = finish and save   x = abort")
    keys = Keys()
    samples, keyframes = [], []
    t0 = time.time()
    saved = False
    try:
        while True:
            t = time.time() - t0
            joints = arm.read()
            pose = fk(joints)
            obs = poller.get() if poller else None
            samples.append({"t": round(t, 3), "joints": {j: round(joints[j], 2) for j in JOINTS}, "gripper": round(joints["gripper"], 1),
                            "pose": {k: round(v, 4) for k, v in pose.items()},
                            "robot_floor": None if obs is None else {**{k: round(v, 2) for k, v in obs["robot"].items()}, "unit": obs["unit"]},
                            "arm_floor": None if obs is None or not obs["arm"] else {k: round(v, 2) for k, v in obs["arm"].items()}})
            key = keys.get()
            if key in LABELS:
                keyframes.append({"t": round(t, 3), "index": len(samples) - 1, "label": LABELS[key]})
                print(f"\n  {LABELS[key]:8s} at {t:5.1f} s: x={pose['x']*100:5.1f} y={pose['y']*100:5.1f} z={pose['z']*100:5.1f} cm  "
                      f"pitch {pose['pitch']:6.1f}  jaw {pose['jaw_yaw']:6.1f}  gripper {joints['gripper']:4.0f}")
            elif key == "q":
                saved = True
                break
            elif key == "x":
                break
            sys.stdout.write(f"\r{t:6.1f} s  x={pose['x']*100:5.1f} y={pose['y']*100:5.1f} z={pose['z']*100:5.1f} cm  pitch {pose['pitch']:6.1f}  "
                             f"jaw {pose['jaw_yaw']:6.1f}  grip {joints['gripper']:4.0f}  [{len(keyframes)} marks]   ")
            sys.stdout.flush()
            time.sleep(max(0.0, 1 / args.rate - (time.time() - t0 - t)))
    finally:
        keys.restore()
        arm.close()
        print()
    if not saved:
        print("aborted, nothing saved. Torque is still off: hold the arm.")
        return 1
    demo = {"name": args.name, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rate_hz": args.rate,
            "frame": "base_link, metres and degrees; pose = gripper_frame_link via so101_ik.fk",
            "tracker": None if poller is None else {"host": tracker.host, "zUp": (poller.get() or first)["zUp"]},
            "samples": samples, "keyframes": keyframes}
    with open(path, "w") as f:
        json.dump(demo, f, indent=1)
    print(f"saved {len(samples)} samples over {samples[-1]['t']:.1f} s with {len(keyframes)} marks to {path}")
    for k in keyframes:
        p = samples[k["index"]]["pose"]
        print(f"  {k['label']:8s} {k['t']:5.1f} s  x={p['x']*100:5.1f} y={p['y']*100:5.1f} z={p['z']*100:5.1f} cm  pitch {p['pitch']:6.1f}  jaw {p['jaw_yaw']:6.1f}")
    print("torque is still off: hold the arm, or run check_arm.py --halfway to stand it up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
