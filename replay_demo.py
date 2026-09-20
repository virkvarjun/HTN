"""Replay a recorded grasp demonstration on the SO-101, through the safety guard.

Run: .venv/bin/python replay_demo.py NAME [--port PORT] [--speed 1.0] [--squeeze 8] [--dry-run] [--release]

Plays demos/NAME.json (from record_demo.py) back in joint space, so the arm repeats the grasp exactly as
it was shown: same approach, same wrist, same closing point. Before moving it checks every sample against
the safe range and table clearance and refuses if any is outside. It first slews slowly from wherever the
arm is to the demo's first pose, then follows the recording at --speed times the recorded rate. From the
"grasp" mark on, the gripper is commanded --squeeze units tighter than it read during the demo, so it
holds the robot firmly rather than resting at the hand-squeezed reading. At the end the arm holds its
last pose with torque on (the robot stays lifted); --release lets it go limp.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from so101_ik import JOINTS, check_pose
from so101_safe import send, default_port, arm_config, connect_with_retries

FPS = 30
APPROACH_DEG_PER_S = 15.0
GRIPPER_MIN = 0.0


def load(name):
    path = name if os.path.exists(name) else os.path.join("demos", f"{name}.json")
    with open(path) as f:
        return json.load(f)


LEAD_IN_DEG = 2.0       # samples before the arm first moves this far from its start pose are skipped


def build_trajectory(demo, speed, squeeze):
    """(t, joints dict, gripper) per sample, with the squeeze applied after the grasp mark and until release.
    The static lead-in (the arm sitting still before the person starts moving it) is dropped."""
    marks = sorted(demo["keyframes"], key=lambda k: k["t"])
    first = demo["samples"][0]["joints"]
    start = next((i for i, s in enumerate(demo["samples"]) if max(abs(s["joints"][j] - first[j]) for j in JOINTS) > LEAD_IN_DEG), 0)
    start = max(0, start - 5)
    demo = {**demo, "samples": demo["samples"][start:]}
    t_off = demo["samples"][0]["t"]
    grasp_t = next((k["t"] for k in marks if k["label"] == "grasp"), None)
    release_t = next((k["t"] for k in marks if k["label"] == "release" and (grasp_t is None or k["t"] > grasp_t)), None)
    traj = []
    for s in demo["samples"]:
        g = s["gripper"]
        if grasp_t is not None and s["t"] >= grasp_t and (release_t is None or s["t"] < release_t):
            g = max(GRIPPER_MIN, g - squeeze)
        traj.append(((s["t"] - t_off) / speed, {j: float(s["joints"][j]) for j in JOINTS}, float(g)))
    return traj, grasp_t, release_t, t_off


class Arm:
    def __init__(self, port):
        if not port:
            raise SystemExit("no arm found: plug in the SO-101 (a USB serial device), or set SO101_PORT / --port")
        from lerobot.robots.so_follower import SO101Follower
        self.robot = SO101Follower(arm_config(port))
        connect_with_retries(self.robot)

    def read(self):
        return {k[:-4]: float(v) for k, v in self.robot.get_observation().items() if k.endswith(".pos")}

    def send(self, joints, grip):
        send(self.robot, {**{f"{j}.pos": joints[j] for j in JOINTS}, "gripper.pos": grip})

    def slew(self, target, grip):
        start = self.read()
        span = max(abs(target[j] - start[j]) for j in JOINTS)
        n = max(1, int(span / APPROACH_DEG_PER_S * FPS))
        for i in range(1, n + 1):
            self.send({j: start[j] + (target[j] - start[j]) * i / n for j in JOINTS}, grip)
            time.sleep(1 / FPS)

    def close(self, release):
        self.robot.config.disable_torque_on_disconnect = release
        self.robot.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--port", default=default_port(), help="arm serial port (default: the USB serial device found, or $SO101_PORT)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    ap.add_argument("--squeeze", type=float, default=8.0, help="gripper units tighter than the demo while grasping")
    ap.add_argument("--dry-run", action="store_true", help="check and print the plan, do not connect")
    ap.add_argument("--release", action="store_true", help="torque off at the end (the arm drops what it holds)")
    args = ap.parse_args()

    demo = load(args.name)
    traj, grasp_t, release_t, t_off = build_trajectory(demo, args.speed, args.squeeze)
    if t_off:
        print(f"skipping the first {t_off:.1f} s: the arm had not moved yet")
    bad = [(i, msg) for i, (_, q, _) in enumerate(traj) if (msg := check_pose(q))]
    print(f"demo '{demo['name']}': {len(traj)} samples, {traj[-1][0]:.1f} s at speed {args.speed}, "
          f"grasp at {grasp_t}, release at {release_t}, squeeze {args.squeeze}")
    for k in sorted(demo["keyframes"], key=lambda k: k["t"]):
        p = demo["samples"][k["index"]]["pose"]
        print(f"  {k['label']:8s} {k['t']:5.1f} s  x={p['x']*100:5.1f} y={p['y']*100:5.1f} z={p['z']*100:5.1f} cm  pitch {p['pitch']:6.1f}  jaw {p['jaw_yaw']:6.1f}")
    if bad:
        print(f"REFUSED: {len(bad)} of {len(traj)} samples are outside the safe range. First: sample {bad[0][0]} at {traj[bad[0][0]][0]:.1f} s")
        print("  " + bad[0][1])
        print("  Re-record the demo inside the range, or widen SAFE_LIMITS in so101_ik.py after checking the arm by hand.")
        return 1
    print("all samples inside the safe range")
    if args.dry_run:
        return 0

    arm = Arm(args.port)
    try:
        t0, q0, g0 = traj[0]
        print("slewing to the demo's first pose")
        arm.slew(q0, g0)
        time.sleep(0.5)
        print("replaying")
        start = time.perf_counter()
        i = 0
        while i < len(traj):
            now = time.perf_counter() - start
            while i + 1 < len(traj) and traj[i + 1][0] <= now:
                i += 1
            t, q, g = traj[i]
            if i + 1 < len(traj):                     # interpolate between samples for a smooth send at FPS
                t2, q2, g2 = traj[i + 1]
                f = 0.0 if t2 <= t else min(1.0, (now - t) / (t2 - t))
                q = {j: q[j] + (q2[j] - q[j]) * f for j in JOINTS}
                g = g + (g2 - g) * f
            arm.send(q, g)
            if i + 1 >= len(traj):
                break
            time.sleep(1 / FPS)
        print("done: holding the final pose")
    finally:
        arm.close(args.release)
        print("torque released" if args.release else "torque left on (use --release to let go)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
