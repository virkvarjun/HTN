"""Grip the quadruped wherever it is: find its tag through the Pi tracker, move the recorded grasp there
with the IK, and replay the grip and release the way it was demonstrated.

Run: .venv/bin/python grasp_robot.py DEMO [--tracker HOST] [--frame arm_frame.json] [--port PORT]
                                     [--approach 4] [--retract 2] [--speed 1] [--squeeze 8] [--dry-run] [--release]

Needs: arm_frame.json from calibrate_arm_frame.py, and a demo recorded with the tracker running
(record_demo.py), so the quadruped's tag pose at the moment of the grasp is known.

How the grasp moves:
  1. The demo's gripper poses around the grasp (from --approach seconds before the "grasp" mark to
     --retract seconds after "release") are expressed relative to the quadruped's tag as it was during the
     demo: an offset along and across the tag's heading, a height, and a jaw angle relative to the heading.
  2. The tracker gives the tag's pose now; the same offsets around it give the new gripper poses.
  3. Each pose goes through so101_ik.ik (approach pitch as demonstrated), then so101_safe.send. If any
     pose is unreachable or unsafe the run is refused before anything moves.
The gripper opens and closes on the demo's timing, squeezed --squeeze units tighter while grasping.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from arm_frame import ArmFrame, rot
from replay_demo import Arm, load, FPS
from sesame_tracker import Tracker
from so101_safe import default_port
from so101_ik import JOINTS, ik, fk

GRIPPER_MIN = 0.0
PRE_APPROACH_CM = 5.0        # start this far above the first transformed pose


def grasp_segment(demo, approach_s, retract_s):
    marks = sorted(demo["keyframes"], key=lambda k: k["t"])
    grasp = next((k for k in marks if k["label"] == "grasp"), None)
    if grasp is None:
        raise SystemExit("the demo has no 'grasp' mark")
    release = next((k for k in marks if k["label"] == "release" and k["t"] > grasp["t"]), None)
    t0 = grasp["t"] - approach_s
    t1 = (release["t"] if release else demo["samples"][-1]["t"]) + retract_s
    samples = [s for s in demo["samples"] if t0 <= s["t"] <= t1]
    return samples, grasp, release


def tag_pose_at(sample_or_mark, demo):
    """The quadruped's floor pose stored with a sample (the recorder's poller), or None."""
    if "robot_floor" in sample_or_mark:
        return sample_or_mark["robot_floor"]
    return demo["samples"][sample_or_mark["index"]].get("robot_floor")


def relative_offsets(samples, tag_base):
    """Each sample's gripper pose relative to the tag pose (base cm, deg): (along, across, z, jaw_rel, pitch, gripper, t)."""
    Rinv = rot(-tag_base["heading"])
    out = []
    for s in samples:
        p = s["pose"]
        d = Rinv @ (np.array([100 * p["x"], 100 * p["y"]]) - [tag_base["x"], tag_base["y"]])
        out.append({"along": float(d[0]), "across": float(d[1]), "z": 100 * p["z"], "jaw_rel": (p["jaw_yaw"] - tag_base["heading"] + 180) % 360 - 180,
                    "pitch": p["pitch"], "gripper": s["gripper"], "t": s["t"]})
    return out


def place(offsets, tag_base):
    """Offsets around a new tag pose -> gripper poses in base (m, deg)."""
    R = rot(tag_base["heading"])
    poses = []
    for o in offsets:
        xy = np.array([tag_base["x"], tag_base["y"]]) + R @ [o["along"], o["across"]]
        poses.append({"x": xy[0] / 100, "y": xy[1] / 100, "z": o["z"] / 100, "jaw_yaw": (o["jaw_rel"] + tag_base["heading"] + 180) % 360 - 180,
                      "pitch": o["pitch"], "gripper": o["gripper"], "t": o["t"]})
    return poses


def solve(poses, grasp_t, release_t, squeeze):
    traj, failed = [], []
    t_off = poses[0]["t"]
    for i, p in enumerate(poses):
        q = ik(p["x"], p["y"], p["z"], p["jaw_yaw"], p["pitch"], exact_jaw=True)
        if q is None:
            failed.append((i, p))
            continue
        g = p["gripper"]
        if p["t"] >= grasp_t and (release_t is None or p["t"] < release_t):
            g = max(GRIPPER_MIN, g - squeeze)
        traj.append((p["t"] - t_off, q, float(g)))
    return traj, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo")
    ap.add_argument("--tracker", help="Pi host (default: pi/host, else qnxpi78.local)")
    ap.add_argument("--frame", default="arm_frame.json")
    ap.add_argument("--port", default=default_port(), help="arm serial port (default: the USB serial device found, or $SO101_PORT)")
    ap.add_argument("--approach", type=float, default=4.0, help="seconds of the demo before the grasp mark to replay")
    ap.add_argument("--retract", type=float, default=2.0, help="seconds after the release mark to replay")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--squeeze", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--release", action="store_true", help="torque off at the end")
    args = ap.parse_args()

    try:
        demo = load(args.demo)
    except FileNotFoundError:
        print(f"no demo '{args.demo}': record one with record_demo.py (with the tracker running)"); return 1
    if not os.path.exists(args.frame):
        print(f"{args.frame} not found: run calibrate_arm_frame.py first"); return 1
    frame0 = ArmFrame.load(args.frame)
    samples, grasp, release = grasp_segment(demo, args.approach, args.retract)
    tag_demo_floor = tag_pose_at(grasp, demo)
    if not tag_demo_floor:
        print("the demo has no tracker pose at the grasp mark: record it again with the tracker running (record_demo.py without --no-tracker)")
        return 1
    demo_arm_tag = demo["samples"][grasp["index"]].get("arm_floor")
    frame_demo = frame0.adjusted_for_arm_tag(demo_arm_tag)
    tag_demo = frame_demo.pose_to_base(tag_demo_floor)

    tracker = Tracker(args.tracker)
    print(f"reading the tracker at {tracker.host}, units {tracker.units}")
    obs = tracker.wait_for_robot(30.0)
    if obs is None:
        print("no camera reported the quadruped's tag in 30 s"); return 1
    frame_now = frame0.adjusted_for_arm_tag(obs["arm"])
    if frame_now is not frame0:
        print(f"arm base tag moved since calibration, transform re-derived from it: {frame_now.describe()}")
    tag_now = frame_now.pose_to_base(obs["robot"])
    print(f"quadruped now: floor ({obs['robot']['x']:.1f}, {obs['robot']['y']:.1f}) heading {obs['robot']['heading']:.0f}  ->  "
          f"base ({tag_now['x']:.1f}, {tag_now['y']:.1f}) cm heading {tag_now['heading']:.0f}   [camera {obs['unit']}, {obs.get('samples', 1)} readings]")
    print(f"during the demo:  floor ({tag_demo_floor['x']:.1f}, {tag_demo_floor['y']:.1f}) heading {tag_demo_floor['heading']:.0f}  ->  "
          f"base ({tag_demo['x']:.1f}, {tag_demo['y']:.1f}) cm heading {tag_demo['heading']:.0f}")

    offsets = relative_offsets(samples, tag_demo)
    gi = next(i for i, s in enumerate(samples) if s["t"] >= grasp["t"])
    og = offsets[gi]
    print(f"grasp offset from the tag: {og['along']:+.1f} cm along, {og['across']:+.1f} cm across, {og['z']:.1f} cm up, jaws at {og['jaw_rel']:+.0f} deg to the heading, pitch {og['pitch']:.0f}")
    poses = place(offsets, tag_now)
    # start a little above the first pose when the arm can reach it, so the descent is vertical
    for lift in (PRE_APPROACH_CM, PRE_APPROACH_CM / 2, 1.0):
        pre = dict(poses[0]); pre["z"] += lift / 100; pre["t"] = poses[0]["t"] - 1.5
        if ik(pre["x"], pre["y"], pre["z"], pre["jaw_yaw"], pre["pitch"]) is not None:
            poses = [pre] + poses
            gi += 1
            break
    traj, failed = solve(poses, grasp["t"], release["t"] if release else None, args.squeeze)
    pg = poses[gi]
    print(f"grasp pose now: x={100*pg['x']:.1f} y={100*pg['y']:.1f} z={100*pg['z']:.1f} cm, jaws {pg['jaw_yaw']:.0f} deg, {len(traj)} of {len(poses)} poses solved")
    if failed:
        i, p = failed[0]
        print(f"REFUSED: {len(failed)} poses unreachable or unsafe. First at t={p['t']:.1f} s: x={100*p['x']:.1f} y={100*p['y']:.1f} z={100*p['z']:.1f} cm jaws {p['jaw_yaw']:.0f}")
        print("  Move the quadruped closer to the arm, or turn it so the demonstrated approach side faces the arm.")
        return 1
    # consistency: the solved grasp pose, via FK, must sit at the demonstrated offset from the new tag
    q = next(q for t, q, g in traj if abs(t - (pg["t"] - poses[0]["t"])) < 1e-9); f = fk(q)
    Rinv = rot(-tag_now["heading"]); d = Rinv @ (np.array([100 * f["x"], 100 * f["y"]]) - [tag_now["x"], tag_now["y"]])
    print(f"check: solved grasp is {d[0]:+.2f} along, {d[1]:+.2f} across, {100*f['z']:.2f} up, jaws {((f['jaw_yaw'] - tag_now['heading'] + 90) % 180 - 90):+.1f} deg (demo {og['along']:+.2f}, {og['across']:+.2f}, {og['z']:.2f}, {((og['jaw_rel'] + 90) % 180 - 90):+.1f})")
    if args.dry_run:
        return 0

    arm = Arm(args.port)
    try:
        _, q0, g0 = traj[0]
        print("slewing above the quadruped")
        arm.slew(q0, g0)
        time.sleep(0.5)
        print("grasping")
        start = time.perf_counter()
        i = 0
        while i < len(traj):
            now = (time.perf_counter() - start) * args.speed
            while i + 1 < len(traj) and traj[i + 1][0] <= now:
                i += 1
            t, q, g = traj[i]
            if i + 1 < len(traj):
                t2, q2, g2 = traj[i + 1]
                fr = 0.0 if t2 <= t else min(1.0, (now - t) / (t2 - t))
                q = {j: q[j] + (q2[j] - q[j]) * fr for j in JOINTS}
                g = g + (g2 - g) * fr
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
