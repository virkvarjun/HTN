"""Offline test of the pick-and-place chain, no arm and no Pi: a fake tracker on localhost, a demo with a
known tag pose, and a known floor-to-arm frame. Checks that the grasp lands at the demonstrated offset
from the tag wherever the tag is, that a moved arm base tag is corrected for, that a mirrored floor frame
works, and that every phase of sesame_pickup's plan solves.

    .venv/bin/python test_pickup.py            (or: pytest test_pickup.py)
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

from arm_frame import ArmFrame, rot
from fake_tracker import serve
from grasp_robot import grasp_segment, tag_pose_at, relative_offsets, place, solve
from replay_demo import load
from sesame_tracker import Tracker
from so101_ik import fk, ik, JOINTS
import sesame_pickup

DEMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos", "grip.json")


def synthetic_demo():
    """A grasp demonstration made up for the test: a top-down descent from 5 cm above to a grip at 9.4 cm
    height, jaws closing, then a short lift. Same layout as record_demo.py writes, so the test needs no
    recording on disk."""
    samples, keyframes = [], []
    t = 0.0
    x, y, yaw = 0.229, 0.051, -173.4
    # the approach comes in tilted and straightens to vertical for the grip, as a hand-guided one does:
    # held straight down this arm reaches nothing above about 12 cm
    for z, pitch, grip, label in [(0.125, -65, 22, None), (0.115, -72, 22, None), (0.105, -80, 22, None), (0.098, -87, 22, "keyframe"), (0.094, -90, 22, None),
                                  (0.094, -90, 8, None), (0.094, -90, 0, "grasp"), (0.094, -90, 0, None), (0.094, -90, 0, None),
                                  (0.098, -87, 0, None), (0.105, -80, 0, None), (0.115, -72, 0, "keyframe")]:
        q = ik(x, y, z, yaw, pitch)
        assert q is not None, "the synthetic grasp pose must be reachable"
        p = fk(q)
        samples.append({"t": round(t, 3), "joints": {j: float(q[j]) for j in JOINTS}, "gripper": float(grip), "pose": p})
        if label:
            keyframes.append({"t": round(t, 3), "index": len(samples) - 1, "label": label})
        t += 0.5
    return {"name": "synthetic", "rate_hz": 2.0, "samples": samples, "keyframes": keyframes}


def load_demo():
    return load(DEMO) if os.path.exists(DEMO) else synthetic_demo()


def tracked_demo(demo, frame, tag_base_heading_from_jaw=90.0):
    """Copy of a demo with the tag pose that would have been recorded: the tag under the gripper at the grasp."""
    d = json.loads(json.dumps(demo))
    g = next(k for k in d["keyframes"] if k["label"] == "grasp")
    p = d["samples"][g["index"]]["pose"]
    base_xy = np.array([100 * p["x"], 100 * p["y"]])
    heading_base = p["jaw_yaw"] + tag_base_heading_from_jaw
    floor_xy = frame.to_floor(base_xy)[0]
    d_floor = frame.R.T @ [np.cos(np.radians(heading_base)), np.sin(np.radians(heading_base))]
    heading_floor = float(np.degrees(np.arctan2(d_floor[1], d_floor[0])))
    for s in d["samples"]:
        s["robot_floor"] = {"x": float(floor_xy[0]), "y": float(floor_xy[1]), "z": 10.5, "heading": heading_floor, "unit": 3}
        s["arm_floor"] = dict(frame.arm_tag, z=5.0)
    d["tracker"] = {"host": "localhost", "zUp": not frame.mirrored}
    return d


def grasp_offset(frame, demo, obs):
    """(along, across, z, jaw_rel) of the solved grasp relative to the tag now, and the demo's."""
    samples, grasp, release = grasp_segment(demo, 4.0, 2.0)
    tag_demo = frame.adjusted_for_arm_tag(demo["samples"][grasp["index"]]["arm_floor"]).pose_to_base(tag_pose_at(grasp, demo))
    frame_now = frame.adjusted_for_arm_tag(obs["arm"])
    tag_now = frame_now.pose_to_base(obs["robot"])
    offsets = relative_offsets(samples, tag_demo)
    poses = place(offsets, tag_now)
    traj, failed = solve(poses, grasp["t"], release["t"] if release else None, 8.0)
    gi = next(i for i, s in enumerate(samples) if s["t"] >= grasp["t"])
    if failed:
        return None, None, failed
    f = fk(traj[gi][1])
    d = rot(-tag_now["heading"]) @ (np.array([100 * f["x"], 100 * f["y"]]) - [tag_now["x"], tag_now["y"]])
    jaw = (f["jaw_yaw"] - tag_now["heading"] + 90) % 180 - 90
    og = offsets[gi]
    return (d[0], d[1], 100 * f["z"], jaw), (og["along"], og["across"], og["z"], (og["jaw_rel"] + 90) % 180 - 90), failed


def run_case(name, frame, robot_floor, arm_floor, zup, expect_fail=False):
    servers = serve(robot_floor, arm_floor, zup, noise=0.02)
    try:
        time.sleep(0.2)
        demo = tracked_demo(load_demo(), frame)
        obs = Tracker("localhost").observe_steady(0.5)
        assert obs is not None, "fake tracker not seen"
        got, want, failed = grasp_offset(frame, demo, obs)
        if expect_fail:
            assert failed, f"{name}: expected an out-of-reach refusal"
            print(f"  PASS {name}: refused ({len(failed)} poses out of reach)")
            return
        assert not failed, f"{name}: {len(failed)} poses unreachable"
        err = max(abs(g - w) for g, w in zip(got[:3], want[:3]))
        jerr = abs((got[3] - want[3] + 90) % 180 - 90)
        assert err < 0.05 and jerr < 0.2, f"{name}: grasp offset off by {err:.3f} cm / {jerr:.2f} deg"
        print(f"  PASS {name}: grasp at the demonstrated offset (err {err:.3f} cm, {jerr:.2f} deg)")
        # sesame_pickup's full plan
        args = argparse.Namespace(drop=None, drop_offset=None, lift=6.0, approach=4.0, squeeze=8.0)
        traj, info = sesame_pickup.plan(demo, frame, obs, args)
        assert traj is not None, f"{name}: pickup plan refused: {info}"
        assert [n for n, _, _ in info["phases"]] == ["grip", "lift", "carry", "lower", "release", "retract"], info["phases"]
        assert all(done == total for _, done, total in info["phases"]), info["phases"]
        print(f"  PASS {name}: pickup plan, all 6 phases solve, carry pitch {info['carry_pitch']:.0f}, {info['seconds']:.1f} s")
    finally:
        for s in servers:
            s.shutdown()


def main():
    print("demo:", "recorded " + DEMO if os.path.exists(DEMO) else "synthetic (no recording on disk)")
    base = ArmFrame(rot(180.0), np.array([70.0, 30.0]), False, {"x": 70.0, "y": 8.0, "heading": 180.0}, 0.05, [3])
    run_case("tag moved and turned", base, (46, 28, 75), (70, 8, 180), True)
    run_case("arm base tag moved 3 cm (board shifted)", base, (49, 28, 75), (73, 8, 180), True)
    mirrored = ArmFrame(rot(180.0) @ np.diag([1, -1]), np.array([70.0, 30.0]), True, {"x": 70.0, "y": 8.0, "heading": 180.0}, 0.05, [3])
    run_case("mirrored floor frame (zUp false)", mirrored, (46, -28, -75), (70, 8, 180), False)
    run_case("too far away", base, (40, 30, 60), (70, 8, 180), True, expect_fail=True)
    print("RESULT: PASS")
    return 0


def test_pickup_chain():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
