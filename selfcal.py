"""Self-calibration: the arm holds the Sesame by its hinge and moves it around while the camera watches
the tag. From the known gripper poses and the tag sightings a least-squares fit gives, at once:
  - the floor-to-arm frame (rotation and shift), measured, not assumed from the arm base tag
  - where the tag sits relative to the jaws when they hold the hinge, i.e. the exact grip offset
  - the tag's height while held
    sh run.sh selfcal            (sh run.sh go runs it when no measured frame exists)

One hand action: when the jaws open, place the Sesame's hinge between them, tag up, and press space. The
arm closes on it, visits POSES (skipping any it cannot reach), waits for tag sightings at each, opens
the jaws at the last pose so you can take the Sesame, returns to zero, and writes arm_frame.json with
"measured": true and the grip offset. sesame_pickup uses both.
"""
import argparse
import json
import sys
import time

import numpy as np
from scipy.optimize import least_squares

from arm_frame import ArmFrame, rot, FLOOR_Z_IN_BASE_CM
from record_demo import Keys
from replay_demo import Arm
from sesame_tracker import Tracker, floor_to_pixel, open_camera_page
from so101_ik import JOINTS, ik, fk
from so101_safe import default_port

READY = {j: 0.0 for j in JOINTS}
GRIPPER_OPEN, GRIPPER_CLOSED = 30.0, 0.0
HOLD_Z_CM = 9.5            # gripper frame height while carrying the Sesame around: its tag ends up near the height the tracker expects
# gripper targets in base cm (x, y) and jaw heading (deg); spread out and turned, so the fit is well conditioned
POSES = [(20, 0, 0), (24, 8, 30), (24, -8, -30), (17, 9, 60), (17, -9, -60), (28, 3, 15), (28, -3, -15), (21, 0, 90)]
MIN_POSES = 4
SIGHTINGS_PER_POSE = 4
WAIT_PER_POSE_S = 25.0


def collect(tracker, seconds):
    """Tag-0 sightings over `seconds`: list of (px, camera, heading_floor, unit)."""
    out, end = [], time.time() + seconds
    while time.time() < end and len(out) < SIGHTINGS_PER_POSE:
        for u in tracker.units:
            s = tracker.state(u)
            if s and s.get("calibrated") and s.get("camera") and s.get("robot") and "px" in s["robot"] and "heading" in s["robot"]:
                out.append((s["robot"]["px"], s["camera"], s["robot"]["heading"], u, s.get("zUp", True)))
        time.sleep(0.15)
    return out


def fit(records, mirrored):
    """records: list of (gripper pose dict {x,y,jaw_yaw} in base cm/deg, [(px, camera, heading)]).
    Unknowns: theta, tx, ty (floor -> base: p_b = R(theta) p_f + t), ox, oy (tag in the gripper frame, cm,
    along the jaw axis and to its left), dh (tag heading minus jaw heading), zt (tag height, floor cm)."""
    S = np.diag([1.0, -1.0 if mirrored else 1.0])

    def unpack(p):
        th, tx, ty, ox, oy, dh, zt = p
        return th, np.array([tx, ty]), np.array([ox, oy]), dh, zt

    def residual(p):
        th, t, o, dh, zt = unpack(p)
        Rfb = rot(th) @ S                                   # floor -> base
        res = []
        for g, sightings in records:
            tag_b = np.array([g["x"], g["y"]]) + rot(g["jaw_yaw"]) @ o
            tag_f = np.linalg.solve(Rfb, tag_b - t)
            for px, cam, heading, *_ in sightings:
                u, v = floor_to_pixel(cam, tag_f[0], tag_f[1], zt)
                res += [(u - px[0]) / 10.0, (v - px[1]) / 10.0]           # 10 px ~ 1 cm at this height
                # the tag's heading in the floor frame vs the jaw heading in the base frame
                d = np.linalg.solve(Rfb, np.array([np.cos(np.radians(g["jaw_yaw"] + dh)), np.sin(np.radians(g["jaw_yaw"] + dh))]))
                pred = np.degrees(np.arctan2(d[1], d[0]))
                res.append(((pred - heading + 180) % 360 - 180) / 5.0)   # 5 deg ~ 1 cm of weight
        return np.array(res)

    best = None
    for th0 in (0, 90, 180, 270):                          # the frame rotation is unknown: try each quadrant
        for dh0 in (0, 90, 180, 270):
            p0 = np.array([th0, 0.0, 0.0, 0.0, 0.0, dh0, 10.5])
            sol = least_squares(residual, p0, loss="soft_l1", f_scale=1.0)
            if best is None or sol.cost < best.cost:
                best = sol
    th, t, o, dh, zt = unpack(best.x)
    r = residual(best.x).reshape(-1)
    px_rms = float(np.sqrt(np.mean((r[: len(r) // 3 * 2] * 10) ** 2))) if len(r) else float("nan")
    return {"R": (rot(th) @ S).tolist(), "t": t.tolist(), "mirrored": mirrored, "theta": float(th % 360),
            "tag_in_gripper": {"along": float(o[0]), "left": float(o[1]), "heading_offset": float((dh + 180) % 360 - 180)},
            "tag_height_cm": float(zt), "px_rms": px_rms, "n_records": len(records)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker"); ap.add_argument("--port", default=default_port()); ap.add_argument("--out", default="arm_frame.json")
    ap.add_argument("--fake", action="store_true", help="no arm: simulate (for the test)")
    args = ap.parse_args()
    tracker = Tracker(args.tracker)
    arm = Arm(args.port)
    keys = Keys()
    records, mirrored = [], False
    try:
        print("to zero, jaws open")
        arm.slew(READY, GRIPPER_OPEN)
        first = None
        for p in POSES:
            q = ik(p[0] / 100, p[1] / 100, HOLD_Z_CM / 100, p[2], -90.0) or ik(p[0] / 100, p[1] / 100, HOLD_Z_CM / 100, p[2], -70.0)
            if q:
                first = q; break
        arm.slew(first, GRIPPER_OPEN)
        print("PLACE the Sesame's hinge between the jaws, tag up, then press space (x = abort)")
        while True:
            k = keys.get()
            if k == " ":
                break
            if k == "x":
                print("aborted"); return 1
            time.sleep(0.05)
        print("closing on the hinge")
        for g in np.linspace(GRIPPER_OPEN, GRIPPER_CLOSED, 20):
            arm.send(first, float(g)); time.sleep(0.05)
        time.sleep(0.5)
        for i, (x, y, yaw) in enumerate(POSES):
            q = ik(x / 100, y / 100, HOLD_Z_CM / 100, yaw, -90.0) or ik(x / 100, y / 100, HOLD_Z_CM / 100, yaw, -70.0)
            if q is None:
                print(f"  pose {i + 1}: ({x}, {y}) jaw {yaw} out of reach, skipped"); continue
            arm.slew(q, GRIPPER_CLOSED)
            time.sleep(1.0)
            g = fk(q)
            sightings = collect(tracker, WAIT_PER_POSE_S)
            if sightings:
                mirrored = not sightings[-1][4]
                records.append(({"x": 100 * g["x"], "y": 100 * g["y"], "jaw_yaw": g["jaw_yaw"]}, [s[:4] for s in sightings]))
            print(f"  pose {i + 1}: gripper at ({100*g['x']:.1f}, {100*g['y']:.1f}) jaw {g['jaw_yaw']:.0f}: {len(sightings)} tag sightings")
            if len(records) >= 6:
                break
        print("opening the jaws: take the Sesame")
        for g in np.linspace(GRIPPER_CLOSED, GRIPPER_OPEN, 20):
            arm.send(q, float(g)); time.sleep(0.05)
        time.sleep(2.0)
        arm.slew(READY, GRIPPER_OPEN)
    finally:
        keys.restore()
        arm.close(False)
    if len(records) < MIN_POSES:
        print(f"only {len(records)} poses had tag sightings (need {MIN_POSES}). Is the tag up and in view while held?")
        open_camera_page(tracker.host)
        return 1
    result = fit(records, mirrored)
    frame = ArmFrame(result["R"], result["t"], mirrored, None, residual_cm=result["px_rms"] / 10.0)
    frame.save(args.out)
    with open(args.out) as f:
        d = json.load(f)
    d.update({"measured": True, "grip": {**result["tag_in_gripper"], "tag_height_cm": result["tag_height_cm"], "hold_z_cm": HOLD_Z_CM},
              "tag_height_cm": result["tag_height_cm"], "px_rms": result["px_rms"], "poses": result["n_records"]})
    with open(args.out, "w") as f:
        json.dump(d, f, indent=1)
    o = result["tag_in_gripper"]
    print(f"fit over {result['n_records']} poses: {result['px_rms']:.1f} px rms (about {result['px_rms']/10:.1f} cm)")
    print(f"frame: rotate {result['theta']:.1f} deg, shift ({result['t'][0]:.1f}, {result['t'][1]:.1f}) cm")
    print(f"tag relative to the jaws on the hinge: {o['along']:+.1f} cm along the jaw axis, {o['left']:+.1f} cm to its left, turned {o['heading_offset']:+.0f} deg; tag height {result['tag_height_cm']:.1f} cm")
    print(f"saved {args.out} (measured). sesame_pickup uses this grip offset from now on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
