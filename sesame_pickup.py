"""Press a key: the arm finds the Sesame through the Pi tracker, grips it the demonstrated way, lifts it,
carries it a short way to the side, sets it down, lets go and returns to its ready pose.

Run: .venv/bin/python sesame_pickup.py [DEMO|auto] [--tracker HOST] [--frame arm_frame.json] [--port PORT]
                                       [--grip-along 0] [--grip-across 0] [--grip-above-tag 0.7] [--jaw-angle 90]
                                       [--drop-offset DX DY | --drop X Y] [--lift 6] [--once] [--dry-run]

With no demo (or "auto") the grasp is built from the tag itself: come down vertically onto the point
--grip-along / --grip-across cm from the tag's centre (along its heading, and to its left), close the jaws
--grip-above-tag cm above the tag's reported height, jaws at --jaw-angle to the heading (90 = across the
body). Those are the numbers a recorded demo would supply; the defaults are the ones measured from one.

Correction loop (auto mode): after each attempt the camera says whether the Sesame's tag moved with the
gripper. If it did not, the grip missed: the arm returns to zero, the target is shifted by the next trial
offset (closer, further, left, right, in cm) and it tries again; the offset that works is kept in
grip_correction.json and used from then on. Keys c / f / l / r / u / d nudge the target 1 cm closer /
further / left / right / up / down by hand between attempts (also kept), 0 clears it.

--auto (what sh run.sh go uses): no keypress. Whenever a camera sees the Sesame within reach, the arm grips
it, carries it aside, sets it down and returns; then it waits until the Sesame is seen somewhere else
(more than RETRIGGER_CM from where it was set down) before going again. q still quits.

Keys:  space / g  find the Sesame and run the whole pick-and-place
       p          plan only: print every phase and whether it is reachable, no motion
       z          go to the rest pose (folded, where a run starts)
       o          open the gripper where it is
       q          quit (torque stays on)

Where it puts the Sesame down: --drop-offset DX DY is in cm relative to the pick point in the arm's base
frame (+y is the arm's left); the default tries a few nearby offsets and takes the first the arm can reach.
--drop X Y is an absolute base-frame position in cm. The end-position logic can replace pick_drop() later.

Phases, all solved through so101_ik.ik and checked before the first move:
  grip      the demo's approach and close, placed around the Sesame's current tag pose (grasp_robot.py)
  lift      straight up by --lift cm. Held vertically the arm reaches nothing above about 12 cm, so the
            gripper tilts on the way up, to the least tilt that reaches the carry height (the Sesame is
            already held, so a tilt does no harm)
  carry     straight line at that height to the drop point, jaws kept at the same angle to the body
  lower     straight down to the grasp height, tilting back to vertical for the release
  release   open the gripper to the demo's open value
  retract   straight up, then a slow joint-space slew to zero and down into the folded rest pose
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from arm_frame import ArmFrame, rot
from frame_from_arm_tag import frame_from_tag
from grasp_robot import grasp_segment, tag_pose_at, relative_offsets, place, solve, PRE_APPROACH_CM
from record_demo import Keys
from replay_demo import Arm, load, FPS
from sesame_tracker import Tracker, Poller
from so101_safe import default_port
from so101_ik import JOINTS, ik, fk, TABLE_Z

HOLD_S = 0.6            # keep the demo running this long past the grasp mark (the jaws finish closing)
LIFT_CM = 6.0
CARRY_PITCHES = (-92.0, -85.0, -78.0, -70.0, -62.0, -55.0, -45.0, -35.0, -25.0)   # least tilt that reaches wins
GRIP_PITCHES = (-90.0, -85.0, -80.0, -75.0, -70.0, -65.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0)
CARRY_CM_PER_S = 5.0
VERTICAL_CM_PER_S = 4.0
RELEASE_S = 1.0
DROP_CANDIDATES = [(0, 10), (0, -10), (0, 7), (0, -7), (-4, 8), (-4, -8), (-6, 0), (-3, 4), (-3, -4), (0, 0)]   # (0, 0): set it back down in place
READY = {j: 0.0 for j in JOINTS}                                     # LeRobot's zero: the calibrated mid-range pose
REST = {"shoulder_pan": 0.0, "shoulder_lift": -102.0, "elbow_flex": 97.0, "wrist_flex": 79.0, "wrist_roll": 0.0}   # folded on the table, where a run starts
GRIPPER_REST = 5.0
RETRIGGER_CM = 5.0      # auto mode: grip again once the Sesame is this far from where it was last set down
CORRECTION_FILE = "grip_correction.json"
# ahead, left, up (cm) relative to the kept correction, tried in this order after a miss; 2 cm steps on every axis
TRIALS = [(0, 0, 0), (-2, 0, 0), (2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, -2), (0, 0, 2),
          (-2, 0, -2), (2, 0, -2), (0, 2, -2), (0, -2, -2), (-2, 2, 0), (-2, -2, 0), (2, 2, 0), (2, -2, 0),
          (-2, 0, 2), (2, 0, 2), (0, 2, 2), (0, -2, 2), (-2, 2, -2), (-2, -2, -2), (2, 2, -2), (2, -2, -2), (-4, 0, 0), (4, 0, 0)]
MOVED_CM = 5.0          # the Sesame's tag must move at least this far during the carry for the grip to count
RETRY_S = 6.0           # auto mode: after a refused plan, wait this long (or a moved Sesame) before trying again


def load_correction():
    try:
        with open(CORRECTION_FILE) as f:
            d = json.load(f)
        return {"ahead": float(d.get("ahead", 0.0)), "left": float(d.get("left", 0.0)), "up": float(d.get("up", 0.0))}
    except (OSError, ValueError):
        return {"ahead": 0.0, "left": 0.0, "up": 0.0}


def save_correction(c):
    with open(CORRECTION_FILE, "w") as f:
        json.dump({**c, "note": "cm added to every grip target in the arm's base frame: ahead = +x, left = +y, up = +z"}, f, indent=1)


def measured_grip(frame_path):
    """The grip offset measured by selfcal.py, if the frame file carries one."""
    import json
    try:
        with open(frame_path) as f:
            d = json.load(f)
        return d["grip"] if d.get("measured") and d.get("grip") else None
    except (OSError, KeyError, ValueError):
        return None


def auto_demo(obs, frame, args):
    """A grasp 'demo' built from the tag: the same record layout, so plan() treats it like a recorded one.
    Poses are placed around the tag as the tracker sees it now; plan() then re-places them around the same
    pose, so the offsets are exactly the requested ones. Approach tilted, straightening for the grip."""
    tag = frame.pose_to_base(obs["robot"])
    R = rot(tag["heading"])
    centre = np.array([tag["x"], tag["y"]])
    grip = getattr(args, "measured_grip", None)
    if grip:
        # measured by selfcal: when the jaws hold the hinge, the tag sits (along, left) cm from the gripper frame
        # along the jaw axis, turned heading_offset. Invert it: jaws = tag heading - offset, gripper = tag - R(jaws) o.
        jaw = (tag["heading"] - grip["heading_offset"] + 180) % 360 - 180
        gx, gy = centre - rot(jaw) @ [grip["along"], grip["left"]]
        # both hinges: selfcal measured one of them; the other is its mirror through the tag centre.
        # take the one nearer the arm
        gx2, gy2 = centre + rot(jaw) @ [grip["along"], grip["left"]]
        if np.hypot(gx2, gy2) < np.hypot(gx, gy) - 0.5:
            gx, gy = gx2, gy2
            jaw = (jaw + 180 + 180) % 360 - 180
        tag_above_gripper = grip.get("tag_height_cm", 10.5) - (grip.get("hold_z_cm", 9.5) - 100 * TABLE_Z)
        z_grip = (obs["robot"].get("z", 10.5) - tag_above_gripper + 100 * TABLE_Z) / 100
    else:
        along = args.grip_along
        if args.hinge:                                      # two hinges, +-hinge along the heading: take the one nearer the arm
            ends = [(np.linalg.norm(centre + R @ [h, args.grip_across]), h) for h in (args.hinge, -args.hinge)]
            along += min(ends)[1]
        gx, gy = centre + R @ [along, args.grip_across]
        z_grip = (obs["robot"].get("z", 10.5) + args.grip_above_tag + 100 * TABLE_Z) / 100
        jaw = (tag["heading"] + args.jaw_angle + 180) % 360 - 180
    corr = getattr(args, "correction", None) or {"ahead": 0.0, "left": 0.0, "up": 0.0}
    gx, gy = gx + corr["ahead"], gy + corr["left"]          # the kept correction plus this attempt's trial shift
    z_grip += corr.get("up", 0.0) / 100
    # the grip pitch: straight down when the arm reaches it, else tilted forward only as far as needed
    grip_pitch = next((pp for pp in GRIP_PITCHES if ik_exact(gx / 100, gy / 100, z_grip, jaw, pp) is not None), None)
    if grip_pitch is None:
        grip_pitch = -90.0                                  # let plan() report the unreachable grip
    approach = max(grip_pitch, min(-45.0, grip_pitch + 25.0))   # a little flatter than the grip, never steeper
    def approach_steps(height):
        return [(height, approach), (height * 0.7, approach + (grip_pitch - approach) * 0.3),
                (height * 0.43, approach + (grip_pitch - approach) * 0.6), (height * 0.17, approach + (grip_pitch - approach) * 0.85)]
    # the approach from 3.5 cm above; if the arm cannot reach that high out here, come in lower
    for height in (0.035, 0.025, 0.015, 0.008, 0.0):
        if all(ik_exact(gx / 100, gy / 100, z_grip + dz, jaw, pp) is not None for dz, pp in approach_steps(height)):
            break
    steps = [(dz, pp, GRIPPER_OPEN_AUTO) for dz, pp in approach_steps(height)] + \
            [(0.0, grip_pitch, GRIPPER_OPEN_AUTO), (0.0, grip_pitch, GRIPPER_OPEN_AUTO * 0.5), (0.0, grip_pitch, 0.0), (0.0, grip_pitch, 0.0)]
    samples, keyframes, t = [], [], 0.0
    for dz, pitch, grip in steps:
        pose = {"x": gx / 100, "y": gy / 100, "z": z_grip + dz, "pitch": pitch, "jaw_yaw": jaw}
        samples.append({"t": round(t, 3), "joints": {}, "gripper": grip, "pose": pose,
                        "robot_floor": dict(obs["robot"]), "arm_floor": obs.get("arm")})
        t += 0.5
    keyframes.append({"t": samples[-2]["t"], "index": len(samples) - 2, "label": "grasp"})
    return {"name": "auto", "samples": samples, "keyframes": keyframes, "auto": True}


GRIPPER_OPEN_AUTO = 30.0


def cartesian(p0, p1, seconds, t_start, gripper):
    """Straight-line poses from p0 to p1 (dicts with x y z jaw_yaw pitch, m and deg) over seconds; the
    pitch blends from p0's to p1's, the jaw heading stays."""
    n = max(2, int(seconds * FPS))
    out = []
    for i in range(n + 1):
        f = i / n
        p = {k: p0[k] + (p1[k] - p0[k]) * f for k in ("x", "y", "z", "pitch")}
        p["jaw_yaw"] = p0["jaw_yaw"]; p["gripper"] = gripper; p["t"] = t_start + seconds * f
        out.append(p)
    return out


def ik_exact(x, y, z, jaw, pitch):
    """The pickup's IK: the jaw heading's direction is kept, so the fixed jaw is always on the same side of the tag."""
    return ik(x, y, z, jaw, pitch, exact_jaw=True)


def solvable(poses):
    return all(ik_exact(p["x"], p["y"], p["z"], p["jaw_yaw"], p["pitch"]) is not None for p in poses)


def carry_ok(pick, drop, lift_m, pitch, n=12):
    """Every pose of lift, carry and lower at this lift and tilt is reachable (sampled along each leg)."""
    up = {**pick, "z": pick["z"] + lift_m, "pitch": pitch}
    down = {**drop, "z": drop["z"] + lift_m, "pitch": pitch}
    legs = [(pick, up), (up, down), (down, drop)]
    for a, b in legs:
        for i in range(n + 1):
            f = i / n
            q = {k: a[k] + (b[k] - a[k]) * f for k in ("x", "y", "z", "pitch")}
            if ik_exact(q["x"], q["y"], q["z"], pick["jaw_yaw"], q["pitch"]) is None:
                return False
    return True


def carry_pitch(pick, drop, lift_m):
    """The least tilt (closest to the grasp pitch) at which the whole lift, carry and lower path is reachable."""
    for pitch in CARRY_PITCHES:
        if pitch < pick["pitch"] - 2:                 # never steeper than the grip itself
            continue
        if carry_ok(pick, drop, lift_m, pitch):
            return pitch
    return None


def pick_drop(pick, args):
    """The drop pose (same height and jaw angle as the pick) and the carry pitch.
    Replace this with the end-position logic later."""
    if args.drop:
        cands = [{**pick, "x": args.drop[0] / 100, "y": args.drop[1] / 100}]
    else:
        offsets = [tuple(args.drop_offset)] if args.drop_offset else DROP_CANDIDATES
        cands = [{**pick, "x": pick["x"] + dx / 100, "y": pick["y"] + dy / 100} for dx, dy in offsets]
    for lift in (args.lift, args.lift * 0.66, args.lift * 0.5, 2.0):           # a lower carry beats no carry
        for cand in cands:
            if not solvable([cand]):
                continue
            pitch = carry_pitch(pick, cand, lift / 100)
            if pitch is not None:
                if lift != args.lift:
                    print(f"  lift reduced to {lift:.1f} cm so the carry stays in reach")
                args.lift = lift
                return cand, pitch
    return None, None


def plan(demo, frame0, obs, args):
    """All phases as one trajectory [(t, joints, gripper)], or (None, reason)."""
    if demo is None or demo.get("auto"):
        demo = auto_demo(obs, frame0.adjusted_for_arm_tag(obs["arm"]), args)
    samples, grasp, release = grasp_segment(demo, args.approach, 0.0)
    samples = [s for s in samples if s["t"] <= grasp["t"] + HOLD_S]
    tag_demo_floor = tag_pose_at(grasp, demo)
    gp = demo["samples"][grasp["index"]]["pose"]
    if tag_demo_floor:
        frame_demo = frame0.adjusted_for_arm_tag(demo["samples"][grasp["index"]].get("arm_floor"))
        tag_demo = frame_demo.pose_to_base(tag_demo_floor)
    else:
        # the demo is the METHOD only: its own grasp point stands in for the tag, with the jaw direction
        # such that, placed at a real tag, the jaws sit at --jaw-angle to the tag's heading
        tag_demo = {"x": 100 * gp["x"], "y": 100 * gp["y"], "z": 0.0, "heading": gp["jaw_yaw"] - args.jaw_angle}
    frame_now = frame0.adjusted_for_arm_tag(obs["arm"])
    tag_now = frame_now.pose_to_base(obs["robot"])
    offsets = relative_offsets(samples, tag_demo)
    poses = place(offsets, tag_now)
    corr = getattr(args, "correction", None) or {"ahead": 0.0, "left": 0.0, "up": 0.0}
    for pz in poses:                                        # the kept correction, as for the tag-geometry grasp
        pz["x"] += corr["ahead"] / 100; pz["y"] += corr["left"] / 100; pz["z"] += corr.get("up", 0.0) / 100
    for lift in (PRE_APPROACH_CM, PRE_APPROACH_CM / 2, 1.0):
        pre = dict(poses[0]); pre["z"] += lift / 100; pre["t"] = poses[0]["t"] - 1.5
        if ik_exact(pre["x"], pre["y"], pre["z"], pre["jaw_yaw"], pre["pitch"]) is not None:
            poses = [pre] + poses
            break
    open_value = max(s["gripper"] for s in demo["samples"] if s["t"] <= grasp["t"])   # how far the jaws were open before the grip
    grip_end = poses[-1]
    pick = {k: grip_end[k] for k in ("x", "y", "z", "jaw_yaw", "pitch")}
    if ik_exact(pick["x"], pick["y"], pick["z"], pick["jaw_yaw"], pick["pitch"]) is None:
        dist = 100 * np.hypot(pick["x"], pick["y"])
        return None, (f"the grip point is {dist:.0f} cm from the arm's base, at ({100*pick['x']:.0f}, {100*pick['y']:.0f}, {100*pick['z']:.0f}) cm in the arm's frame, "
                      f"and no gripper tilt from straight down to 20 deg reaches it.")
    drop, pitch = pick_drop(pick, args)
    if drop is None:
        return None, "no reachable drop point near the pick, even tilted for the carry: lower --lift, or pass --drop-offset / --drop"
    closed = max(0.0, grip_end["gripper"] - args.squeeze)
    t = grip_end["t"]
    lifted = {**pick, "z": pick["z"] + args.lift / 100, "pitch": pitch}
    phases = [("grip", poses)]
    seg = cartesian(pick, lifted, args.lift / VERTICAL_CM_PER_S, t, closed); t = seg[-1]["t"]; phases.append(("lift", seg))
    drop_lifted = {**drop, "z": lifted["z"], "pitch": pitch}
    dist = 100 * np.hypot(drop["x"] - pick["x"], drop["y"] - pick["y"])
    seg = cartesian(lifted, drop_lifted, max(1.0, dist / CARRY_CM_PER_S), t, closed); t = seg[-1]["t"]; phases.append(("carry", seg))
    seg = cartesian(drop_lifted, drop, args.lift / VERTICAL_CM_PER_S, t, closed); t = seg[-1]["t"]; phases.append(("lower", seg))
    seg = cartesian(drop, drop, RELEASE_S, t, closed)
    for i, p in enumerate(seg):
        p["gripper"] = closed + (open_value - closed) * i / (len(seg) - 1)
    t = seg[-1]["t"]; phases.append(("release", seg))
    seg = cartesian(drop, drop_lifted, args.lift / VERTICAL_CM_PER_S, t, open_value); phases.append(("retract", seg))

    traj, report, counts = [], [], []
    t0 = phases[0][1][0]["t"]
    for name, ps in phases:
        tr, failed = solve(ps, grasp["t"] if name == "grip" else -1e9, None if name == "grip" else -1e9, args.squeeze if name == "grip" else 0.0)
        if name != "grip":
            tr = [(p["t"] - ps[0]["t"] + ps[0]["t"] - t0, q, g) for (_, q, g), p in zip(tr, [p for p in ps if ik_exact(p["x"], p["y"], p["z"], p["jaw_yaw"], p["pitch"]) is not None])]
        else:
            tr = [(tt + ps[0]["t"] - t0, q, g) for tt, q, g in tr]
        last = ps[-1]
        counts.append((name, len(tr), len(ps)))
        report.append(f"  {name:8s} {len(tr):3d}/{len(ps):3d} poses  ends at x={100*last['x']:5.1f} y={100*last['y']:5.1f} z={100*last['z']:5.1f} cm  pitch {last['pitch']:5.0f}  jaws {last['jaw_yaw']:6.1f}  gripper {last['gripper']:4.0f}")
        if failed:
            f = failed[0][1]
            return None, "\n".join(report) + f"\n  {name}: {len(failed)} poses unreachable or unsafe, first x={100*f['x']:.1f} y={100*f['y']:.1f} z={100*f['z']:.1f}"
        traj += tr
    info = {"tag_now": tag_now, "obs": obs, "pick": pick, "drop": drop, "report": report, "seconds": traj[-1][0], "carry_pitch": pitch, "phases": counts}
    return traj, info


def go_home(arm, grip=GRIPPER_REST):
    """Back to the folded rest pose: unfold to zero first (a clean, checked path), then fold down."""
    arm.slew(READY, grip)
    arm.slew(REST, grip)


def execute(arm, traj, speed=1.0):
    _, q0, g0 = traj[0]
    arm.slew(q0, g0)
    time.sleep(0.3)
    start = time.perf_counter(); i = 0
    while i < len(traj):
        now = (time.perf_counter() - start) * speed
        while i + 1 < len(traj) and traj[i + 1][0] <= now:
            i += 1
        t, q, g = traj[i]
        if i + 1 < len(traj):
            t2, q2, g2 = traj[i + 1]
            f = 0.0 if t2 <= t else min(1.0, (now - t) / (t2 - t))
            q = {j: q[j] + (q2[j] - q[j]) * f for j in JOINTS}; g = g + (g2 - g) * f
        arm.send(q, g)
        if i + 1 >= len(traj):
            break
        time.sleep(1 / FPS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo", nargs="?", default="auto", help="a recorded demo name, or 'auto' (default) to grip from the tag geometry")
    ap.add_argument("--grip-along", type=float, default=0.0, help="cm from the tag centre along its heading to the grip point (added to --hinge)")
    ap.add_argument("--hinge", type=float, default=0.0, help="0 (default): grip at the tag centre. N: the hinges sit N cm from the centre at either end along the heading; the nearer one to the arm is gripped")
    ap.add_argument("--grip-across", type=float, default=0.0, help="cm to the tag's left")
    ap.add_argument("--grip-above-tag", type=float, default=0.7, help="jaws close this many cm above the tag's reported height")
    ap.add_argument("--jaw-angle", type=float, default=90.0, help="jaw axis relative to the tag heading, kept exactly: 90 and -90 both close across the body but with the fixed and moving jaws swapped; 0 closes along it")
    ap.add_argument("--tracker"); ap.add_argument("--frame", default="arm_frame.json"); ap.add_argument("--port", default=default_port(), help="arm serial port (default: the USB serial device found, or $SO101_PORT)")
    ap.add_argument("--drop-offset", type=float, nargs=2, metavar=("DX", "DY"), help="cm from the pick point, base frame")
    ap.add_argument("--drop", type=float, nargs=2, metavar=("X", "Y"), help="absolute base-frame cm")
    ap.add_argument("--lift", type=float, default=LIFT_CM); ap.add_argument("--approach", type=float, default=4.0)
    ap.add_argument("--squeeze", type=float, default=8.0); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--once", action="store_true", help="run once without the key loop")
    ap.add_argument("--auto", action="store_true", help="grip whenever the Sesame is seen, no keypress; q quits")
    ap.add_argument("--no-check", action="store_true", help="do not judge the grip by whether the tag moved")
    ap.add_argument("--tag-offset", type=float, nargs=2, metavar=("AHEAD", "LEFT"), default=[-1.0, 0.0], help="the arm's base origin relative to its tag (cm). -1 0: measured on 2026-09-20, the gripper landed 1 cm past the tag with 0 0")
    ap.add_argument("--tag-turn", type=float, default=0.0, help="the arm's forward relative to its tag's up (deg)")
    ap.add_argument("--dry-run", action="store_true", help="plan only, never connect to the arm")
    args = ap.parse_args()

    args.measured_grip = measured_grip(args.frame) if args.demo == "auto" else None
    if args.measured_grip:
        g = args.measured_grip
        demo = None
        print(f"grasp from the tag with the MEASURED hinge offset: tag {g['along']:+.1f} cm along the jaw axis, {g['left']:+.1f} cm left of the jaws, turned {g['heading_offset']:+.0f} deg (selfcal)")
    elif args.demo == "auto":
        demo = None
        where = f"the nearer hinge {args.hinge:.1f} cm from the centre along the heading" if args.hinge else "the tag CENTRE"
        print(f"grasp from the tag: {where}{f' {args.grip_along:+.1f} along' if args.grip_along else ''}, {args.grip_across:+.1f} cm across, {args.grip_above_tag:+.1f} cm above the tag, jaws at {args.jaw_angle:.0f} deg")
    else:
        try:
            demo = load(args.demo)
        except FileNotFoundError:
            print(f"no demo '{args.demo}': record one with record_demo.py, or run without a name to grip from the tag geometry"); return 1
        has_tag = any(x.get("robot_floor") for x in demo["samples"])
        print(f"grasp METHOD from demo '{args.demo}'" + (" (anchored to its tag pose)" if has_tag else " (its own grasp point stands in for the tag)") + f", placed at the Sesame's tag, jaws at {args.jaw_angle:.0f} deg to its heading")
    if not os.path.exists(args.frame):
        print(f"{args.frame} not found: run calibrate_arm_frame.py first (fingertips on the Sesame's tag, 3 placements)"); return 1
    frame0 = ArmFrame.load(args.frame)
    tracker = Tracker(args.tracker); poller = Poller(tracker, period=0.3)
    arm = None if args.dry_run else Arm(args.port)

    last = {"drop_floor": None, "refused_at": 0.0, "refused_floor": None}

    def run(dry, obs=None):
        # the Sesame relative to the arm's tag, both from one camera in the same frames, medianed
        pair = tracker.observe_pair(min_samples=5, timeout=30.0)
        if pair is not None:
            obs = {"robot": pair["robot"], "arm": pair["arm"], "unit": pair["unit"], "zUp": pair["zUp"], "floor": pair["floor"], "samples": pair["samples"]}
            frame_run = frame_from_tag(pair["arm"], args.tag_offset[0], args.tag_offset[1], args.tag_turn, not pair["zUp"])
            d = np.array([pair["robot"]["x"] - pair["arm"]["x"], pair["robot"]["y"] - pair["arm"]["y"]])
            along = d @ [np.cos(np.radians(pair["arm"]["heading"])), np.sin(np.radians(pair["arm"]["heading"]))]
            left = d @ [-np.sin(np.radians(pair["arm"]["heading"])), np.cos(np.radians(pair["arm"]["heading"]))]
            print(f"\n  camera {pair['unit']}, {pair['samples']} frames with both tags: Sesame is {along:.1f} cm ahead and {left:+.1f} cm left of the arm's tag "
                  f"(arm tag heading {pair['arm']['heading']:.0f}, Sesame heading {pair['robot']['heading']:.0f})")
        else:
            obs = obs or tracker.wait_for_robot(30.0)
            if obs is None:
                print("\n  no camera reported the Sesame's tag in 30 s"); return False
            frame_run = frame0
            print("\n  no camera sees both tags at once: using the saved frame")
        traj, info = plan(demo, frame_run, obs, args)
        if traj is None:
            print("\n  REFUSED:\n" + info)
            last["refused_at"], last["refused_floor"] = time.time(), (obs["robot"]["x"], obs["robot"]["y"])
            return False
        tn = info["tag_now"]
        print(f"\n  Sesame at base ({tn['x']:.1f}, {tn['y']:.1f}) cm heading {tn['heading']:.0f} [camera {obs['unit']}]; "
              f"pick x={100*info['pick']['x']:.1f} y={100*info['pick']['y']:.1f}, drop x={100*info['drop']['x']:.1f} y={100*info['drop']['y']:.1f}, "
              f"carry tilted to pitch {info['carry_pitch']:.0f}, {info['seconds']:.1f} s")
        print("\n".join(info["report"]))
        if dry or arm is None:
            return True
        before = (obs["robot"]["x"], obs["robot"]["y"])
        print(f"  running (target correction: {args.correction['ahead']:+.1f} ahead, {args.correction['left']:+.1f} left, {args.correction['up']:+.1f} up)")
        try:
            execute(arm, traj, args.speed)
        except Exception as e:                      # a refused or failed move mid-way: still go back to zero
            print(f"  MOVE STOPPED: {type(e).__name__}: {str(e)[:200]}")
            print("  returning to the rest pose")
            try:
                go_home(arm)
            except Exception as e2:
                print(f"  could not return to rest either: {str(e2)[:120]}. Move the arm by hand to a free pose and press z.")
            return False
        print("  back to the rest pose")
        go_home(arm)
        if not args.no_check:
            after = tracker.wait_for_robot(15.0, say=None)
            if after is None:
                print("  cannot tell whether the grip worked: the tag was not seen after the carry")
            else:
                moved = np.hypot(after["robot"]["x"] - before[0], after["robot"]["y"] - before[1])
                if moved >= MOVED_CM:
                    print(f"  GRIP WORKED: the Sesame moved {moved:.1f} cm with the gripper. Keeping the correction.")
                    save_correction(args.correction)
                    args.trial_index = 0
                else:
                    print(f"  GRIP MISSED: the Sesame moved only {moved:.1f} cm.")
                    if args.auto and args.trial_index < len(TRIALS) - 1:
                        args.trial_index += 1
                        da, dl, du = TRIALS[args.trial_index]
                        base = load_correction()
                        args.correction = {"ahead": base["ahead"] + da, "left": base["left"] + dl, "up": base["up"] + du}
                        print(f"  next attempt with the target {da:+.0f} cm ahead, {dl:+.0f} cm left, {du:+.0f} cm up of the kept correction")
                        last["drop_floor"] = None           # try again right away, without waiting for the Sesame to move
                    return False
        frame_now = frame0.adjusted_for_arm_tag(obs["arm"])
        last["drop_floor"] = tuple(frame_now.to_floor([[100 * info["drop"]["x"], 100 * info["drop"]["y"]]])[0])
        print("  done; waiting for the Sesame to be seen somewhere new")
        return True

    args.correction = load_correction(); args.trial_index = 0
    if args.once or args.dry_run:
        ok = run(args.dry_run)
        if arm: arm.close(False)
        return 0 if ok else 1

    args.correction = load_correction()
    args.trial_index = 0
    print(f"grip target correction: {args.correction['ahead']:+.1f} cm ahead, {args.correction['left']:+.1f} cm left, {args.correction['up']:+.1f} cm up  (c/f/l/r/u/d nudge 1 cm, 0 clears)")
    if args.auto:
        print("AUTO: the arm grips the Sesame whenever a camera sees it in reach; a missed grip is retried with a shifted target. q = quit, p = plan, z = ready pose")
    else:
        print("space/g = pick up and move the Sesame   p = plan only   z = ready pose   o = open gripper   q = quit")
    keys = Keys()
    try:
        last_line = ""
        while True:
            key = keys.get()
            if args.auto and key is None:
                o = poller.get()
                if o is not None and arm is not None:
                    here = (o["robot"]["x"], o["robot"]["y"])
                    moved_since_drop = last["drop_floor"] is None or np.hypot(here[0] - last["drop_floor"][0], here[1] - last["drop_floor"][1]) > RETRIGGER_CM
                    refused_recently = time.time() - last["refused_at"] < RETRY_S and last["refused_floor"] is not None \
                        and np.hypot(here[0] - last["refused_floor"][0], here[1] - last["refused_floor"][1]) < 3.0
                    if moved_since_drop and not refused_recently:
                        print(f"\n  Sesame seen at floor ({here[0]:.1f}, {here[1]:.1f}): gripping")
                        run(False, tracker.observe_steady(1.0) or o)
            if key in ("c", "f", "l", "r", "u", "d", "0"):
                if key == "0":
                    args.correction = {"ahead": 0.0, "left": 0.0, "up": 0.0}
                else:
                    da, dl, du = {"c": (-1, 0, 0), "f": (1, 0, 0), "l": (0, 1, 0), "r": (0, -1, 0), "u": (0, 0, 1), "d": (0, 0, -1)}[key]
                    args.correction = {"ahead": args.correction["ahead"] + da, "left": args.correction["left"] + dl, "up": args.correction["up"] + du}
                save_correction(args.correction); args.trial_index = 0; last["drop_floor"] = None
                print(f"\n  correction now {args.correction['ahead']:+.1f} cm ahead, {args.correction['left']:+.1f} cm left, {args.correction['up']:+.1f} cm up (kept)")
            if key in (" ", "g"):
                run(False)
            elif key == "p":
                run(True)
            elif key == "z":
                try:
                    go_home(arm); print("\n  rest pose")
                except Exception as e:
                    print(f"\n  cannot reach the rest pose from here: {str(e)[:120]}")
            elif key == "o":
                q = arm.read(); arm.send({j: q[j] for j in JOINTS}, 40.0); print("\n  gripper opened")
            elif key == "q":
                break
            o = poller.get()
            line = (f"Sesame seen by camera {o['unit']} at floor ({o['robot']['x']:.0f}, {o['robot']['y']:.0f}) heading {o['robot']['heading']:.0f}   "
                    if o else "Sesame not seen by any camera   ")
            if line != last_line:
                sys.stdout.write("\r" + line); sys.stdout.flush(); last_line = line
            time.sleep(0.1)
    finally:
        keys.restore()
        if arm: arm.close(False)
        print("\ntorque left on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
