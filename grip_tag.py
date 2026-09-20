"""Grip the Sesame at the centre of its tag. One target, one grip, every number printed.

    sh run.sh grip [--dry-run] [--grip-z 8.5] [--jaw-angle 90] [--open 45] [--tag-offset 1 0] [--hover 5] [--lift 10] [--carry 0 10]

How the target is found:
  1. From one camera, median of several frames holding both the Sesame's tag and the arm's tag (floor cm).
  2. Sesame minus arm tag, rotated by the arm tag's heading -> the tag centre ahead/left of the arm's base.
     The base is where the arm's tag is. --tag-offset pulls the grip point back toward the base (default 1 cm).
  3. Target: x, y = tag centre; z = --grip-z (8.5 cm: the demonstrated 9.0 was a few mm high); pitch straight down;
     jaw heading = tag heading + --jaw-angle, kept exactly (the moving jaw always on the same side).
  4. Closed-form IK. If straight down cannot reach, the gripper tilts only as far as needed, in 5 deg steps.
Then: rest -> hover above the target, jaws open to --open -> straight down -> close over 1 s -> lift --lift cm
straight up from where it gripped -> carry --carry (ahead, left) -> lower -> let go -> lift away -> rest.
"""
import argparse
import sys
import time

import numpy as np

from replay_demo import Arm, FPS
from sesame_pickup import READY, REST, GRIPPER_REST, go_home
from sesame_tracker import Tracker, ensure_trackers, open_camera_page
from so101_ik import JOINTS, ik, fk, TABLE_Z
from so101_safe import default_port

OPEN_DEFAULT, CLOSED = 45.0, 0.0     # wider than the body so the jaws land around the two lips a cm or two off (60 was too wide)
PITCHES = [-90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40]


def solve(x_cm, y_cm, z_cm, jaw):
    """Joints for the target, straight down if possible. The lips are a symmetric pair, so the wrist may take
    the jaw axis or the jaw axis + 180: the exact direction first, the flipped one when that is out of reach.
    Returns (joints, pitch) or (None, None)."""
    for pitch in PITCHES:
        for exact in (True, False):
            q = ik(x_cm / 100, y_cm / 100, z_cm / 100, jaw, pitch, exact_jaw=exact)
            if q is not None:
                return q, pitch
    return None, None


def line(a, b, seconds, grip_a, grip_b=None):
    """Poses along a straight line between two joint solutions' targets are solved per step by the caller;
    here: linear joint-space interpolation between two solved poses, with the gripper blending."""
    n = max(2, int(seconds * FPS)); grip_b = grip_a if grip_b is None else grip_b
    return [({j: a[j] + (b[j] - a[j]) * i / n for j in JOINTS}, grip_a + (grip_b - grip_a) * i / n) for i in range(n + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker"); ap.add_argument("--port", default=default_port())
    ap.add_argument("--grip-z", type=float, default=8.5, help="cm above the arm's base where the jaws close (demo 9.0 was a few mm high)")
    ap.add_argument("--jaw-angle", type=float, default=90.0, help="jaw axis relative to the tag's top edge: 90 = across the tag (the lips are at its left and right edges); 0 if the tag is stuck rotated 90 deg on the body")
    ap.add_argument("--open", type=float, default=OPEN_DEFAULT, help="gripper opening before the grip (0 closed .. 100 fully open)")
    ap.add_argument("--tag-offset", type=float, nargs=2, metavar=("AHEAD", "LEFT"), default=[1.0, 0.0], help="cm to pull the grip point back toward the base / to the right of the tag centre (default: 1 cm back, the gripper landed 1 cm past the tag)")
    ap.add_argument("--hover", type=float, default=5.0); ap.add_argument("--lift", type=float, default=10.0, help="cm to lift the Sesame straight up from where it was gripped, before carrying it")
    ap.add_argument("--carry", type=float, nargs=2, metavar=("DX", "DY"), default=[0.0, 10.0], help="cm to carry the Sesame at the lifted height, ahead and left in the arm's frame, before setting it down")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tracker = Tracker(args.tracker)
    up = ensure_trackers(tracker)
    if not up:
        return 1
    print(f"camera view (cameras up: {up}): both tags, the Sesame's and the arm's, must be in one camera's view")
    open_camera_page(tracker.host, units=up)                       # the same viewer as go: Sean's page, one tab per camera
    print(f"1. reading both tags from one camera at {tracker.host}...")
    pair = tracker.observe_pair(min_samples=5, timeout=60.0)
    if pair is None:
        print("   no camera reported both the Sesame's tag and the arm's tag in 60 s: check the page"); return 1
    r, a = pair["robot"], pair["arm"]
    print(f"   camera {pair['unit']}, {pair['samples']} frames: Sesame tag at floor ({r['x']:.1f}, {r['y']:.1f}) heading {r['heading']:.1f}; "
          f"arm tag at ({a['x']:.1f}, {a['y']:.1f}) heading {a['heading']:.1f}")

    th = np.radians(a["heading"])
    d = np.array([r["x"] - a["x"], r["y"] - a["y"]])
    ahead = d @ [np.cos(th), np.sin(th)]
    left = d @ [-np.sin(th), np.cos(th)]
    x, y = ahead - args.tag_offset[0], left - args.tag_offset[1]
    tag_rel = (r["heading"] - a["heading"] + 180) % 360 - 180        # the tag's orientation in the arm's frame
    jaw = (tag_rel + args.jaw_angle + 180) % 360 - 180
    print(f"2. tag centre relative to the arm's tag: {ahead:.1f} cm ahead, {left:+.1f} cm left; Sesame heading {r['heading'] - a['heading']:+.1f} deg relative to the arm")
    print(f"3. target in the arm's frame: x={x:.1f} y={y:.1f} z={args.grip_z:.1f} cm, {np.hypot(x, y):.1f} cm from the base (tag centre pulled {args.tag_offset[0]:g} cm back, {args.tag_offset[1]:g} cm right)")
    print(f"   wrist: tag orientation {tag_rel:+.1f} deg in the arm's frame, jaw axis = tag {args.jaw_angle:+.0f} = {jaw:+.1f} deg")

    q_grip, pitch = solve(x, y, args.grip_z, jaw)
    if q_grip is None:
        print("   IK: unreachable at every tilt from straight down to 40 deg"); return 1
    q_hover, p_hover = solve(x, y, args.grip_z + args.hover, jaw)
    carry_z = args.grip_z + args.lift                          # cm above the base: --lift above the grip
    q_lift, p_lift = solve(x, y, carry_z, jaw)
    if q_hover is None or q_lift is None:
        print("   IK: the hover/lift height above the target is unreachable; lower --hover/--lift"); return 1
    # the drop point: --carry from the pick at the lifted height, then down to the grip height; if that is out of
    # reach, try the same distance the other way, then closer, then set it down where it was picked up
    drop = None
    for dx, dy in ([tuple(args.carry), (args.carry[0], -args.carry[1]), (-args.carry[1] * 0.0 - 6.0, 0.0), (0.0, 0.0)]):
        q_c, p_c = solve(x + dx, y + dy, carry_z, jaw)
        q_d, p_d = solve(x + dx, y + dy, args.grip_z, jaw)
        if q_c is not None and q_d is not None:
            drop = (dx, dy, q_c, q_d); break
    dx, dy, q_carry, q_drop = drop
    f = fk(q_grip)
    print(f"4. IK: grip pitch {pitch} deg" + (" (straight down)" if pitch == -90 else " (tilted to reach)") + f", hover pitch {p_hover}, lift pitch {p_lift}")
    print(f"   after the grip: lift {args.lift:.0f} cm up from the grip (to {carry_z - 100 * TABLE_Z:.1f} cm off the ground), carry {dx:+.0f} cm ahead / {dy:+.0f} cm left to x={x+dx:.1f} y={y+dy:.1f}, lower, let go, lift, rest")
    print(f"   joints at the grip: " + "  ".join(f"{j.split('_')[0]}={q_grip[j]:.1f}" for j in JOINTS))
    got = (f["jaw_yaw"] - tag_rel + 180) % 360 - 180
    ok = min(abs(got - args.jaw_angle), abs((got - args.jaw_angle + 180) % 360 - 180)) < 0.5   # +-180 is the same jaw axis
    print(f"   check, FK of those joints: x={100*f['x']:.1f} y={100*f['y']:.1f} z={100*f['z']:.1f} cm, jaw axis {f['jaw_yaw']:+.1f} deg = tag {got:+.1f} deg"
          + ("  MATCH" if ok else "  MISMATCH") + f"; jaws open to {args.open:.0f} before the grip")
    if args.dry_run:
        return 0

    arm = Arm(args.port)
    try:
        print("5. moving: rest -> hover")
        arm.slew(READY, args.open); arm.slew(q_hover, args.open); time.sleep(0.3)
        print("   down to the target")
        for q, g in line(q_hover, q_grip, 2.0, args.open):
            arm.send(q, g); time.sleep(1 / FPS)
        print("   closing")
        for q, g in line(q_grip, q_grip, 1.0, args.open, CLOSED):
            arm.send(q, g); time.sleep(1 / FPS)
        time.sleep(0.5)
        print(f"   lift {args.lift:.0f} cm up")
        for q, g in line(q_grip, q_lift, 2.5, CLOSED):
            arm.send(q, g); time.sleep(1 / FPS)
        time.sleep(0.3)
        print(f"   carry to x={x+dx:.1f} y={y+dy:.1f}")
        for q, g in line(q_lift, q_carry, 3.0, CLOSED):
            arm.send(q, g); time.sleep(1 / FPS)
        print("   lower")
        for q, g in line(q_carry, q_drop, 2.5, CLOSED):
            arm.send(q, g); time.sleep(1 / FPS)
        print("   let go")
        for q, g in line(q_drop, q_drop, 1.0, CLOSED, args.open):
            arm.send(q, g); time.sleep(1 / FPS)
        time.sleep(0.3)
        print("   lift away")
        for q, g in line(q_drop, q_carry, 2.0, args.open):
            arm.send(q, g); time.sleep(1 / FPS)
        after = tracker.wait_for_robot(10.0, say=None)
        if after:
            moved = np.hypot(after["robot"]["x"] - r["x"], after["robot"]["y"] - r["y"])
            print(f"   the Sesame's tag moved {moved:.1f} cm from where it was picked up: " + ("GRIP WORKED" if moved > 3.0 else "it did not come along: grip missed"))
        print("   back to rest")
        go_home(arm, GRIPPER_REST)
    finally:
        arm.close(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
