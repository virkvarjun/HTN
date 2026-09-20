"""Closed-form inverse kinematics for the SO-101 arm (LeRobot SOFollower joint order).

Safety
- SAFE_LIMITS is the range the arm may be commanded to; ik() never returns a pose
  outside it or with a link closer to the table than MIN_LINK_Z. check_pose()
  explains why a pose is rejected and assert_safe() raises UnsafePoseError.
  Use so101_safe.send() to command the robot so nothing past the limit is sent.

Frames and conventions
- Positions are metres in base_link. Angles in and out are degrees.
- shoulder_pan rotates about -z, so azimuth = -pan.
- approach_pitch is the pitch of the gripper_frame_link z axis (the approach
  direction): 0 = horizontal forward, -90 = straight down (top-down grasp).
  In the URDF, lift + elbow + wrist_flex = -approach_pitch.
- jaw_yaw_deg is the table-plane heading of the gripper_frame_link x axis, the
  direction the jaws open and close. Jaws are symmetric, so of the two rolls
  that give that heading the one furthest from a wrist_roll limit is used.

Geometry (from so101_new_calib.urdf, verified against placo FK to 0.005 mm)
The lift/elbow/wrist_flex chain lives in a plane 18.28 mm beside the pan
axis; the wrist_roll origin hops back to 0.18 mm off the pan plane. Only the
in-plane components matter here, so R0 and L3 below are in-plane lengths and
are shorter than the 3D distances between the same frames.
"""
import numpy as np

P0 = np.array([0.0388353, 0.0, 0.0624])   # pan axis origin in base_link
R0, Z0 = 0.0303992, 0.0542               # shoulder_lift pivot: radial and height from P0 (in-plane)
L1, A1 = 0.1160000, np.radians(76.03225)  # shoulder_lift -> elbow_flex, and its angle at q=0
L2, A2 = 0.1350002, np.radians(2.207492)  # elbow_flex -> wrist_flex, and its angle at q=0
L3 = 0.0610999                            # wrist_flex -> wrist_roll origin, along the roll axis
LA = 0.0981274                            # wrist_roll origin -> gripper_frame, along the roll axis
LAT_CHAIN = -0.0182779                    # lateral offset of the lift/elbow/wrist_flex chain from the pan plane
LAT_R = -0.0001768                        # lateral offset of the roll axis from the pan plane
# gripper_frame offset perpendicular to the roll axis at roll=0: (lateral, in-plane) components.
# With roll: lateral = N0*cos(roll) + V0*sin(roll); in-plane = -N0*sin(roll) + V0*cos(roll).
N0, V0 = 0.0001676, -0.0079006
JAW_PHASE = np.radians(2.7896)            # jaw axis is rotated this much about the roll axis at roll=0
TABLE_Z = -0.02                           # table surface in base_link z (m); targets below it are rejected

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
URDF_LIMITS = {"shoulder_pan": (-110.0, 110.0), "shoulder_lift": (-100.0, 100.0),
               "elbow_flex": (-96.8, 96.8), "wrist_flex": (-95.0, 95.0), "wrist_roll": (-157.0, 163.0)}

# Safe range the arm may be COMMANDED to, in degrees. ik() never returns a pose outside this box and
# check_pose() rejects one. Widen a value only after moving the arm there by hand and seeing that nothing
# binds. Where each value comes from:
# - shoulder_pan: hit a mechanical stop at about +79 deg with the arm extended on 2026-09-19, so +-75.
# - shoulder_lift low end, elbow_flex high end: the folded rest pose the arm sits in (lift -104, elbow 97.5,
#   hand-guided demo of 2026-09-19); the other ends are 5 deg inside the URDF limits.
# - wrist_flex: a hand-guided top-down grasp at 9.4 cm height reached 110 deg without binding
#   (2026-09-19), so +-112; the URDF says 95. A top-down approach above 7 cm needs more than 90 deg anyway.
# - wrist_roll: 5 deg inside the URDF limits.
SAFE_LIMITS = {"shoulder_pan": (-75.0, 75.0), "shoulder_lift": (-106.0, 95.0),
               "elbow_flex": (-91.8, 98.0), "wrist_flex": (-112.0, 112.0), "wrist_roll": (-152.0, 158.0)}
LIMITS = SAFE_LIMITS
# Minimum z in base_link (m) for the elbow, wrist_flex pivot and wrist_roll origin: the motor bodies sit
# around those points, so this is the crash margin. The gripper frame is the fingertips and may come down
# to the table, which is TABLE_Z below base_link's origin (the base plate): with the arm resting on the
# table in its folded pose the fingertips read z = -1.8 cm (hand-guided demo, 2026-09-19).
# The gripper frame floor is 3.5 cm below base_link: in the folded rest pose the fingertips rest on the table and
# read -1.8 to -2.4 cm (seen 2026-09-19/20), and a move out of rest must not be refused at its first step.
MIN_LINK_Z = {"elbow_flex": 0.04, "wrist_flex": 0.04, "wrist_roll": 0.04, "gripper_frame": -0.035}
UNSAFE_MSG = "Refusing to command it: driving the arm past this position risks breaking the robot."


class UnsafePoseError(ValueError):
    """Raised by assert_safe() for a pose outside SAFE_LIMITS or too close to the table."""


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _roll_for(jaw_yaw, azimuth, pitch, exact=False):
    """wrist_roll (rad) that points the jaw axis at heading jaw_yaw, given the pan azimuth and pitch.
    exact=True: honour the heading's direction (which jaw is on which side), flipping by 180 only if the
    direct roll is outside the limits. Otherwise the candidate furthest from a limit is taken."""
    h = jaw_yaw - azimuth
    s = np.sin(pitch)
    if abs(s) < 1e-9:
        return None                                       # horizontal approach: heading not set by roll
    direct = np.arctan2(-abs(s) * np.sin(h), np.sign(s) * np.cos(h)) + JAW_PHASE
    lo, hi = np.radians(LIMITS["wrist_roll"])
    # The jaws are symmetric, so direct and direct+180 both work; take the one furthest from a limit.
    cands = [np.radians(_wrap(np.degrees(c))) for c in (direct, direct + np.pi)]
    if exact:
        return next((c for c in cands if lo <= c <= hi), None)
    cands = [c for c in cands if lo <= c <= hi]
    return max(cands, key=lambda c: min(c - lo, hi - c)) if cands else None


def link_points(joints):
    """Base-frame positions (m) of the elbow, wrist_flex pivot, wrist_roll origin and gripper frame
    for joint angles in degrees. Same planar model as ik(), matches the URDF FK to 0.005 mm."""
    pan, lift, elbow, wf, roll = (np.radians(joints[j]) for j in JOINTS)
    az = -pan
    er = np.array([np.cos(az), np.sin(az), 0.0]); el = np.array([-np.sin(az), np.cos(az), 0.0]); ez = np.array([0, 0, 1.0])
    pitch = -(lift + elbow + wf)
    u = np.array([np.cos(pitch), np.sin(pitch)]); v = np.array([-np.sin(pitch), np.cos(pitch)])
    e = np.array([R0, Z0]) + L1 * np.array([np.cos(A1 - lift), np.sin(A1 - lift)])
    w = e + L2 * np.array([np.cos(A2 - lift - elbow), np.sin(A2 - lift - elbow)])
    r = w + L3 * u
    g = r + LA * u + (-N0 * np.sin(roll) + V0 * np.cos(roll)) * v
    lat_g = LAT_R + N0 * np.cos(roll) + V0 * np.sin(roll)
    pts = {"elbow_flex": (e, LAT_CHAIN), "wrist_flex": (w, LAT_CHAIN), "wrist_roll": (r, LAT_R), "gripper_frame": (g, lat_g)}
    return {k: P0 + rz[0] * er + lat * el + rz[1] * ez for k, (rz, lat) in pts.items()}


def fk(joints):
    """Forward kinematics for joint angles in degrees: gripper_frame_link position (m, base_link), the
    approach pitch (deg, -90 = pointing down) and the jaw heading in the table plane (deg). The inverse of
    ik(): fk(ik(x, y, z, yaw, pitch)) gives back x, y, z, pitch and yaw up to the jaws' 180-degree symmetry."""
    pts = link_points(joints)
    x, y, z = pts["gripper_frame"]
    pan, lift, elbow, wf, roll = (np.radians(joints[j]) for j in JOINTS)
    pitch = -(lift + elbow + wf)
    az = -pan
    s = np.sin(pitch)
    # horizontal projection of the jaw axis: cos(roll - phase) * sin(pitch) along the radial, -sin(roll - phase) lateral
    a = roll - JAW_PHASE
    heading = az + np.arctan2(-np.sin(a), np.cos(a) * s) if abs(s) > 1e-9 else float("nan")
    return {"x": float(x), "y": float(y), "z": float(z), "pitch": float(_wrap(np.degrees(pitch))), "jaw_yaw": float(_wrap(np.degrees(heading)))}


def check_pose(joints):
    """None if the pose (dict of the 5 arm joints, deg) is safe to command, else a message saying why not."""
    for j in JOINTS:
        lo, hi = SAFE_LIMITS[j]
        if not lo <= joints[j] <= hi:
            return (f"UNSAFE POSE: {j}={joints[j]:+.1f} deg is outside the safe range {lo:+.1f}..{hi:+.1f} deg. "
                    + UNSAFE_MSG)
    for name, pt in link_points(joints).items():
        if pt[2] < MIN_LINK_Z[name] - 1e-6:
            return (f"UNSAFE POSE: {name} would be {pt[2]*100:.1f} cm above the table "
                    f"(minimum {MIN_LINK_Z[name]*100:.0f} cm). " + UNSAFE_MSG)
    return None


def assert_safe(joints):
    """Raise UnsafePoseError with the reason if the pose must not be commanded."""
    msg = check_pose(joints)
    if msg:
        raise UnsafePoseError(msg)
    return joints


def clamp_pose(joints):
    """Clip each arm joint into SAFE_LIMITS. Used to recover from a pose the arm is already in
    (e.g. gravity-drooped past a software limit) without commanding anything further outside it."""
    return {j: min(max(float(joints[j]), SAFE_LIMITS[j][0]), SAFE_LIMITS[j][1]) for j in JOINTS}


def _lerp(a, b, n):
    n = max(1, int(n))
    return [{j: a[j] + (b[j] - a[j]) * i / n for j in JOINTS} for i in range(1, n + 1)]


def _safe_suffix(poses):
    """Drop leading poses that fail check_pose. Remaining poses must all be safe."""
    i = next((k for k, q in enumerate(poses) if check_pose(q) is None), None)
    if i is None or any(check_pose(q) for q in poses[i:]):
        return None
    return poses[i:]


def recovery_waypoints(start, target, n_from_span):
    """Joint-space path from start to target whose every pose passes check_pose.

    start is clamped into SAFE_LIMITS. An already-below-table droop (typical limp hang
    with wrist_flex high) is recovered by tucking wrist_flex to the target first, and
    by not commanding the unsafe prefix the arm is already in. n_from_span(span_deg)
    returns the number of steps for a segment.
    """
    start = clamp_pose(start)
    target = {j: float(target[j]) for j in JOINTS}

    def segment(a, b):
        span = max(abs(b[j] - a[j]) for j in JOINTS)
        return _lerp(a, b, n_from_span(span))

    direct = _safe_suffix(segment(start, target))
    if direct is not None:
        return direct

    vias = [{**start, "wrist_flex": target["wrist_flex"]}]
    vias.append({**vias[0], "shoulder_lift": target["shoulder_lift"]})
    for via in vias:
        if check_pose(via) is not None:
            continue
        head = _safe_suffix(segment(start, via))
        tail = segment(via, target)
        if head is not None and not any(check_pose(q) for q in tail):
            return head + tail
    raise UnsafePoseError(
        "UNSAFE POSE: no joint-space recovery path stays above the table. " + UNSAFE_MSG
    )


def ik(x, y, z, jaw_yaw_deg, approach_pitch_deg=-90.0, exact_jaw=False):
    """Joint angles (deg) placing gripper_frame_link at (x, y, z), or None if unreachable or unsafe.
    exact_jaw=True keeps the jaw heading's direction (which jaw is on which side of the target)."""
    if z < TABLE_Z:
        return None
    dx, dy, dz = x - P0[0], y - P0[1], z - P0[2]
    rho = np.hypot(dx, dy)
    yaw, pitch = np.radians(jaw_yaw_deg), np.radians(approach_pitch_deg)
    heading = np.arctan2(dy, dx)

    # The gripper frame sits off the pan plane by an amount that depends on wrist_roll,
    # and wrist_roll depends on the pan azimuth. Fixed-point iterate; converges in 2-3 steps.
    azimuth = heading
    for _ in range(8):
        roll = _roll_for(yaw, azimuth, pitch, exact_jaw)
        if roll is None:
            return None
        lateral = LAT_R + N0 * np.cos(roll) + V0 * np.sin(roll)
        if abs(lateral) >= rho:
            return None
        azimuth = heading - np.arcsin(lateral / rho)

    # In-plane target for the wrist_flex pivot: walk back along the roll axis and the
    # in-plane part of the perpendicular offset.
    r_g = np.sqrt(rho**2 - lateral**2)
    u = np.array([np.cos(pitch), np.sin(pitch)])          # roll/approach axis in (radial, z)
    v = np.array([-np.sin(pitch), np.cos(pitch)])         # in-plane perpendicular
    inplane = -N0 * np.sin(roll) + V0 * np.cos(roll)
    w = np.array([r_g, dz]) - (L3 + LA) * u - inplane * v

    # 2R planar IK from the shoulder_lift pivot, elbow-up branch.
    d = w - np.array([R0, Z0])
    dist = np.linalg.norm(d)
    if dist > L1 + L2 or dist < abs(L1 - L2):
        return None
    beta = np.arctan2(d[1], d[0])
    alpha = np.arccos(np.clip((dist**2 + L1**2 - L2**2) / (2 * dist * L1), -1.0, 1.0))
    gamma = np.arccos(np.clip((L1**2 + L2**2 - dist**2) / (2 * L1 * L2), -1.0, 1.0))
    psi1 = beta + alpha                                    # link 1 steeper than the chord: elbow up
    lift = A1 - psi1
    elbow = A2 - A1 + np.pi - gamma
    wrist_flex = -pitch - lift - elbow

    sol = dict(zip(JOINTS, map(_wrap, np.degrees([-azimuth, lift, elbow, wrist_flex, roll]))))
    return None if check_pose(sol) else sol


def reachable(x, y, z, jaw_yaw_deg, approach_pitch_deg=-90.0):
    return ik(x, y, z, jaw_yaw_deg, approach_pitch_deg) is not None
