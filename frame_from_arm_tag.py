"""Floor-to-arm frame from the arm base tag (id 5), with no hands-on calibration.

    sh run.sh frame [--offset AHEAD LEFT] [--turn DEG] [--tracker HOST]

The camera places tag 5 on the floor beside the arm. Measure once with a ruler where the arm's base centre
is relative to that tag, and the transform follows:
  --offset AHEAD LEFT   cm from the tag's centre to the centre of the arm's base plate, along the tag's
                        "up" direction (its top edge, the heading the tracker reports) and to its left.
                        Example: the tag is taped in front of the base, centred, 3 cm from the plate, its top
                        edge pointing away from the arm; the plate is 6 cm deep -> base centre is 6 cm BEHIND
                        the tag centre: --offset -6 0
  --turn DEG            the arm's forward direction relative to the tag's up direction, counter-clockwise
                        seen from above. 0 when the tag's top edge points the way the arm faces.
Writes arm_frame.json. The hands-on calibration (calibrate_arm_frame.py) is more accurate when the offset is
uncertain; this one is exact only as far as the ruler is.
"""
import argparse
import sys
import time

import numpy as np

from arm_frame import ArmFrame, rot
from sesame_tracker import Tracker, open_camera_page


def frame_from_tag(arm_tag, ahead_cm, left_cm, turn_deg, mirrored=False):
    """p_base = R p_floor + t from the tag's floor pose and the base's offset in the tag's frame."""
    c = np.array([arm_tag["x"], arm_tag["y"]])
    R_tag = rot(-arm_tag["heading"])                     # floor -> tag frame (x along the tag's up)
    if mirrored:                                         # left-handed floor frame: mirror y before rotating
        R_tag = R_tag @ np.diag([1.0, -1.0])
    R_base = rot(-turn_deg)                              # tag -> base
    R = R_base @ R_tag
    t = -R_base @ (R_tag @ c + np.array([ahead_cm, left_cm]))
    return ArmFrame(R, t, mirrored, dict(arm_tag), residual_cm=None, units=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offset", type=float, nargs=2, metavar=("AHEAD", "LEFT"), default=[0.0, 0.0],
                    help="base_link origin relative to the tag centre; 0 0 = the base is where the tag is")
    ap.add_argument("--turn", type=float, default=0.0)
    ap.add_argument("--tracker")
    ap.add_argument("--out", default="arm_frame.json")
    args = ap.parse_args()
    tracker = Tracker(args.tracker)
    print(f"reading the arm base tag (id 5) from {tracker.host}...")
    got = tracker.wait_for_arm_tag(60.0)
    if got is None:
        print("no calibrated camera reported the arm base tag (id 5) in 60 s. Is it flat, in view, and is the camera calibrated (height shown on the page)?")
        open_camera_page(tracker.host)
        return 1
    arm = {k: got[k] for k in ("x", "y", "z", "heading")}
    mirrored, unit = got["mirrored"], got["unit"]
    frame = frame_from_tag(arm, args.offset[0], args.offset[1], args.turn, mirrored)
    frame.units = [unit]
    frame.save(args.out)
    print(f"arm tag at floor ({arm['x']:.1f}, {arm['y']:.1f}) heading {arm['heading']:.0f} [camera {unit}, {got['sightings']} sightings]; base {args.offset[0]:+.1f} ahead, {args.offset[1]:+.1f} left, turned {args.turn:.0f} deg")
    print(frame.describe())
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
