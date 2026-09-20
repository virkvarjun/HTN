"""Attach the Sesame's tag pose to a demo recorded without the tracker, so go can move it to any position.

    sh run.sh anchor NAME        with the trackers running and the Sesame still exactly where it was recorded

Reads the Sesame's tag (and the arm's tag) from the cameras now, medianed over several frames, and writes
that pose into every sample of demos/NAME.json as if the recorder had done it. Only valid if the Sesame has
not moved since the recording.
"""
import json
import sys

from sesame_tracker import Tracker, open_camera_page


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    name = sys.argv[1]
    path = f"demos/{name}.json"
    d = json.load(open(path))
    if any(s.get("robot_floor") for s in d["samples"]):
        print(f"{path} already carries a tag pose"); return 0
    tracker = Tracker()
    print(f"reading the Sesame's tag from {tracker.host} (it must not have moved since the recording)...")
    pair = tracker.observe_pair(min_samples=5, timeout=40.0)
    obs = pair or tracker.wait_for_robot(30.0)
    if obs is None:
        print("no camera reported the Sesame's tag"); open_camera_page(tracker.host); return 1
    robot = {**{k: round(float(v), 2) for k, v in obs["robot"].items()}, "unit": obs["unit"]}
    arm = None if not obs.get("arm") else {k: round(float(v), 2) for k, v in obs["arm"].items()}
    for s in d["samples"]:
        s["robot_floor"], s["arm_floor"] = robot, arm
    d["tracker"] = {"host": tracker.host, "zUp": obs.get("zUp", True), "anchored_after": True}
    json.dump(d, open(path, "w"), indent=1)
    print(f"anchored {path}: Sesame at floor ({robot['x']}, {robot['y']}) heading {robot['heading']} [camera {obs['unit']}, {obs.get('samples', 1)} frames]"
          + (f", arm tag at ({arm['x']}, {arm['y']}) heading {arm['heading']}" if arm else ", arm tag not seen"))
    print(f"now:  sh run.sh go {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
