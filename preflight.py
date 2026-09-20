"""What is ready for the arm demo and what is missing, with the fix for each. Run: sh run.sh check"""
import glob
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def row(ok, what, fix=""):
    print(f"  {'ok  ' if ok else 'MISSING'}  {what}" + ("" if ok or not fix else f"\n           -> {fix}"))
    return ok


def main():
    print(f"preflight on {platform.system()} {platform.machine()}, python {sys.version.split()[0]}")
    ready = True
    try:
        import lerobot, scipy, placo, cv2  # noqa: F401
        ready &= row(True, "python environment (LeRobot, scipy, placo, opencv)")
    except ImportError as e:
        ready &= row(False, f"python environment ({e.name} missing)", "sh run.sh setup")
    from so101_safe import default_port
    port = default_port()
    if port:
        readable = os.access(port, os.R_OK | os.W_OK)
        fix = "" if readable else (f"sudo usermod -aG dialout $USER, then log out and in" if platform.system() == "Linux" else f"check permissions on {port}")
        row(readable, f"arm on {port}" + ("" if readable else " (no permission)"), fix)
    else:
        row(False, "arm (no USB serial device found)", "plug the SO-101 in, or set SO101_PORT=/dev/... if it is on an unusual name")
    ready &= row(os.path.exists("calibration/so_follower/follower.json"), "arm motor calibration (calibration/so_follower/follower.json)", "it ships with the repo; if it is gone, lerobot-calibrate --robot.type=so_follower")
    has_pi = os.path.isfile("pi/common.sh") and os.path.isdir("pi/tracker")
    row(has_pi, "Sean's Pi tracker files (pi/)", "sh run.sh trackers fetches them from origin/devel/sean")
    from sesame_tracker import Tracker
    t = Tracker()
    seen = {u: t.state(u) for u in t.units}
    up = [u for u, s in seen.items() if s]
    robot = [u for u, s in seen.items() if s and s.get("calibrated") and s.get("robot") and "x" in s["robot"]]
    row(bool(up), f"tracker reachable at {t.host} (cameras {up or 'none'})", "join the robot's WiFi, put the Pi's address in pi/host if its name does not resolve, then sh run.sh trackers")
    if up:
        row(bool(robot), f"a camera sees the Sesame's tag (cameras {robot or 'none'})", "put the Sesame on the board in view, check the corner tags are visible")
    row(os.path.exists("arm_frame.json"), "floor-to-arm calibration (arm_frame.json)", "sh run.sh calibrate   (fingertips on the Sesame's tag at 3 placements)")
    demos = sorted(os.path.basename(p)[:-5] for p in glob.glob("demos/*.json"))
    tracked = []
    for d in demos:
        try:
            import json
            with open(f"demos/{d}.json") as f:
                dd = json.load(f)
            g = next((k for k in dd["keyframes"] if k["label"] == "grasp"), None)
            if g and dd["samples"][g["index"]].get("robot_floor"):
                tracked.append(d)
        except Exception:
            pass
    row(bool(tracked), f"a grasp demo recorded with the tracker ({', '.join(tracked) or 'none'}; untracked demos: {', '.join(d for d in demos if d not in tracked) or 'none'})",
        "sh run.sh record grip2   (with the tracker running; g when the jaws close on the Sesame, q to save)")
    print()
    if tracked and os.path.exists("arm_frame.json") and robot and port:
        print(f"everything is in place:  sh run.sh pickup {tracked[-1]}   (p to see the plan, space to go)")
    else:
        print("fix the MISSING lines above in order, then:  sh run.sh pickup NAME")
    return 0


if __name__ == "__main__":
    sys.exit(main())
