"""Live view of the Pi cameras: what the tracker sees, with its detections drawn, side by side.

    sh run.sh view                 window with both cameras (q to quit, s to save a snapshot)
    sh run.sh view --once out.png  one snapshot to a file, no window

Each camera's /annotated.jpg is the tracker's own frame with the markers it detected drawn on it, so what
you see is exactly what the tracker sees. Under each frame: whether the tracker is calibrated (it needs
floor markers 1-4 learned), which marker ids it sees, and the Sesame's pose if it has one. When a tracker
does not answer, the panel says so. The Pi's address comes from pi/host (Sean's convention) or qnxpi78.local.
"""
import argparse
import json
import sys
import time
import urllib.request

import numpy as np
import cv2

from sesame_tracker import default_host

W = 960


def fetch(host, unit):
    """(image, state) for one camera, or (None, reason)."""
    base = f"http://{host}:{8000 + unit}"
    try:
        with urllib.request.urlopen(f"{base}/state.json", timeout=2) as r:
            state = json.loads(r.read())
    except Exception as e:
        return None, f"no tracker on port {8000 + unit} ({type(e).__name__})"
    try:
        with urllib.request.urlopen(f"{base}/annotated.jpg?w={W}", timeout=5) as r:
            img = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        try:
            with urllib.request.urlopen(f"{base}/frame.jpg?w={W}", timeout=5) as r:
                img = cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            return None, f"tracker up, no frame ({type(e).__name__})"
    return img, state


def panel(unit, img, state):
    if img is None:
        p = np.zeros((W * 9 // 16 + 90, W, 3), np.uint8)
        cv2.putText(p, f"camera {unit}: {state}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return p
    if img.shape[1] != W:
        img = cv2.resize(img, (W, img.shape[0] * W // img.shape[1]))
    bar = np.zeros((90, W, 3), np.uint8)
    robot = state.get("robot")
    seen = state.get("markers", [])
    lines = [f"camera {unit}   calibrated: {state.get('calibrated')}   floor markers learned {state.get('learned')}, in view {state.get('floorMarkers')}   fps {state.get('fps')}",
             f"markers seen: {seen}   " + ("Sesame (id 0) SEEN" if 0 in seen else "Sesame (id 0) NOT SEEN") + ("   arm base (id 5) seen" if 5 in seen else ""),
             (f"Sesame at ({robot['x']:.1f}, {robot['y']:.1f}) cm heading {robot['heading']:.0f}" if robot and "x" in robot
              else "Sesame position: none" + (" (tag seen but the tracker is not calibrated: it needs floor markers 1-4)" if robot else ""))]
    for i, text in enumerate(lines):
        colour = (0, 255, 0) if (i == 1 and 0 in seen) or (i == 2 and robot and "x" in robot) else (255, 255, 255)
        cv2.putText(bar, text, (12, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def compose(host, units):
    panels = []
    for u in units:
        img, state = fetch(host, u)
        panels.append(panel(u, img, state))
    h = max(p.shape[0] for p in panels)
    panels = [np.pad(p, ((0, h - p.shape[0]), (0, 0), (0, 0))) for p in panels]
    return np.hstack(panels)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=None)
    ap.add_argument("--units", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--once", metavar="PNG", help="save one composite image and exit")
    args = ap.parse_args()
    host = args.host or default_host()
    print(f"cameras at {host}, units {args.units}")
    if args.once:
        cv2.imwrite(args.once, compose(host, args.units)); print(f"wrote {args.once}"); return 0
    print("q = quit, s = save a snapshot")
    while True:
        img = compose(host, args.units)
        cv2.imshow("cameras", img)
        k = cv2.waitKey(400) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            name = f"cameras-{int(time.time())}.png"; cv2.imwrite(name, img); print("saved", name)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
