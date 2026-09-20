"""A stand-in for the Pi tracker, for testing on the laptop: serves /state.json for units 3 and 4 on
localhost with a fixed robot and arm-tag pose in the floor frame.

    .venv/bin/python fake_tracker.py --robot 30 20 45 --arm 70 8 180 [--zup 1] [--noise 0.2]
then point the clients at it:  --tracker localhost
"""
import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def serve(robot=(30.0, 20.0, 45.0), arm=(70.0, 8.0, 180.0), zup=True, noise=0.1, units=(3, 4), host="127.0.0.1"):
    """Start the fake tracker on daemon threads and return the servers (call .shutdown() on each to stop)."""
    t0 = time.time()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if not self.path.startswith("/state.json"):
                self.send_response(404); self.end_headers(); return
            n = lambda: random.gauss(0, noise)
            from sesame_tracker import floor_to_pixel, ARM_TAG_HEIGHT_CM
            camera = {"f": 1693.0, "cx": 1152.0, "cy": 648.0, "rvec": [3.14159, 0, 0], "tvec": [-31.5, 31.5, 115]}
            rx, ry, ax, ay = robot[0] + n(), robot[1] + n(), arm[0] + n(), arm[1] + n()
            state = {"t": int(1000 * (time.time() - t0)), "calibrated": True, "frame": [2304, 1296], "floor": [63, 63],
                     "zUp": bool(zup), "learned": 4, "cameraHeight": 115, "floorMarkers": 4 if self.server.unit == 3 else 3, "fps": 15,
                     "markers": [0, 1, 2, 3, 4, 5],
                     "robot": {"x": rx, "y": ry, "z": 10.5, "heading": robot[2] + 3 * n(), "px": list(floor_to_pixel(camera, rx, ry, 10.5))},
                     # the real tracker reads tag 5 near floor level from its size; its px is where the tag really is
                     "arm": {"x": ax, "y": ay, "z": -2.0, "heading": arm[2] + 3 * n(), "px": list(floor_to_pixel(camera, ax, ay, ARM_TAG_HEIGHT_CM))},
                     "camera": camera}
            body = json.dumps(state).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)

    servers = []
    for unit in units:
        s = HTTPServer((host, 8000 + unit), H); s.unit = unit
        threading.Thread(target=s.serve_forever, daemon=True).start(); servers.append(s)
    return servers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", type=float, nargs=3, metavar=("X", "Y", "HEADING"), default=[30.0, 20.0, 45.0])
    ap.add_argument("--arm", type=float, nargs=3, metavar=("X", "Y", "HEADING"), default=[70.0, 8.0, 180.0])
    ap.add_argument("--zup", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.1, help="cm of jitter on the poses")
    ap.add_argument("--units", type=int, nargs="+", default=[3, 4])
    args = ap.parse_args()
    serve(args.robot, args.arm, bool(args.zup), args.noise, args.units)
    print(f"fake tracker: units {args.units} on ports {[8000 + u for u in args.units]}, robot {args.robot}, arm {args.arm}, zUp {bool(args.zup)}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
