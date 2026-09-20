#!/bin/sh
# The one command for the arm demo. Sets itself up on first use.
#
#   sh run.sh go                 THE command: trackers, frame, then the arm grips the Sesame whenever a camera sees it
#                                (--manual: wait for space before each grip; --now: grip once and exit)
#   sh run.sh check              what is ready and what is missing, with the fix for each
#   sh run.sh test               the whole pick-and-place chain offline (no arm, no Pi)
#   sh run.sh trackers           start both Pi cameras' trackers (fetches Sean's Pi files if this checkout lacks them)
#   sh run.sh view               window with both cameras and what the tracker detects (q quits, s saves)
#   sh run.sh page               Sean's live web page for a camera, in the browser (sh run.sh page 4 for the second)
#   sh run.sh calibrate          fingertips on the Sesame's tag at 3 placements -> arm_frame.json (hands-on, most accurate)
#   sh run.sh frame --offset A L the same transform from the arm base tag (id 5) and a ruler: no hands-on step
#   sh run.sh selfcal            measured frame AND hinge offset: the arm holds the Sesame by its hinge and moves it (what go uses)
#   sh run.sh record NAME        guide the grasp by hand with the tracker running -> demos/NAME.json
#   sh run.sh anchor NAME        give a demo recorded without the tracker its tag pose (Sesame unmoved since)
#   sh run.sh pickup NAME        p = plan, space = find the Sesame, grip, lift, carry, set down, release
#   sh run.sh grip               the demo: to the centre of the Sesame's tag, close, lift 10 cm, carry left, lower, release, rest; every number printed
#   sh run.sh replay NAME        play a demo back where it was recorded
#   sh run.sh arm                ping the motors;  sh run.sh arm --halfway  moves to the midpoint and back
#   sh run.sh setup              (re)build the Python environment
# Anything after the subcommand is passed on (for example: sh run.sh pickup grip2 --drop-offset 0 10).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
cmd=${1:-help}; [ $# -gt 0 ] && shift

ensure_env() {
  if [ ! -x "$PY" ] || ! "$PY" -c "import lerobot, scipy, placo" 2>/dev/null; then
    echo "setting up the Python environment first (about a minute)"
    sh setup.sh
  fi
}

ensure_pi() {
  if [ ! -f pi/common.sh ] || [ ! -d pi/tracker ]; then
    echo "this checkout has no pi/ (Sean's tracker and Pi scripts). Fetching them from origin/devel/sean into the working tree."
    echo "They are his code: do not commit them from here."
    git fetch -q origin devel/sean
    git restore --source=origin/devel/sean -- pi
  fi
}

case "$cmd" in
  setup)     sh setup.sh ;;
  go)        ensure_env; "$PY" arm_go.py "$@" ;;
  check)     ensure_env; "$PY" preflight.py "$@" ;;
  test)      ensure_env; "$PY" test_pickup.py "$@" && "$PY" -m pytest -q test_so101_ik.py "$@" ;;
  trackers)  ensure_pi; sh start_trackers.sh "$@" ;;
  view)      ensure_env; "$PY" camera_view.py "$@" ;;
  page)      ensure_pi; HOST=$(cat pi/host 2>/dev/null || echo qnxpi78.local); UNIT=${1:-3}
             (cd pi/client && python3 -m http.server 5500 --bind 127.0.0.1 >/dev/null 2>&1 &)
             sleep 1; URL="http://localhost:5500/demo.html?host=$HOST&unit=$UNIT"; echo "$URL"; open "$URL" 2>/dev/null || xdg-open "$URL" ;;
  calibrate) ensure_env; "$PY" calibrate_arm_frame.py "$@" ;;
  frame)     ensure_env; "$PY" frame_from_arm_tag.py "$@" ;;
  selfcal)   ensure_env; "$PY" selfcal.py "$@" ;;
  record)    ensure_env; "$PY" record_demo.py "$@" ;;
  anchor)    ensure_env; "$PY" anchor_demo.py "$@" ;;
  pickup)    ensure_env; "$PY" sesame_pickup.py "$@" ;;
  grip)      ensure_env; "$PY" grip_tag.py "$@" ;;
  grasp)     ensure_env; "$PY" grasp_robot.py "$@" ;;
  replay)    ensure_env; "$PY" replay_demo.py "$@" ;;
  arm)       ensure_env; "$PY" check_arm.py "$@" ;;
  *)         sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
