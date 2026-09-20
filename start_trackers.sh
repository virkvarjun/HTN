#!/bin/sh
# Starts Sean's tracker on BOTH Pi cameras (units 3 and 4) at once. pi/start-tracker.sh kills any running
# tracker by name before starting one, so starting the second camera with it stops the first; this script
# builds once and launches both in one ssh session instead. Same arguments as pi/start-tracker.sh:
#   sh start_trackers.sh [floor width cm] [floor height cm] [lens code | -] [robot marker height cm]
#   defaults: 63 63, the saved lens codes, 10.5
# States: http://<pi>:8003/state.json and :8004/state.json. Ctrl-C stops both.
W=${1:-63}; H=${2:-63}; LENS=${3:--}; ROBOT_HEIGHT=${4:-10.5}
cd "$(dirname "$0")" || exit 1
[ -f pi/common.sh ] && [ -d pi/tracker ] || { echo "pi/ (Sean's tracker and Pi scripts) is not in this checkout: it lives on origin/devel/sean. Merge or check out that branch first."; exit 1; }
. pi/common.sh
# common.sh looks for the host file next to itself only when run from pi/; sourced from here, read it explicitly.
[ -f pi/host ] && PI_HOST=$(cat pi/host) && PI=qnxuser@$PI_HOST
# a hotspot hands the Pi a new address now and then; accept its host key at a new address without a prompt
SSH_OPTS="$SSH_OPTS -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
scp -q $SSH_OPTS -r pi/tracker $PI: || exit 1
ssh -t $SSH_OPTS $PI "[ -w /dev/i2c6 ] && [ -w /dev/i2c4 ] || { echo 'lens access was reset by a reboot, fixing it with sudo:'; sudo chmod 666 /dev/i2c4 /dev/i2c6; }; \
cd tracker && slay -f tracker >/dev/null 2>&1; \
clang++ -std=c++17 -O2 tracker.cpp -o tracker \$(pkg-config --cflags --libs opencv4) -lcamapi -lsocket && \
(./tracker 3 $W $H $LENS $ROBOT_HEIGHT 2>&1 | sed 's/^/[cam3] /' & ./tracker 4 $W $H $LENS $ROBOT_HEIGHT 2>&1 | sed 's/^/[cam4] /'; wait)" 2>&1 | tee pi/trackers-live.log
