# HTN

## Arm: grip the Sesame anywhere

One command. Plug the arm in, join the robot's WiFi, and:

```
sh run.sh grip           # the demo: to the centre of the Sesame's tag, close, lift 10 cm, carry left, release, rest
sh run.sh go             # the full pipeline: trackers, frame, correction loop; grips again after every miss
```

`grip` (`grip_tag.py`) is the one used at the venue. It opens Sean's camera page, starts the trackers if none
answers, reads the Sesame's tag and the arm's tag from one camera, prints the target and the joints (with a
forward-kinematics check), then moves: rest, hover, down to 8.5 cm, close, lift 10 cm straight up, carry 10 cm to
the left, lower, let go, lift away, rest. Tuning flags: `--grip-z`, `--open`, `--jaw-angle`,
`--carry AHEAD LEFT`, `--tag-offset`; `--dry-run` prints the plan without moving.

It sets itself up on first use (installs uv and the Python environment if needed, about a minute). The steps
it runs for you, also available one at a time:

```
sh run.sh check          # what is ready and what is missing, with the fix for each
sh run.sh test           # the whole pick-and-place chain offline: no arm, no Pi
sh run.sh trackers       # both Pi cameras (fetches Sean's Pi files from origin/devel/sean if this checkout lacks them)
sh run.sh calibrate      # fingertips on the Sesame's tag at 3 placements
sh run.sh record grip2   # guide the grasp by hand: g when the jaws close, q to save
sh run.sh pickup grip2   # p = plan, space = find the Sesame, grip, lift, carry, set down, release
```

`sh run.sh` alone lists every command. The arm's motor calibration ships in `calibration/`, the serial port
is found automatically (or set `SO101_PORT`), the Pi's address comes from `pi/host` when its name does not
resolve. `docs/INVENTORY.md` describes everything else in the repo; `nav/README.md` the quadruped's vision
navigation.
