#!/bin/sh
# One-time setup on a fresh clone: the Python environment for the arm and the vision code.
#   sh setup.sh
# Needs uv (https://docs.astral.sh/uv/). Makes .venv (Python 3.12), checks out LeRobot at the commit this was
# built against into ./lerobot (gitignored) and installs it with the Feetech and kinematics extras, plus
# scipy, matplotlib and pytest. Afterwards:  .venv/bin/python test_pickup.py
set -e
cd "$(dirname "$0")"
LEROBOT_COMMIT=5aa74557f84c54d4b458f8b9643c5aa2982acfed
if ! command -v uv >/dev/null; then
  echo "installing uv (the Python environment tool) into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || { echo "uv did not install; see https://docs.astral.sh/uv/"; exit 1; }
fi
[ -d .venv ] || uv venv --python 3.12 .venv
if [ ! -d lerobot/.git ]; then
  git clone -q https://github.com/huggingface/lerobot.git lerobot
fi
git -C lerobot fetch -q origin "$LEROBOT_COMMIT" 2>/dev/null || true
git -C lerobot checkout -q "$LEROBOT_COMMIT"
uv pip install --python .venv/bin/python -q -e "./lerobot[feetech,kinematics]" scipy matplotlib pytest
.venv/bin/python -c "import lerobot, cv2, scipy, placo; print('ok: lerobot', lerobot.__version__, 'opencv', cv2.__version__)"
echo "done. Try:  .venv/bin/python test_pickup.py"
