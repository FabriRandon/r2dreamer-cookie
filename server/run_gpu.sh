#!/usr/bin/env bash
# Wrapper to run anything of this repo under task spooler (the server's `ts`):
#
#   ts -G 1 bash server/run_gpu.sh python -u train.py env=cookie ...
#
# It uses the repo's virtualenv, sets the environment variables the cookie
# domain needs on a machine without a display or sound card, and then *execs*
# the command. The exec matters: `ts -k` sends SIGTERM to the job's process
# group, and train.py only saves the agent and the replay buffer before quitting
# when that signal reaches Python itself and not a bash waiting for it.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${R2D_VENV:-$REPO/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "No existe $VENV/bin/python. Corre primero: bash server/setup.sh" >&2
    exit 1
fi
if [ "$#" -eq 0 ]; then
    echo "Uso: bash server/run_gpu.sh python -u train.py <opciones de hydra>" >&2
    exit 2
fi

# `ts -G 1` hands the job the free GPU it picked through CUDA_VISIBLE_DEVICES.
# R2D_GPU (launch.py --gpu-id N) overrides that choice, e.g. to use the second
# GPU. ts then does not know the job sits there, so only use it when you must.
if [ -n "${R2D_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$R2D_GPU"
fi

export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
cd "$REPO"

# Environment workers and the main process each start torch; without a cap every
# one of them opens as many threads as the server has cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# pygame (loaded by minigrid / cookie-env / moviepy) must not look for a screen
# or a sound card. envs/__init__.py also sets these; kept here for watch.py etc.
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
# torch.compile (model.compile=True) keeps its generated kernels here, so a
# second run with the same model does not compile again.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"

echo "=== $(date '+%F %T') host=$(hostname) pid=$$ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<sin definir>}"
echo "=== python: $(command -v python)"
echo "=== cmd: $*"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
fi
exec "$@"
