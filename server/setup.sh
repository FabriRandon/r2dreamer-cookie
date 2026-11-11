#!/usr/bin/env bash
# One-time installation on the server (Linux, Python 3.11).
#
#   bash server/setup.sh                      # torch from PyPI (CUDA 12.8 wheels)
#   TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash server/setup.sh
#                                             # for an older NVIDIA driver
#
# Creates <repo>/.venv (or $R2D_VENV), installs the repo with the cookie
# environment and matplotlib, and checks that the GPU, the cookie domain and
# torch.compile work.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${R2D_VENV:-$REPO/.venv}"
PY="${PYTHON:-python3.11}"
cd "$REPO"

command -v "$PY" >/dev/null || { echo "No hay $PY. Pídele a quien administra el servidor Python 3.11 (o define PYTHON=...)." >&2; exit 1; }
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' \
    || { echo "$PY no es Python 3.11: pyproject.toml exige >=3.11,<3.12." >&2; exit 1; }
command -v git >/dev/null || { echo "Falta git (cookie-env se instala desde GitHub)." >&2; exit 1; }

if [ ! -x "$VENV/bin/python" ]; then
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip --version >/dev/null 2>&1 || {
    echo "El venv de $VENV no tiene pip (¿falta el paquete python3.11-venv?). Bórralo con 'rm -rf $VENV' y vuelve a correr, o usa un Python de uv/conda (PYTHON=...)." >&2
    exit 1
}
python -m pip install -U pip

# torch is pinned by pyproject.toml; installing it first from another index
# makes pip keep that build when it resolves the repo's dependencies.
if [ -n "${TORCH_INDEX:-}" ]; then
    python -m pip install "torch==2.8.0" --index-url "$TORCH_INDEX"
fi
python -m pip install -e ".[cookie,analysis]"

echo
echo "=== Revisión del entorno"
python server/check_env.py
