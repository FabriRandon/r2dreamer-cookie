#!/usr/bin/env bash
# Run on YOUR computer: copy the light files of the server's runs to analyze
# them locally (progress.py, compare_runs.py, TensorBoard). Never copies the
# replay buffers or the checkpoints (latest.pt), which weigh gigabytes.
#
#   R2D_SSH=usuario@servidor bash server/sync_logs.sh
#   R2D_SSH=usuario@servidor bash server/sync_logs.sh --light        # only json/yaml/log
#   R2D_SSH=usuario@servidor R2D_REMOTE='~/logdir/r2dreamer/presentacion' R2D_LOCAL=./logdir_server bash server/sync_logs.sh
#
# Run it as many times as you like: rsync only transfers what changed.
set -eo pipefail

SSH="${R2D_SSH:?Define R2D_SSH=usuario@servidor (o un alias de ~/.ssh/config)}"
REMOTE="${R2D_REMOTE:-~/logdir/r2dreamer/presentacion}"
LOCAL="${R2D_LOCAL:-./logdir_server}"
mkdir -p "$LOCAL"

filters=(
    --exclude='replay/' --exclude='replay.tmp/' --exclude='replay.old/'
    --exclude='latest.pt' --exclude='*.tmp'
)
if [ "${1:-}" = "--light" ]; then
    # metrics, status, config and console log only (no TensorBoard events / videos).
    filters+=(--include='*/' --include='metrics.jsonl' --include='status.json' --include='console.log'
              --include='config.yaml' --include='*.mp4' --exclude='*')
fi

# The trailing slash copies the folder's content; --prune-empty-dirs skips folders left empty.
rsync -avz --partial --prune-empty-dirs --info=progress2 "${filters[@]}" "$SSH:$REMOTE/" "$LOCAL/"
echo
echo "Listo en $LOCAL. Ahora:"
echo "  python3 progress.py --logdir $LOCAL/<run> --out progreso.png"
echo "  python3 compare_runs.py --root $LOCAL --out $LOCAL/comparacion"
