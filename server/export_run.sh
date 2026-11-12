#!/usr/bin/env bash
# Pack the results of ONE run into a single .tar.gz to take to another computer.
# Run it where the run is (the server). It never modifies the run's folder.
#
#   bash server/export_run.sh ~/logdir/r2dreamer/presentacion/cookie_partial_hall5_view5_s0_expert
#
# What goes in (the replay buffer, which weighs gigabytes, never does unless asked):
#   default           everything to see the run AND to probe its world model:
#                     TensorBoard events, metrics.jsonl, status.json, console.log,
#                     .hydra/config.yaml, latest.pt (the learned agent) and VERSION.txt
#   --sin-eventos     leave out the TensorBoard events (the biggest file): ~130 MB,
#                     enough for explorar_modelo.py / watch.py
#   --solo-tensorboard   events and metrics only, without latest.pt
#   --con-replay      also the replay buffer (diagnostics that read it; many GB)
#   --out DIR         where the archive is written (default ~/exports)
set -eo pipefail

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }

EVENTS=1; MODEL=1; REPLAY=0; OUT="$HOME/exports"; RUN=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --sin-eventos) EVENTS=0 ;;
        --solo-tensorboard) MODEL=0 ;;
        --con-replay) REPLAY=1 ;;
        --out) OUT="${2:?--out necesita una carpeta}"; shift ;;
        -h|--help) usage 0 ;;
        -*) echo "Opción desconocida: $1" >&2; usage ;;
        *) RUN="$1" ;;
    esac
    shift
done
[ -n "$RUN" ] || usage
[ "$EVENTS$MODEL" != "00" ] || { echo "No hay nada que exportar con --sin-eventos y --solo-tensorboard a la vez." >&2; exit 2; }

RUN="$(cd "$RUN" 2>/dev/null && pwd)" || { echo "No existe la carpeta del run." >&2; exit 1; }
NAME="$(basename "$RUN")"
PARENT="$(dirname "$RUN")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$RUN/metrics.jsonl" ] || { echo "$RUN no parece un run (falta metrics.jsonl)." >&2; exit 1; }
if [ "$MODEL" = 1 ]; then
    [ -f "$RUN/latest.pt" ] || { echo "Falta $RUN/latest.pt: el run no guardó el agente todavía." >&2; exit 1; }
    [ -f "$RUN/.hydra/config.yaml" ] || { echo "Falta $RUN/.hydra/config.yaml." >&2; exit 1; }
fi

# What the archive holds, relative to the run's parent folder.
FILES=("$NAME/metrics.jsonl" "$NAME/.hydra")
for f in status.json console.log; do [ -e "$RUN/$f" ] && FILES+=("$NAME/$f"); done
if [ "$EVENTS" = 1 ]; then
    shopt -s nullglob
    events=("$RUN"/events.out.tfevents.*)
    shopt -u nullglob
    [ "${#events[@]}" -gt 0 ] || echo "AVISO: no hay archivos events.out.tfevents.* en el run; TensorBoard no tendrá nada." >&2
    for f in "${events[@]}"; do FILES+=("$NAME/$(basename "$f")"); done
fi
[ "$MODEL" = 1 ] && FILES+=("$NAME/latest.pt")
[ "$REPLAY" = 1 ] && [ -d "$RUN/replay" ] && FILES+=("$NAME/replay")

# VERSION.txt: the code version the checkpoint needs to be loaded with.
STAGE="$(mktemp -d)"
ARCHIVE=""
# Also removes a half-written archive if the script stops before the final rename.
trap 'rm -rf "$STAGE"; [ -z "$ARCHIVE" ] || rm -f "$ARCHIVE.tmp"' EXIT
mkdir -p "$STAGE/$NAME"
{
    echo "run: $NAME"
    echo "exportado: $(date '+%F %T %z') en $(hostname)"
    if git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
        echo "commit del repo AL EXPORTAR: $(git -C "$REPO" rev-parse HEAD) (rama $(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
        echo "(puede ser posterior al commit con el que se entrenó el run: si hiciste git pull después de lanzarlo, usa el que lanzaste)"
        if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
            echo "AVISO: el repo tenía cambios sin commit al exportar"
        fi
    else
        echo "AVISO: no hay git en $REPO, así que no se puede registrar el commit del código."
    fi
    echo "status.json: $(cat "$RUN/status.json" 2>/dev/null || echo 'no hay')"
    echo "Para cargar latest.pt usa el mismo código con el que se entrenó (modelo: networks.py, rssm.py, dreamer.py)."
} > "$STAGE/$NAME/VERSION.txt"

# Warn about a run that is still training: its files change while they are copied.
if [ -f "$RUN/status.json" ] && command -v python3 >/dev/null; then
    python3 - "$RUN/status.json" <<'PY' || true
import json, sys
s = json.load(open(sys.argv[1]))
if s["step"] < s["steps"]:
    print(f"AVISO: el run va en el paso {s['step']:,} de {s['steps']:,} (no terminó); se exporta el último guardado.", file=sys.stderr)
PY
fi

mkdir -p "$OUT"
MODE=completo; [ "$EVENTS" = 0 ] && MODE=modelo; [ "$MODEL" = 0 ] && MODE=tensorboard; [ "$REPLAY" = 1 ] && MODE="$MODE+replay"
ARCHIVE="$OUT/${NAME}_${MODE}.tar.gz"
echo "Empaquetando en $ARCHIVE ..."
# tar exits with 1 when a file changed while it was read, which is what happens to the
# events and metrics of a run that is still training; that is fine, only >1 is an error.
rc=0
tar -czf "$ARCHIVE.tmp" -C "$PARENT" "${FILES[@]}" -C "$STAGE" "$NAME/VERSION.txt" || rc=$?
if [ "$rc" -gt 1 ]; then
    echo "tar falló (código $rc)." >&2
    exit "$rc"
fi
[ "$rc" = 1 ] && echo "AVISO: algún archivo cambió mientras se copiaba (el run seguía escribiendo). El final del archivo de TensorBoard puede estar cortado; es inofensivo."
mv "$ARCHIVE.tmp" "$ARCHIVE"
( cd "$OUT" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256" )

echo
ls -lh "$ARCHIVE" | awk '{print "Listo: " $9 "  (" $5 ")"}'
echo "Contenido:"; tar -tzf "$ARCHIVE" | awk -F/ 'NF<=2 || /\.hydra/' | sed 's/^/  /' | head -20
echo
echo "Para bajarlo a tu PC (desde tu PC):"
echo "  scp ${USER}@$(hostname -f 2>/dev/null || hostname):$ARCHIVE ."
