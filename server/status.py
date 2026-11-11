"""One-table summary of every run in a folder, from the server's terminal.

    python3 server/status.py [--root ~/logdir/r2dreamer/presentacion] [--window 20]

Standard library only (no venv, no GPU): it reads metrics.jsonl and status.json,
which the trainer writes as it goes, so it is safe to run while runs train.
`progress.py --logdir <run>` gives the detailed view of a single run.
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from progress import collectors, load, rolling, series  # noqa: E402


def age(path):
    """How long ago a file changed, to tell a running run from a stopped one."""
    if not path.exists():
        return "-"
    seconds = int(time.time() - path.stat().st_mtime)
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("~/logdir/r2dreamer/presentacion"))
    parser.add_argument("--window", type=int, default=20, help="training episodes averaged")
    args = parser.parse_args()
    root = args.root.expanduser()
    runs = sorted(p.parent for p in root.glob("*/metrics.jsonl"))
    if not runs:
        print(f"No hay runs con metrics.jsonl en {root}")
        return 1

    header = f"{'run':52s} {'datos':>7s} {'paso':>9s} {'meta':>9s} {'%':>4s} {'galletas':>8s} {'eval':>6s} {'pasos/s':>7s} {'últ. métrica':>12s} {'últ. guardado':>13s}"
    print(header)
    print("-" * len(header))
    for run in runs:
        rows, status, task = load(run)
        step = status["step"] if status else max((r["step"] for r in rows), default=0)
        goal = status["steps"] if status else None
        # The last train step is more recent than the last checkpoint.
        step = max([step] + [r["step"] for r in rows[-50:] if "fps/fps" in r or "train/rew" in r])
        played = collectors(rows)
        # With data from the expert or a random agent the training score is theirs, not the actor's.
        data = "actor" if not played or played == {"actor"} else "+".join(sorted(n[:3] for n in played))
        score_key = "episode/score" if data == "actor" else "episode/score_actor"
        score = [v for _, v in series(rows, score_key)[-args.window:]]
        evals = series(rows, "episode/eval_score")
        fps = [v for _, v in series(rows, "fps/fps") if v > 0][-3:]
        pct = f"{100 * step / goal:.0f}" if goal else "?"
        print(
            f"{run.name[:52]:52s} {data:>7s} {step:>9,} {goal or 0:>9,} {pct:>4s} "
            f"{(sum(score) / len(score)) if score else float('nan'):>8.2f} "
            f"{evals[-1][1] if evals else float('nan'):>6.2f} "
            f"{(sum(fps) / len(fps)) if fps else float('nan'):>7.1f} "
            f"{age(run / 'metrics.jsonl'):>12s} {age(run / 'status.json'):>13s}"
        )
    print("\n'datos' = quién juega los episodios de entrenamiento (exp = experto, ran = azar); si no es el actor, 'galletas' (del actor) queda vacío y lo que mide es 'eval'.")
    print("'últ. métrica' = hace cuánto se escribió metrics.jsonl: si pasan >15 min con el run en la cola, revisa `ts -t <id>`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
