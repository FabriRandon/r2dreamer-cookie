"""Summarize how a run of the cookie domain is going, from its logdir alone.

Reads only what the trainer writes as it goes (metrics.jsonl and status.json),
so it works while the run is still training, from another session, without a
GPU or the agent's dependencies:

    python3 progress.py --logdir ./logdir/run --out progress.png

Prints where the run is and how its latest episodes compare with a random
agent. With --out it also draws the curves that show progress before the agent
gets any cookie: button presses, cells visited, steps in the button's room,
the reward the world model predicts and the actor's entropy.
"""

import argparse
import json
import math
import pathlib
import re

# Per-episode averages of a uniformly random agent with the 3 actions of
# env.press_on_move, over 60 episodes of 2088 steps, measured with
# envs/cookie.py. Keyed by env.task.
RANDOM = {
    "cookie_full": {"score": 0.02, "presses": 7.1, "cells": 51.0, "button_room": 448.0},
}
# Step around which the JAX DreamerV3 run of the same task started to get
# cookies. By 700k steps it got about 21 per episode.
TAKEOFF = {"cookie_full": 620_000}

EPISODE_KEYS = {
    "score": "galletas por partida",
    "presses": "veces que aprieta el botón",
    "cells": "casillas distintas visitadas",
    "button_room": "pasos en el cuarto del botón",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", required=True, type=pathlib.Path, help="run folder")
    parser.add_argument("--out", type=pathlib.Path, help="image to draw the curves into (.png)")
    parser.add_argument("--window", type=int, default=20, help="episodes averaged in each point of the curves")
    return parser.parse_args()


def load(logdir):
    rows = []
    metrics = logdir / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            # The file can end mid-line while the run is writing it.
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    status = logdir / "status.json"
    status = json.loads(status.read_text()) if status.exists() else None
    # Read the task without a YAML parser, which the notebooks' Python may lack.
    config = logdir / ".hydra" / "config.yaml"
    task = None
    if config.exists():
        match = re.search(r"^\s+task:\s*(\S+)", config.read_text(), re.M)
        task = match.group(1) if match else None
    return rows, status, task


def series(rows, key):
    return [(row["step"], row[key]) for row in rows if key in row]


def rolling(points, window):
    """Average of the last `window` points at each point."""
    out, total = [], 0.0
    for i, (step, value) in enumerate(points):
        total += value
        if i >= window:
            total -= points[i - window][1]
        out.append((step, total / min(i + 1, window)))
    return out


def summarize(logdir, rows, status, task, window):
    print(f"Run: {logdir.name}" + (f" ({task})" if task else ""))
    if status is None and not rows:
        print("Todavía no hay nada guardado.")
    elif status is None:
        print(f"Va en el paso {max(row['step'] for row in rows):,} según sus métricas (run sin status.json).")
    else:
        print(
            f"Paso {status['step']:,} de {status['steps']:,} ({100 * status['step'] / status['steps']:.0f} %), "
            f"guardado el {status['saved_at']}."
        )
        if status.get("buffer_step") is not None:
            print(f"Buffer en disco: del paso {status['buffer_step']:,} ({status['buffer_count']:,} transiciones).")
    step = status["step"] if status else max((row["step"] for row in rows), default=0)

    random_ref = RANDOM.get(task, {})
    latest = {}
    print(f"\nÚltimas {window} partidas de entrenamiento" + (" (entre paréntesis, un agente al azar):" if random_ref else ":"))
    for key, label in EPISODE_KEYS.items():
        points = series(rows, f"episode/{key}")
        if not points:
            continue
        values = [value for _, value in points[-window:]]
        latest[key] = sum(values) / len(values)
        reference = f"  ({random_ref[key]:g})" if key in random_ref else ""
        print(f"  {label:30s} {latest[key]:8.2f}{reference}")
    evals = series(rows, "episode/eval_score")
    if evals:
        print(f"Última evaluación (paso {evals[-1][0]:,}): {evals[-1][1]:.2f} galletas por partida.")

    # Rough reading of the numbers, from a single run of the JAX study.
    takeoff = TAKEOFF.get(task)
    score = latest.get("score", 0.0)
    last_eval = evals[-1][1] if evals else 0.0
    print()
    if score > 3 * random_ref.get("score", 0.0) + 0.1 or last_eval > 0:
        print("Ya consigue galletas por su cuenta: el caso base está aprendiendo.")
    elif takeoff and step < takeoff:
        print(f"Es normal que todavía no consiga galletas: el run de referencia empezó cerca del paso {takeoff:,}.")
        if step >= 200_000 and "cells" in latest and latest["cells"] < 0.8 * random_ref.get("cells", 0):
            print("Ojo: visita menos casillas que un agente al azar; está explorando poco.")
    elif takeoff and step >= takeoff + 180_000:
        print("A esta altura el run de referencia ya sacaba ~20 galletas por partida. Si sigue en 0, conviene cortar.")
    return takeoff, random_ref


def plot(rows, task, window, takeoff, random_ref, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    panels = [
        ("score", "Galletas por partida"),
        ("presses", "Veces que aprieta el botón"),
        ("cells", "Casillas distintas visitadas"),
        ("button_room", "Pasos en el cuarto del botón"),
        ("train/rew", "Recompensa que predice el modelo"),
        ("train/action_entropy", "Entropía del actor"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    thousands = FuncFormatter(lambda x, _: f"{x / 1000:.0f}k")
    for ax, (key, title) in zip(axes.flat, panels):
        ax.set_title(title)
        ax.xaxis.set_major_formatter(thousands)
        ax.grid(alpha=0.3)
        if key.startswith("train/"):
            points = series(rows, key)
        else:
            points = rolling(series(rows, f"episode/{key}"), window)
        if points:
            steps, values = zip(*points)
            label = f"entrenamiento (promedio de {window} partidas)" if not key.startswith("train/") else None
            ax.plot(steps, values, color="tab:blue", label=label)
        if key == "score":
            evals = series(rows, "episode/eval_score")
            if evals:
                steps, values = zip(*evals)
                ax.plot(steps, values, "o", color="tab:orange", markersize=4, label="evaluación (sin azar)")
        if key in random_ref:
            ax.axhline(random_ref[key], color="gray", linestyle="--", label="agente al azar")
        if key == "train/rew" and points and min(points, key=lambda p: p[1])[1] > 0:
            ax.set_yscale("log")
        if key == "train/action_entropy":
            ax.axhline(math.log(3), color="gray", linestyle="--", label="azar puro (3 acciones)")
        if takeoff:
            ax.axvline(takeoff, color="tab:green", linestyle=":", label="referencia empieza a aprender")
        if not points and key != "score":
            ax.text(0.5, 0.5, "sin datos", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{task or ''}: progreso por pasos del ambiente")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    print(f"\nGráficos en {out}")


def main():
    args = parse_args()
    rows, status, task = load(args.logdir)
    takeoff, random_ref = summarize(args.logdir, rows, status, task, args.window)
    if args.out:
        plot(rows, task, args.window, takeoff, random_ref, args.out)


if __name__ == "__main__":
    main()
