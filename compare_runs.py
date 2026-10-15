"""Compare saved cookie runs on CPU, including incomplete training logs.

Only metrics.jsonl, status.json and .hydra/config.yaml are read. Checkpoints
and replay buffers are never opened. Run files are never modified.
"""

import argparse
import csv
import json
import math
import pathlib
import re
import textwrap
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yaml

from progress import random_reference, rolling


@dataclass
class Run:
    path: pathlib.Path
    rows: list
    config: dict
    status: dict
    warnings: list = field(default_factory=list)

    @property
    def task(self):
        return self.config.get("env", {}).get("task") or self.path.name.split("_s")[0]

    @property
    def view(self):
        match = re.search(r"_view(\d+)$", self.task)
        return f"{match[1]}×{match[1]}" if match else ("Completa" if "full" in self.task else "?")

    @property
    def label(self):
        seed = self.config.get("seed")
        sampler = number(self.config.get("buffer", {}).get("reward_slices"))
        entropy = number(self.config.get("model", {}).get("act_entropy"))
        label = "Mapa completo (referencia)" if self.view == "Completa" else f"Vista local {self.view}"
        if sampler:
            label = f"{self.view} + repaso de éxitos"
            if sampler != 1:
                label += f" ({sampler:g} secuencias)"
        if entropy is not None and not math.isclose(entropy, 0.0003):
            label += " + más incentivo a explorar" if entropy > 0.0003 else " + menos incentivo a explorar"
        if seed not in (0, None):
            label += f" · semilla {seed}"
        return label

    @property
    def description(self):
        sampler = number(self.config.get("buffer", {}).get("reward_slices"))
        entropy = number(self.config.get("model", {}).get("act_entropy"))
        view = "Observa el mapa completo." if self.view == "Completa" else f"Observa una ventana local de {self.view} casillas."
        if sampler is None:
            replay = "Muestreo no identificado."
        elif sampler == 0:
            replay = "Todas las secuencias de entrenamiento se eligen al azar."
        else:
            action = "se incluye" if sampler == 1 else "se incluyen"
            unit = "secuencia" if sampler == 1 else "secuencias"
            replay = f"En cada actualización {action} {sampler:g} {unit} con recompensa, si ya existen éxitos."
        exploration = ""
        if entropy is not None and not math.isclose(entropy, 0.0003):
            exploration = f" El incentivo a mantener acciones variadas es {entropy / 0.0003:g} veces el original."
        return f"{view} {replay}{exploration}"

    def points(self, key):
        return [(row["step"], row[key]) for row in self.rows if key in row]


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def read_object(path, loader, warnings):
    try:
        result = loader(path.read_text()) if path.exists() else {}
        if isinstance(result, dict):
            return result
        warnings.append(f"{path.name}: contenido inesperado")
    except (OSError, ValueError, yaml.YAMLError):
        warnings.append(f"{path.name}: aún no se pudo leer; vuelve a actualizar")
    return {}


def load_run(path):
    """Keep each metric's latest value per step and tolerate partial writes."""
    path = pathlib.Path(path)
    warnings, by_step = [], {}
    status = read_object(path / "status.json", json.loads, warnings)
    config = read_object(path / ".hydra/config.yaml", yaml.safe_load, warnings)
    bad, rollbacks, previous = 0, 0, None
    try:
        lines = (path / "metrics.jsonl").read_text().splitlines()
    except OSError:
        lines = []
        warnings.append("metrics.jsonl: aún no se pudo leer")
    for line in lines:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            bad += 1
            continue
        step = number(row.get("step")) if isinstance(row, dict) else None
        if step is None or step < 0 or not step.is_integer():
            bad += 1
            continue
        step = int(step)
        if previous is not None and step < previous:
            # A resume may append from an older checkpoint. Discard the
            # superseded tail rather than present it as current progress.
            by_step = {s: values for s, values in by_step.items() if s <= step}
            rollbacks += 1
        previous = step
        values = by_step.setdefault(step, {"step": step})
        for key, value in row.items():
            value = number(value)
            if key != "step" and value is not None:
                values[key] = value
    if bad:
        warnings.append(f"{bad} líneas incompletas o inválidas ignoradas")
    if rollbacks:
        warnings.append(f"{rollbacks} reanudaciones con retroceso; se usa el tramo más reciente")
    return Run(path, [by_step[s] for s in sorted(by_step)], config, status, warnings)


def discover(root, extras=()):
    # Immediate children only: never walk the large replay-buffer trees.
    paths = {p.parent for p in pathlib.Path(root).glob("*/metrics.jsonl")}
    paths.update(pathlib.Path(p) for p in extras if (pathlib.Path(p) / "metrics.jsonl").exists())
    return [load_run(p) for p in sorted(paths)]


def find_video(run_path, report_dir=None, seed=0):
    """Find a saved episode without opening a checkpoint or replay buffer."""
    run_path = pathlib.Path(run_path)
    folders = [run_path]
    if report_dir is not None:
        folders.append(pathlib.Path(report_dir) / "videos" / run_path.name)
    pattern = f"agente_episodio{seed}.mp4" if seed is not None else "agente_episodio*.mp4"
    candidates = []
    for folder in folders:
        for path in folder.glob(pattern):
            if not re.fullmatch(r"agente_episodio\d+\.mp4", path.name):
                continue
            try:
                info = path.stat()
                if path.is_file() and info.st_size > 0:
                    candidates.append((info.st_mtime_ns, str(path), path))
            except OSError:
                pass  # Drive may still be synchronizing the video.
    return max(candidates)[2] if candidates else None


def common_evaluations(runs):
    available = [(run, dict(run.points("episode/eval_score"))) for run in runs]
    available = [(run, points) for run, points in available if points]
    if not available:
        return None, []
    common = set.intersection(*(set(points) for _, points in available))
    if not common:
        return None, []
    step = max(common)
    return step, [{"Experimento": run.label, "Carpeta": run.path.name,
                   "Paso común": step, "Galletas en evaluación": points[step]}
                  for run, points in available]


def summarize(run, window, last_evals=3):
    metrics_step = max((r["step"] for r in run.rows), default=None)
    checkpoint = number(run.status.get("step"))
    target = number(run.status.get("steps")) or number(run.config.get("env", {}).get("steps"))
    evaluations = run.points("episode/eval_score")
    training = run.points("episode/score")
    recent = evaluations[-last_evals:]
    complete = target is not None and checkpoint is not None and checkpoint >= target
    result = {
        "Experimento": run.label, "Qué cambia": run.description, "Carpeta": run.path.name, "Tarea": run.task,
        "Vista": run.view, "Semilla": run.config.get("seed"),
        "Sampler": run.config.get("buffer", {}).get("reward_slices"),
        "Coeficiente entropía": run.config.get("model", {}).get("act_entropy"),
        "Partidas por evaluación": run.config.get("env", {}).get("eval_episode_num"),
        "Pasos objetivo": target, "Último paso en métricas": metrics_step,
        "Último checkpoint": checkpoint,
        "Estado guardado": "Checkpoint al objetivo" if complete else "Parcial / final sin confirmar",
        "Evaluaciones": len(evaluations),
        "Paso última evaluación": evaluations[-1][0] if evaluations else None,
        "Última evaluación": evaluations[-1][1] if evaluations else None,
        "Evaluaciones tramo final": len(recent),
        "Paso inicio tramo final": recent[0][0] if recent else None,
        "Media tramo final": sum(v for _, v in recent) / len(recent) if recent else None,
        "Mínimo tramo final": min((v for _, v in recent), default=None),
        "Máximo tramo final": max((v for _, v in recent), default=None),
        "Mejor evaluación": max((v for _, v in evaluations), default=None),
        "Paso primera evaluación con galletas": next((s for s, v in evaluations if v > 0), None),
        "Partidas de entrenamiento": len(training),
        "Partidas con galletas": sum(v > 0 for _, v in training),
        "Promedio entrenamiento reciente": sum(v for _, v in training[-window:]) / min(window, len(training))
        if training else None,
        "Actualización checkpoint": run.status.get("saved_at"),
        "Avisos": "; ".join(run.warnings), "Ruta": str(run.path),
    }
    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(runs, out, window=20, last_evals=3):
    """Export a snapshot. Missing evaluations remain missing, never zero."""
    if window < 1:
        raise ValueError("La ventana debe ser al menos 1")
    if last_evals < 1:
        raise ValueError("El tramo final debe incluir al menos una evaluación")
    if not runs:
        raise ValueError("No se encontraron métricas. Revisa Drive y las rutas de la sección 3.")
    # Present the reference first, then the original views, then interventions.
    def order(run):
        sampler = number(run.config.get("buffer", {}).get("reward_slices")) or 0
        entropy = number(run.config.get("model", {}).get("act_entropy")) or 0.0003
        category = 0 if run.view == "Completa" else (2 if sampler or not math.isclose(entropy, 0.0003) else 1)
        size = int(run.view.split("×")[0]) if "×" in run.view else 0
        return category, size, sampler, entropy, run.path.name

    runs = sorted(runs, key=order)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    out = pathlib.Path(out)
    # Protect run data even if a caller accidentally selects its folder.
    for run in runs:
        if out.resolve() == run.path.resolve() or run.path.resolve() in out.resolve().parents:
            raise ValueError("Elige una carpeta de salida fuera de las carpetas de entrenamiento")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = [summarize(run, window, last_evals) for run in runs]
    write_csv(out / "resumen.csv", summary, list(summary[0]))
    common_step, common = common_evaluations(runs)
    common_fields = ["Experimento", "Carpeta", "Paso común", "Galletas en evaluación"]
    write_csv(out / "evaluacion_paso_comun.csv", common, common_fields)
    curves = [{"Carpeta": run.path.name, "Experimento": run.label, "Paso": row["step"],
               "Métrica": key, "Valor": value}
              for run in runs for row in run.rows for key, value in row.items()
              if key.startswith(("episode/", "train/"))]
    write_csv(out / "curvas.csv", curves, ["Carpeta", "Experimento", "Paso", "Métrica", "Valor"])
    def case_color(run, index):
        sampler = number(run.config.get("buffer", {}).get("reward_slices"))
        entropy = number(run.config.get("model", {}).get("act_entropy"))
        if entropy is not None and entropy > 0.0003:
            return "#D55E00"
        if sampler:
            return "#7B3294"
        return {"Completa": "#666666", "3×3": "#0072B2", "5×5": "#E69F00", "7×7": "#009E73"}.get(
            run.view, plt.get_cmap("tab10")(index % 10))

    colors = {run.path: case_color(run, i) for i, run in enumerate(runs)}
    artifacts = [out / name for name in ["resumen.csv", "evaluacion_paso_comun.csv", "curvas.csv"]]
    images = []

    def save(fig, name, caption="", show=True):
        footer = (caption + "\n" if caption else "") + f"Lectura UTC: {stamp}. Una corrida por caso; las curvas terminan donde hay datos."
        fig.text(0.01, 0.01, footer, fontsize=9, linespacing=1.5)
        fig.tight_layout(rect=(0, 0.04 + 0.025 * footer.count("\n"), 1, 1))
        for extension in ("png", "pdf"):
            path = out / f"{name}.{extension}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
            if extension == "png" and show:
                images.append(path)
        plt.close(fig)

    def format_axis(ax, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel("Pasos del ambiente")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        ax.grid(alpha=0.25)
        ax.set_xlim(left=0)

    def draw(ax, selected, key, smooth=False, labeler=None):
        count = 0
        for run in selected:
            points = run.points(key)
            if smooth:
                points = rolling(points, window)
            if not points:
                continue
            x, y = zip(*points)
            label = labeler(run) if labeler else run.label
            ax.plot(x, y, color=colors[run.path], label=label,
                    marker="o" if "eval_" in key else None,
                    markersize=3, linewidth=1.5, linestyle="--" if run.view == "Completa" else "-")
            count += 1
        if not count:
            ax.text(0.5, 0.5, "Sin datos guardados para esta métrica", ha="center", transform=ax.transAxes)
        return count

    eval_counts = {r.config.get("env", {}).get("eval_episode_num") for r in runs}
    point_note = (f"Cada punto es la media de {next(iter(eval_counts))} partidas."
                  if len(eval_counts) == 1 and None not in eval_counts else
                  "Cada punto es una evaluación; la cantidad de partidas por caso figura en la tabla.")
    seeds = {r.config.get("seed") for r in runs}
    if len(seeds) == 1 and None not in seeds:
        point_note += f" Semilla de entrenamiento: {next(iter(seeds))}."
    original = [r for r in runs if number(r.config.get("buffer", {}).get("reward_slices")) == 0
                and number(r.config.get("model", {}).get("act_entropy")) is not None
                and math.isclose(number(r.config["model"]["act_entropy"]), 0.0003)]
    for selected, name, title, caption, show in [
        (runs, "evaluacion_todos", "Comparación general de los experimentos", point_note, False),
        (original, "evaluacion_vistas", "Efecto del tamaño de la observación · configuración original",
         "Vista local: ventana de casillas que recibe el agente. El mapa completo se incluye como referencia.\n"
         "Todos estos casos usan muestreo al azar y el incentivo original a explorar. " + point_note, True),
        ([r for r in runs if r.view == "5×5"], "evaluacion_5x5", "Cambios en el aprendizaje con vista local de 5×5",
         "Repaso de éxitos: incluir secuencias que contienen una recompensa entre las experiencias de entrenamiento.\n"
         "Más incentivo a explorar: aumentar el peso de mantener acciones variadas. " + point_note, True),
    ]:
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(12, 6.5))
        draw(ax, selected, "episode/eval_score", labeler=lambda r: textwrap.fill(r.label, 42))
        refs = [random_reference(r.task).get("score") for r in selected]
        if refs and None not in refs and len(set(refs)) == 1:
            ax.axhline(refs[0], color="gray", linestyle=":", label=f"Azar (referencia): {refs[0]:g}")
        format_axis(ax, title, "Galletas por partida (media de la evaluación)")
        ax.set_ylim(bottom=0)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10, framealpha=0.9)
        save(fig, name, caption, show=show)

    # Separate panels make zero-score cases and unequal training budgets visible.
    columns = min(3, len(runs))
    rows = math.ceil(len(runs) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.7 * rows), squeeze=False, sharex=True, sharey=True)
    last_step = max((p[0] for r in runs for p in r.points("episode/eval_score")), default=1)
    max_score = max((p[1] for r in runs for p in r.points("episode/eval_score")), default=1)
    for ax, run in zip(axes.flat, runs):
        draw(ax, [run], "episode/eval_score")
        format_axis(ax, textwrap.fill(run.label, 33), "Galletas por partida")
        ax.set_xlim(0, max(last_step, 1))
        ax.set_ylim(-0.03 * max(max_score, 1), 1.18 * max(max_score, 1))
        points = run.points("episode/eval_score")
        note = f"Última: {points[-1][1]:g} galletas · paso {points[-1][0]:,}" if points else "Todavía sin evaluación"
        ax.text(0.03, 0.92, note, transform=ax.transAxes, fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    for ax in list(axes.flat)[len(runs):]:
        ax.set_visible(False)
    save(fig, "evaluacion_por_caso", "Un panel por experimento, con las mismas escalas para facilitar la comparación. " + point_note)

    fig, ax = plt.subplots(figsize=(12, max(4, len(runs) * 0.85)))
    for i, (run, info) in enumerate(zip(runs, summary)):
        points = run.points("episode/eval_score")[-last_evals:]
        if not points:
            ax.text(0.1, i, "Sin evaluación", va="center")
            continue
        mean = info["Media tramo final"]
        low, high = info["Mínimo tramo final"], info["Máximo tramo final"]
        ax.barh(i, mean, color=colors[run.path], alpha=0.3, height=0.55)
        ax.plot([low, high], [i, i], color=colors[run.path], linewidth=2)
        ax.scatter([v for _, v in points], [i] * len(points), color=colors[run.path], s=30, zorder=3)
        ax.scatter([points[-1][1]], [i], marker="D", color=colors[run.path], s=70, edgecolor="white", zorder=4)
        ax.text(1.015, i, f"Media: {mean:.2f} · última: {points[-1][1]:g}\n"
                f"{len(points)} evaluaciones · pasos {points[0][0] / 1000:g}k–{points[-1][0] / 1000:g}k",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.set_yticks(range(len(runs)), [textwrap.fill(r.label, 35) for r in runs])
    ax.invert_yaxis()
    ax.set_title(f"Rendimiento reciente · hasta {last_evals} últimas evaluaciones disponibles")
    ax.set_xlabel("Galletas por partida")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(left=-0.3)
    ax.margins(x=0.12)
    save(fig, "resumen_tramo_final", "Barra: media del tramo. Puntos y línea: valores y rango observados. Rombo: última evaluación.\n"
         "El rango describe variación entre evaluaciones de una corrida; no es un intervalo de confianza. Los pasos pueden diferir entre casos.")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, key, title in zip(axes, ["presses", "cells", "button_room"],
                              ["Veces que aprieta el botón", "Casillas distintas visitadas", "Pasos en el cuarto del botón"]):
        draw(ax, runs, f"episode/{key}", smooth=True)
        refs = [random_reference(r.task).get(key) for r in runs]
        if refs and None not in refs and len(set(refs)) == 1:
            ax.axhline(refs[0], color="gray", linestyle=":", label="Azar (referencia)")
        format_axis(ax, title, f"Promedio móvil de hasta {window} partidas")
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    save(fig, "exploracion")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, key, title, ylabel in zip(
        axes, ["episode/score", "train/rew", "train/action_entropy"],
        ["Galletas durante entrenamiento", "Recompensa imaginada por el modelo", "Entropía del actor en imaginación"],
        [f"Promedio móvil de hasta {window} partidas", "Media (escala symlog)", "Entropía"],
    ):
        draw(ax, runs, key, smooth=key.startswith("episode/"))
        format_axis(ax, title, ylabel)
        if key == "train/rew":
            ax.set_yscale("symlog", linthresh=1e-4)
        if key == "train/action_entropy":
            ax.axhline(math.log(3), color="gray", linestyle=":", label="Azar puro (3 acciones)")
        if key != "train/rew":
            ax.set_ylim(bottom=0)
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        axes[-1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    save(fig, "entrenamiento")

    if common:
        fig, ax = plt.subplots(figsize=(12, max(3.5, len(common) * 0.6)))
        ax.barh([row["Experimento"] for row in common], [row["Galletas en evaluación"] for row in common],
                color=[colors[run.path] for run in runs if run.points("episode/eval_score")])
        for i, row in enumerate(common):
            ax.annotate(f" {row['Galletas en evaluación']:g}", (row["Galletas en evaluación"], i), va="center")
        ax.set_title(f"Evaluación al mismo paso: {common_step:,} · solo runs con evaluaciones")
        ax.set_xlabel("Galletas por partida (media de la evaluación)")
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
        ax.margins(x=0.15)
        save(fig, "evaluacion_paso_comun")

    manifest = {"lectura_utc": stamp, "ventana_entrenamiento": window, "evaluaciones_tramo_final": last_evals,
                "paso_evaluacion_comun": common_step,
                "experimentos": summary,
                "interpretacion": ["Cada curva corresponde a una corrida; no se agregan semillas ni se estiman intervalos.",
                                    "Estado parcial no distingue entrenamiento activo de una sesión detenida.",
                                    "La mejor evaluación es histórica, no una evaluación final independiente.",
                                    "Drive puede mostrar métricas con atraso; vuelve a actualizar.",
                                    "Los puntos ausentes se conservan como ausentes, sin inventar resultados."]}
    manifest_path = out / "lectura.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    artifacts.append(manifest_path)
    archive = out / "comparacion_semana.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in artifacts:
            bundle.write(path, path.name)
    return {"summary": summary, "common": common, "common_step": common_step,
            "images": images, "archive": archive, "out": out, "read_at": stamp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--extra", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    result = generate_report(discover(args.root, args.extra), args.out, args.window)
    print(f"{len(result['summary'])} experimentos. Comparación guardada en {result['out']}")
