import json
import pathlib
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import yaml

from compare_runs import common_evaluations, discover, find_video, generate_report, load_run, summarize


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)

    def run_folder(self, name, rows, status=None, task="cookie_partial_hall5_view5", sampler=0):
        path = self.root / "semana15" / name
        (path / ".hydra").mkdir(parents=True)
        config = {"env": {"task": task, "steps": 500_000, "eval_episode_num": 5},
                  "seed": 0, "buffer": {"reward_slices": sampler}, "model": {"act_entropy": 0.0003}}
        (path / ".hydra/config.yaml").write_text(yaml.safe_dump(config))
        (path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        if status is not None:
            (path / "status.json").write_text(json.dumps(status))
        return path

    def test_live_partial_line_and_actual_zero(self):
        path = self.run_folder("cookie_partial_hall5_view5_s0_base", [
            {"step": 0, "episode/eval_score": 0},
            {"step": 816, "episode/score": 0},
        ])
        with (path / "metrics.jsonl").open("a") as f:
            f.write('{"step": 2500, "train/rew":')
        run = load_run(path)
        self.assertEqual(run.points("episode/eval_score"), [(0, 0)])
        self.assertEqual(run.points("train/rew"), [])
        self.assertEqual(len(run.warnings), 1)
        with (path / "metrics.jsonl").open("a") as f:
            f.write('0.25}\n')
        self.assertEqual(load_run(path).points("train/rew"), [(2500, 0.25)])

    def test_duplicate_steps_merge_independent_metrics(self):
        path = self.run_folder("test", [{"step": 25000, "train/rew": 0.1},
                                        {"step": 25000, "episode/eval_score": 2},
                                        {"step": 25000, "episode/eval_score": 3}])
        run = load_run(path)
        self.assertEqual(run.points("train/rew"), [(25000, 0.1)])
        self.assertEqual(run.points("episode/eval_score"), [(25000, 3)])

    def test_resume_removes_superseded_tail(self):
        path = self.run_folder("test", [{"step": 0, "episode/eval_score": 0},
                                        {"step": 50000, "episode/eval_score": 20},
                                        {"step": 40000, "episode/score": 2},
                                        {"step": 45000, "episode/score": 4}])
        run = load_run(path)
        self.assertEqual(run.points("episode/eval_score"), [(0, 0)])
        self.assertEqual(summarize(run, 20)["Último paso en métricas"], 45000)

    def test_missing_evaluation_is_not_zero_or_a_finished_run(self):
        path = self.run_folder("test", [{"step": 500000, "episode/score": 2}],
                               status={"step": 490000, "steps": 500000})
        summary = summarize(load_run(path), 20)
        self.assertIsNone(summary["Última evaluación"])
        self.assertIn("Parcial", summary["Estado guardado"])
        (path / "status.json").write_text('{"step":')
        self.assertIsNone(summarize(load_run(path), 20)["Último checkpoint"])

    def test_recent_summary_keeps_last_value_and_variation(self):
        path = self.run_folder("test", [{"step": step, "episode/eval_score": score}
                                        for step, score in [(0, 0), (25000, 1), (50000, 10),
                                                            (75000, 12), (100000, 5)]], sampler=1)
        summary = summarize(load_run(path), 20, last_evals=3)
        self.assertEqual(summary["Última evaluación"], 5)
        self.assertEqual(summary["Media tramo final"], 9)
        self.assertEqual(summary["Mínimo tramo final"], 5)
        self.assertEqual(summary["Máximo tramo final"], 12)
        self.assertEqual(summary["Paso inicio tramo final"], 50000)
        self.assertEqual(summary["Evaluaciones tramo final"], 3)
        self.assertIn("experiencias donde comió galletas", summary["Experimento"])
        self.assertIn("recompensa", summary["Qué cambia"])

    def test_only_graph_uses_smoothed_training_cookies_not_evaluation(self):
        from matplotlib.axes import Axes

        path = self.run_folder("test", [{"step": 0, "episode/eval_score": 99},
                                        {"step": 816, "episode/score": 1},
                                        {"step": 1632, "episode/score": 3},
                                        {"step": 2448, "episode/score": 8}])
        curves = []
        original = Axes.plot

        def capture(ax, x, y, *args, **kwargs):
            curves.append((list(x), list(y)))
            return original(ax, x, y, *args, **kwargs)

        with patch.object(Axes, "plot", capture):
            report = generate_report([load_run(path)], self.root / "report", window=2)
        self.assertEqual(curves, [([816, 1632, 2448], [1, 2, 5.5])])
        self.assertEqual(len(report["images"]), 1)

    def test_common_evaluation_uses_exact_steps_and_skips_runs_without_evals(self):
        a = self.run_folder("a", [{"step": 25000, "episode/eval_score": 0},
                                  {"step": 50000, "episode/eval_score": 10}])
        b = self.run_folder("b", [{"step": 25000, "episode/eval_score": 2},
                                  {"step": 40000, "episode/score": 3}])
        c = self.run_folder("c", [{"step": 816, "episode/score": 0}])
        step, rows = common_evaluations([load_run(p) for p in [a, b, c]])
        self.assertEqual(step, 25000)
        self.assertEqual([r["Galletas en evaluación"] for r in rows], [0, 2])

    def test_discovery_reads_week_and_extra_without_walking_buffers(self):
        path = self.run_folder("run", [{"step": 0}])
        replay = path / "replay" / "nested"
        replay.mkdir(parents=True)
        (replay / "metrics.jsonl").write_text('{"step": 999}\n')
        other = self.root / "semana8" / "cookie_full_hall5_s0_base"
        other.mkdir(parents=True)
        (other / "metrics.jsonl").write_text('{"step": 0}\n')
        runs = discover(self.root / "semana15", [path, other, self.root / "missing"])
        self.assertEqual({r.path for r in runs}, {path, other})

    def test_videos_filter_seed_and_skip_empty_or_temporary_files(self):
        path = self.run_folder("test", [{"step": 0}])
        self.assertIsNone(find_video(path))
        (path / "agente_episodio0.mp4").touch()
        (path / "agente_episodio0.tmp.mp4").write_bytes(b"unfinished")
        other_seed = path / "agente_episodio1.mp4"
        other_seed.write_bytes(b"video")
        self.assertIsNone(find_video(path, seed=0))
        self.assertEqual(find_video(path, seed=1), other_seed)
        self.assertEqual(find_video(path, seed=None), other_seed)

    def test_video_can_come_from_report_without_modifying_run(self):
        path = self.run_folder("test", [{"step": 0}])
        video = self.root / "report" / "videos" / path.name / "agente_episodio0.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        self.assertEqual(find_video(path, self.root / "report"), video)
        self.assertFalse((path / "agente_episodio0.mp4").exists())

    def test_video_gallery_with_real_ipython_and_missing_video(self):
        import compare_runs
        import IPython.display

        path = self.run_folder("test", [{"step": 0}])
        (path / "agente_episodio0.mp4").write_bytes(b"video-fixture")
        notebook = pathlib.Path(__file__).resolve().parents[1] / "notebooks/experimentos-semana-del-15/comparacion_semana.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        source = next("".join(c["source"]) for c in cells
                      if c["cell_type"] == "code" and "def mostrar_videos()" in "".join(c["source"]))
        scope = {"Path": pathlib.Path, "RAIZ": self.root / "semana15", "EXTRAS": [], "EXCLUIR": [],
                 "ESPERADOS": {"Caso existente": path, "Caso <pendiente>": self.root / "missing"},
                 "SALIDA": self.root / "report", "compare_runs": compare_runs}
        shown = []
        with patch.object(IPython.display, "display", shown.append):
            exec(source, scope)
        self.assertEqual(len(shown), 1)
        self.assertIn("<video", shown[0].data)
        self.assertIn("base64,", shown[0].data)
        self.assertIn("Aún no hay video", shown[0].data)
        self.assertIn("Caso &lt;pendiente&gt;", shown[0].data)

    def test_export_is_read_only_and_archive_contains_current_snapshot(self):
        a = self.run_folder("a", [{"step": 0, "episode/eval_score": 0},
                                  {"step": 25000, "episode/eval_score": 4},
                                  {"step": 26000, "episode/score": 5, "episode/presses": 10,
                                   "episode/cells": 25, "episode/button_room": 155,
                                   "train/rew": -0.0001, "train/action_entropy": 0.8}],
                            status={"step": 20000, "steps": 500000}, sampler=1)
        b = self.run_folder("b", [{"step": 0, "episode/eval_score": 0},
                                  {"step": 25000, "episode/eval_score": 2}])
        paths = [p for folder in (a, b) for p in folder.rglob("*") if p.is_file()]
        original = {p: p.read_bytes() for p in paths}
        report = generate_report([load_run(a), load_run(b)], self.root / "report")
        self.assertEqual(report["common_step"], 25000)
        self.assertEqual([path.stem for path in report["images"]], ["galletas_experimentos"])
        self.assertEqual(original, {p: p.read_bytes() for p in paths})
        with zipfile.ZipFile(report["archive"]) as archive:
            self.assertIn("resumen.csv", archive.namelist())
            self.assertIn("galletas_experimentos.pdf", archive.namelist())
            manifest = json.loads(archive.read("lectura.json"))
            self.assertEqual(manifest["paso_evaluacion_comun"], 25000)
        with self.assertRaises(ValueError):
            generate_report([load_run(a)], a / "report")


if __name__ == "__main__":
    unittest.main()
