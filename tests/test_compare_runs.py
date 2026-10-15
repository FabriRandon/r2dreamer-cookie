import json
import pathlib
import tempfile
import unittest
import zipfile

import yaml

from compare_runs import common_evaluations, discover, generate_report, load_run, summarize


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
        self.assertEqual(len(report["images"]), 5)
        self.assertEqual(original, {p: p.read_bytes() for p in paths})
        with zipfile.ZipFile(report["archive"]) as archive:
            self.assertIn("resumen.csv", archive.namelist())
            self.assertIn("evaluacion_todos.pdf", archive.namelist())
            manifest = json.loads(archive.read("lectura.json"))
            self.assertEqual(manifest["paso_evaluacion_comun"], 25000)
        with self.assertRaises(ValueError):
            generate_report([load_run(a)], a / "report")


if __name__ == "__main__":
    unittest.main()
