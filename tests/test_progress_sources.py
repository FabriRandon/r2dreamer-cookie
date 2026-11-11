"""progress.py and status.py must not read the expert's (or a random agent's) training score as the actor's."""

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import progress  # noqa: E402


def run_folder(rows, task="cookie_partial_hall5_view5"):
    folder = pathlib.Path(tempfile.mkdtemp()) / "run"
    (folder / ".hydra").mkdir(parents=True)
    (folder / ".hydra/config.yaml").write_text(f"env:\n  task: {task}\n")
    (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return folder


def summary(folder):
    rows, status, task = progress.load(folder)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        progress.summarize(folder, rows, status, task, 20)
    return out.getvalue()


EXPERT_ROWS = [{"step": 816 * i, "episode/score": 20.0, "episode/score_expert": 20.0} for i in range(1, 4)]


class SourcesTests(unittest.TestCase):
    def test_collectors(self):
        self.assertEqual(progress.collectors(EXPERT_ROWS), {"expert"})
        self.assertEqual(progress.collectors([{"step": 1, "episode/score": 1.0}]), set())

    def test_expert_data_is_not_read_as_the_actor_learning(self):
        text = summary(run_folder(EXPERT_ROWS))
        self.assertIn("NO miden al actor", text)
        self.assertIn("Todavía no hay evaluaciones del actor", text)
        self.assertNotIn("el caso base está aprendiendo", text)
        text = summary(run_folder(EXPERT_ROWS + [{"step": 2448, "episode/eval_score": 12.0}]))
        self.assertIn("El actor ya consigue galletas en la evaluación", text)
        text = summary(run_folder(EXPERT_ROWS + [{"step": 2448, "episode/eval_score": 0.0}]))
        self.assertIn("todavía no consigue galletas", text)

    def test_actor_runs_are_read_as_before(self):
        rows = [{"step": 816 * i, "episode/score": 20.0, "episode/score_actor": 20.0} for i in range(1, 4)]
        text = summary(run_folder(rows))
        self.assertIn("el caso base está aprendiendo", text)
        self.assertNotIn("NO miden", text)

    def test_status_labels_the_data_and_leaves_actor_score_empty(self):
        folder = run_folder(EXPERT_ROWS)
        out = subprocess.run([sys.executable, str(ROOT / "server/status.py"), "--root", str(folder.parent)],
                             capture_output=True, text=True, check=True).stdout
        line = out.splitlines()[2]  # after the header and its rule
        self.assertIn("exp", line)
        self.assertIn("nan", line)


if __name__ == "__main__":
    unittest.main()
