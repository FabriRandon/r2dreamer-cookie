"""Contract checks for the four explicitly random-collection Colab runs."""

import ast
import contextlib
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks/experimentos-semana-del-22"
EXPECTED = {
    "vision_completa_random.ipynb": None,
    "vision_parcial_3x3_random.ipynb": 3,
    "vision_parcial_5x5_random.ipynb": 5,
    "vision_parcial_7x7_random.ipynb": 7,
}


def notebook_scope(path):
    notebook = json.loads(path.read_text())
    source = "".join(notebook["cells"][12]["source"])
    tree = ast.parse(source)
    # Load constants and the actual verification function, without starting
    # processes or printing notebook output as a side effect of loading.
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.Assert))]
    namespace = {}
    exec(compile(tree, str(path), "exec"), namespace)
    namespace["REPO"] = str(ROOT)
    return notebook, namespace


class RandomNotebookTests(unittest.TestCase):
    def test_exactly_four_notebooks_and_only_observation_differs(self):
        self.assertEqual({p.name for p in FOLDER.glob("*.ipynb")}, set(EXPECTED))
        all_options, destinations = [], set()
        for filename, view in EXPECTED.items():
            notebook, scope = notebook_scope(FOLDER / filename)
            self.assertEqual(scope["VISTA"], view)
            self.assertEqual(scope["CASO"], "random")
            options = dict(arg.split("=", 1) for arg in shlex.split(scope["OPCIONES"]))
            self.assertEqual(options["trainer.random_policy"], "True")
            self.assertEqual(options["model.train_actor_critic"], "True")
            self.assertEqual(options["buffer.reward_slices"], "0")
            self.assertEqual(options["model.act_entropy"], "0.0003")
            self.assertIn("/galleta/semana22/", scope["LOGDIR"])
            self.assertTrue(scope["LOGDIR"].endswith("_random"))
            self.assertEqual(scope["PASOS"], 500_000)
            self.assertTrue(scope["APAGAR_AL_TERMINAR"])
            destinations.add(scope["LOGDIR"])
            options.pop("env.task")
            all_options.append(options)
            training = "".join(notebook["cells"][21]["source"])
            self.assertLess(training.index("verificar_random(comando[3:])"),
                            training.index("subprocess.Popen("))
            self.assertIn("episode/eval_score", "".join(notebook["cells"][23]["source"]))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
        self.assertEqual(len(destinations), 4)
        self.assertTrue(all(option == all_options[0] for option in all_options))

    @unittest.skipUnless((ROOT / ".venv/bin/python").exists(), "Local training environment is required for Hydra checks")
    def test_effective_hydra_configuration_is_random_in_all_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            for filename in EXPECTED:
                _, scope = notebook_scope(FOLDER / filename)
                overrides = shlex.split(scope["OPCIONES"]) + [f"logdir={tmp}/{filename}"]
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    scope["verificar_random"](overrides)
                self.assertIn("VERIFICADO: trainer.random_policy=True", output.getvalue())

    @unittest.skipUnless((ROOT / ".venv/bin/python").exists(), "Local training environment is required for Hydra checks")
    def test_false_at_end_of_effective_command_is_rejected(self):
        _, scope = notebook_scope(FOLDER / "vision_parcial_5x5_random.ipynb")
        with tempfile.TemporaryDirectory() as tmp:
            overrides = shlex.split(scope["OPCIONES"]) + [f"logdir={tmp}", "trainer.random_policy=False"]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "NO se ha iniciado"):
                scope["verificar_random"](overrides)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    @unittest.skipUnless((ROOT / ".venv/bin/python").exists(), "Local training environment is required for Hydra checks")
    def test_previous_actor_run_is_rejected_and_left_untouched(self):
        _, scope = notebook_scope(FOLDER / "vision_parcial_5x5_random.ipynb")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / ".hydra").mkdir()
            config = path / ".hydra/config.yaml"
            content = "trainer:\n  random_policy: false\n"
            config.write_text(content)
            checkpoint = path / "latest.pt"
            checkpoint.write_bytes(b"existing checkpoint fixture")
            overrides = shlex.split(scope["OPCIONES"]) + [f"logdir={tmp}"]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "NO se ha iniciado"):
                scope["verificar_random"](overrides)
            self.assertEqual(config.read_text(), content)
            self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint fixture")


if __name__ == "__main__":
    unittest.main()
