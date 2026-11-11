"""server/launch.py must queue exactly what the presentation's notebooks ran."""

import argparse
import ast
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import launch  # noqa: E402

FOLDER = ROOT / "EXPERIMENTOS PRESENTACIÓN"
NOTEBOOKS = {
    "exp1-normal": "experimento_1_vision_completa_normal.ipynb",
    "exp1-random": "experimento_1_vision_completa_random.ipynb",
    "exp2-normal": "experimento_2_vista_7x7_normal.ipynb",
    "exp2-random": "experimento_2_vista_7x7_random.ipynb",
    "exp3-normal": "experimento_3_vista_5x5_normal.ipynb",
    "exp3-random": "experimento_3_vista_5x5_random.ipynb",
    "exp4-normal": "experimento_4_vista_3x3_normal.ipynb",
    "exp4-random": "experimento_4_vista_3x3_random.ipynb",
    "extra": "experimento_extra_vista_5x5_normal_repetir_galletas_entropia_10x.ipynb",
}


def args_for(**changes):
    values = dict(
        seed=0, steps=None, logroot="/runs", smoke=False, cpu=False, no_compile=False, stop_at=None,
        no_save_buffer=False, save_buffer_every=None, buffer_device="cpu", extra=[],
    )
    values.update(changes)
    return argparse.Namespace(**values)


def notebook_config(path):
    """The notebook's configuration cell: its constants, evaluated without running anything."""
    cells = json.loads(path.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells if c["cell_type"] == "code" and "OPCIONES = " in "".join(c["source"]))
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.Assign)]
    scope = {}
    exec(compile(tree, str(path), "exec"), scope)
    return scope


def as_dict(options):
    return dict(item.split("=", 1) for item in options)


def same(a, b):
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


class LaunchTests(unittest.TestCase):
    def test_every_experiment_has_a_notebook_except_the_expert_run(self):
        self.assertEqual(set(launch.EXPERIMENTS) - {"exp3-expert"}, set(NOTEBOOKS))

    def test_the_expert_run_is_the_random_run_with_the_expert_collecting(self):
        mine, expected, logdir, folder = launch.build("exp3-expert", args_for())
        mine = as_dict(mine)
        random = as_dict(launch.build("exp3-random", args_for())[0])
        self.assertEqual(mine["env.steps"], "1000000")
        self.assertEqual(folder, "cookie_partial_hall5_view5_s0_expert")
        self.assertEqual(mine["trainer.data_mix.expert"], "1")
        self.assertEqual(mine["trainer.random_policy"], "False")  # trainer refuses random_policy together with data_mix
        # Same task and settings as the random run of the same view; only who collects the data differs.
        differs = {k for k in set(mine) | set(random) if k != "logdir" and mine.get(k) != random.get(k)}
        self.assertEqual(differs, {"trainer.random_policy", "trainer.data_mix.expert"})
        self.assertEqual(expected["trainer.data_mix.expert"], 1)
        self.assertEqual(expected["trainer.data_mix.actor"], 0)
        # The other experiments do not touch data_mix.
        for name in launch.EXPERIMENTS:
            if name != "exp3-expert":
                self.assertNotIn("trainer.data_mix.expert", as_dict(launch.build(name, args_for())[0]))

    def test_matches_the_notebooks(self):
        for name, filename in NOTEBOOKS.items():
            with self.subTest(name):
                scope = notebook_config(FOLDER / filename)
                overrides, expected, logdir, folder = launch.build(name, args_for())
                mine = as_dict(overrides)
                notebook = as_dict(shlex.split(scope["OPCIONES"]))
                # Every option the notebook passes has the same value here.
                for key, value in notebook.items():
                    self.assertIn(key, mine, key)
                    self.assertTrue(same(mine[key], value), (key, mine[key], value))
                # What the notebook adds when it launches training.
                self.assertEqual(int(mine["env.steps"]), scope["PASOS"])
                self.assertEqual(int(mine["env.eval_episode_num"]), scope["PARTIDAS_EVAL"])
                self.assertEqual(int(mine["trainer.eval_every"]), scope["EVALUAR_CADA"])
                self.assertEqual(int(mine["trainer.save_buffer_every"]), scope["GUARDAR_BUFFER_CADA"])
                self.assertEqual(int(mine["trainer.save_every"]), 10_000)
                self.assertEqual(int(mine["trainer.update_log_every"]), 2_500)
                # The folder is named like the notebook's, so compare_runs reads it the same way.
                self.assertEqual(folder, scope["LOGDIR"].rsplit("/", 1)[1])
                self.assertEqual(expected["trainer.random_policy"], name.endswith("random"))

    def test_random_runs_keep_the_actor_learning(self):
        for name in launch.EXPERIMENTS:
            overrides, *_ = launch.build(name, args_for())
            mine = as_dict(overrides)
            self.assertEqual(mine["model.train_actor_critic"], "True")
            self.assertEqual(mine["trainer.random_policy"], str(name.endswith("random")))

    def test_extra_overrides_replace_instead_of_repeating(self):
        overrides, expected, *_ = launch.build(
            "exp3-normal", args_for(extra=["buffer.max_size=20000", "env.steps=7", "+foo=1"])
        )
        keys = [item.split("=", 1)[0] for item in overrides]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(as_dict(overrides)["buffer.max_size"], "20000")
        self.assertIn("+foo=1", overrides)
        self.assertNotIn("env.steps", expected)

    def test_flags(self):
        mine = as_dict(launch.build("exp4-random", args_for(
            seed=3, steps=1000, buffer_device="cuda", no_compile=True, stop_at=500, no_save_buffer=True))[0])
        self.assertEqual(mine["seed"], "3")
        self.assertEqual(mine["env.steps"], "1000")
        self.assertEqual(mine["buffer.storage_device"], "cuda:0")
        self.assertEqual(mine["model.compile"], "False")
        self.assertEqual(mine["trainer.stop_at"], "500")
        self.assertEqual(mine["trainer.save_buffer"], "False")
        cpu = as_dict(launch.build("exp4-random", args_for(cpu=True, buffer_device="cuda"))[0])
        self.assertEqual((cpu["device"], cpu["buffer.storage_device"], cpu["model.compile"]), ("cpu", "cpu", "False"))

    def test_smoke_never_touches_real_runs(self):
        overrides, _, logdir, _ = launch.build("exp1-normal", args_for(smoke=True))
        self.assertIn("_smoke", logdir.parts)
        mine = as_dict(overrides)
        self.assertEqual((mine["env.steps"], mine["trainer.save_buffer"]), ("1500", "False"))

    def test_command_goes_through_ts_with_a_gpu(self):
        args = args_for(no_ts=False, ts_bin="ts", gpus=1, gpu_id=None, cpu=False)
        cmd, env = launch.command("exp3-normal", ["env=cookie"], args, "x")
        self.assertEqual(cmd[:5], ["ts", "-G", "1", "-L", "exp3-normal"])
        self.assertEqual(cmd[5], "bash")
        self.assertTrue(cmd[6].endswith("server/run_gpu.sh"))
        self.assertEqual(cmd[7:10], ["python", "-u", "train.py"])
        self.assertEqual(env, {})
        cmd, env = launch.command("exp3-normal", ["env=cookie"], args_for(
            no_ts=False, ts_bin="ts", gpus=1, gpu_id=1, cpu=False), "x")
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "1", "R2D_GPU": "1"})
        cmd, _ = launch.command("exp3-normal", ["env=cookie"], args_for(
            no_ts=False, ts_bin="ts", gpus=1, gpu_id=None, cpu=True), "x")
        self.assertNotIn("-G", cmd)


def slurm_args(**changes):
    values = dict(
        backend="slurm", no_ts=False, dry_run=False, validate=False, chain=1, cpus=8, gpus=1, gres=None, grace=600,
        partition=None, account=None, qos=None, sbatch_opt=[], cpu=False, ts_bin="ts", gpu_id=None,
    )
    values.update(changes)
    return argparse.Namespace(**values)


class SlurmTests(unittest.TestCase):
    ARGV = ["bash", "/repo/server/run_gpu.sh", "python", "-u", "train.py", "env=cookie", "env.size=[64,64]"]

    def test_script(self):
        script = launch.slurm_script("exp3-expert", self.ARGV, slurm_args(partition="gpu", account="lab"),
                                     Path("/logs/slurm"), "1-00:00:00", "40G")
        lines = script.splitlines()
        self.assertEqual(lines[0], "#!/bin/bash")
        for directive in ("--job-name=exp3-expert", "--output=/logs/slurm/%x-%j.out", "--cpus-per-task=8", "--mem=40G",
                          "--time=1-00:00:00", "--gres=gpu:1", "--signal=TERM@600", "--partition=gpu", "--account=lab"):
            self.assertIn(f"#SBATCH {directive}", lines)
        # One job step; the brackets of env.size are quoted so bash does not glob them.
        run = lines[-1]
        self.assertTrue(run.startswith("srun --cpus-per-task=8 bash "), run)
        self.assertIn("'env.size=[64,64]'", run)
        self.assertNotIn("--qos", script)

    def test_options(self):
        script = launch.slurm_script("x", self.ARGV, slurm_args(gres="gpu:a6000:1", grace=0, sbatch_opt=["--constraint=a"]),
                                     Path("/o"), "00:20:00", "16G")
        self.assertIn("#SBATCH --gres=gpu:a6000:1", script)
        self.assertIn("#SBATCH --constraint=a", script)
        self.assertNotIn("--signal", script)
        self.assertNotIn("--gres", launch.slurm_script("x", self.ARGV, slurm_args(cpu=True), Path("/o"), "1:00", "1G"))

    def test_chain_queues_each_job_after_the_previous_one(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return argparse.Namespace(returncode=0, stdout=f"{100 + len(calls)}\n", stderr="")

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(launch.subprocess, "run", fake_run):
            launch.dispatch("exp3-expert", self.ARGV, slurm_args(chain=3), "1:00", "1G", Path(folder))
            self.assertTrue((Path(folder) / "exp3-expert.sbatch").exists())
        self.assertEqual(len(calls), 3)
        self.assertFalse(any(a.startswith("--dependency") for a in calls[0]))
        self.assertIn("--dependency=afterany:101", calls[1])
        self.assertIn("--dependency=afterany:102", calls[2])
        self.assertTrue(all(c[0] == "sbatch" and "--parsable" in c for c in calls))

    def test_validate_queues_nothing(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return argparse.Namespace(returncode=0, stdout="7\n", stderr="")

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(launch.subprocess, "run", fake_run):
            launch.dispatch("x", self.ARGV, slurm_args(validate=True, chain=3), "1:00", "1G", Path(folder))
        self.assertEqual(len(calls), 1)
        self.assertIn("--test-only", calls[0])

    def test_dry_run_from_the_command_line(self):
        env = {**os.environ, "R2D_VENV": "/nonexistent"}  # skips the Hydra check, which needs the venv
        out = subprocess.run(
            [sys.executable, str(ROOT / "server/launch.py"), "run", "exp3-expert", "--dry-run", "--backend", "slurm",
             "--partition", "gpu", "--logroot", "/tmp/r2d-test-logs"], env=env, capture_output=True, text=True, check=True,
        ).stdout
        self.assertIn("#SBATCH --partition=gpu", out)
        self.assertIn("trainer.data_mix.expert=1", out)
        self.assertIn("1000000", out)
        out = subprocess.run(
            [sys.executable, str(ROOT / "server/launch.py"), "check", "--dry-run", "--backend", "slurm"],
            env=env, capture_output=True, text=True, check=True,
        ).stdout
        self.assertIn("server/check_env.py", out)
        self.assertIn("#SBATCH --time=00:20:00", out)


if __name__ == "__main__":
    unittest.main()
