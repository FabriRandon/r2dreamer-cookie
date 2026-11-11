"""Queue the presentation's cookie experiments on the server's task spooler.

Standard library only, so it runs with the server's system python3 (no venv):

    python3 server/launch.py list
    python3 server/launch.py run exp3-normal --dry-run     # show what would run
    python3 server/launch.py run exp3-normal exp3-random   # enqueue with `ts -G 1`
    python3 server/launch.py run exp3-expert               # 1M steps, data only from the expert
    python3 server/launch.py run all --smoke               # 1,500-step test of every experiment

Every experiment mirrors a Colab notebook of "EXPERIMENTOS PRESENTACIÓN/"
(same task, hyperparameters and folder names), so the runs can be compared
with progress.py and compare_runs.py exactly as the notebooks did. Before
queueing, the effective Hydra configuration is checked against the experiment
(the way the random notebooks did), and so is the configuration of a run that
already exists in the same folder: pointing a run at a folder that holds a
different experiment would silently continue it with the wrong settings.

Running the same command again resumes a run from its last checkpoint.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN_GPU = REPO / "server" / "run_gpu.sh"

# Settings shared by every notebook of the presentation.
COMMON = {
    "env": "cookie",
    "env.train_ratio": 128,
    "env.size": "[64,64]",
    "model.rep_loss": "dreamer",  # with decoder
    "trainer.video_pred_log": True,
    "buffer.max_size": 500_000,  # the whole run, in RAM
    "env.eval_episode_num": 5,
    "trainer.eval_every": 25_000,
    "trainer.update_log_every": 2_500,
    "trainer.save_every": 10_000,
    "trainer.save_buffer_every": 100_000,
}

# name -> what differs. `view` None is the whole map. `random` is the case where
# a uniformly random agent collects the data and the actor learns from it;
# `expert` the one where only the hand-written perfect policy (envs/expert.py)
# plays, through trainer.data_mix. `steps` are the ones in the notebooks (the
# expert run has no notebook); override them with --steps.
EXPERIMENTS = {
    "exp1-normal": dict(view=None, random=False, steps=500_000, what="mapa completo, el actor junta los datos"),
    "exp1-random": dict(view=None, random=True, steps=500_000, what="mapa completo, datos al azar"),
    "exp2-normal": dict(view=7, random=False, steps=500_000, what="vista 7x7, el actor junta los datos"),
    "exp2-random": dict(view=7, random=True, steps=500_000, what="vista 7x7, datos al azar"),
    "exp3-normal": dict(view=5, random=False, steps=500_000, what="vista 5x5, el actor junta los datos"),
    "exp3-random": dict(view=5, random=True, steps=1_000_000, what="vista 5x5, datos al azar"),
    "exp4-normal": dict(view=3, random=False, steps=500_000, what="vista 3x3, el actor junta los datos"),
    "exp4-random": dict(view=3, random=True, steps=750_000, what="vista 3x3, datos al azar"),
    # Needs a view of at least 5x5 and the hallways of 5: what the expert knows.
    "exp3-expert": dict(view=5, random=False, expert=True, steps=1_000_000,
                        what="vista 5x5, SOLO trayectorias del experto"),
    "extra": dict(
        view=5, random=False, steps=500_000, reward_slices=1, act_entropy=3e-3, suffix="sampler1_entropia10x",
        what="vista 5x5, repite experiencias con galleta y 10x entropía",
    ),
}

# The keys that must agree with the run already in a folder before resuming it.
SAME_AS_SAVED = (
    "env.task", "env.size", "model.rep_loss", "trainer.random_policy", "model.train_actor_critic",
    "buffer.reward_slices", "model.act_entropy", "seed",
    "trainer.data_mix.actor", "trainer.data_mix.random", "trainer.data_mix.expert",
)

# Runs inside the repo's venv: composes the config the way train.py will and
# compares it with what is expected and with the folder's saved config.
CHECK = r"""
import json, sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

repo, overrides, expected, same_keys = sys.argv[1], *map(json.loads, sys.argv[2:4]), json.loads(sys.argv[4])
with initialize_config_dir(config_dir=str(Path(repo) / "configs"), version_base=None):
    cfg = compose(config_name="configs", overrides=overrides)
    OmegaConf.resolve(cfg)
errors = []
for key, want in expected.items():
    got = OmegaConf.select(cfg, key)
    got = OmegaConf.to_container(got) if OmegaConf.is_config(got) else got
    if got != want:
        errors.append(f"{key} es {got!r} y debería ser {want!r}")
saved = Path(cfg.logdir) / ".hydra" / "config.yaml"
if len(sys.argv) > 5:  # smoke test: its folder is emptied before the run
    pass
elif saved.exists():
    old = OmegaConf.load(saved)
    for key in same_keys:
        a, b = OmegaConf.select(old, key), OmegaConf.select(cfg, key)
        a = OmegaConf.to_container(a) if OmegaConf.is_config(a) else a
        b = OmegaConf.to_container(b) if OmegaConf.is_config(b) else b
        if a is None and key.startswith("trainer.data_mix."):
            a = 0.0  # a run from before data_mix existed: the actor (or random_policy) played everything
        if a != b:
            errors.append(f"la carpeta {cfg.logdir} ya tiene un run con {key}={a!r}, no {b!r}; usa otra --logroot o bórrala")
elif (Path(cfg.logdir) / "latest.pt").exists():
    errors.append(f"{cfg.logdir} tiene un checkpoint sin configuración verificable; usa otra carpeta")
if errors:
    print("\n".join("ERROR: " + e for e in errors))
    sys.exit(1)
print(f"config ok: task={cfg.env.task} random_policy={cfg.trainer.random_policy} steps={cfg.env.steps:g} logdir={cfg.logdir}")
"""


def fmt(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build(name, args):
    """The Hydra overrides, the expected effective values and the run's folder."""
    spec = EXPERIMENTS[name]
    task = "cookie_full_hall5" if spec["view"] is None else f"cookie_partial_hall5_view{spec['view']}"
    case = "expert" if spec.get("expert") else "random" if spec["random"] else "base"
    folder = f"{task}_s{args.seed}_{case}" + (f"_{spec['suffix']}" if spec.get("suffix") else "")
    logroot = pathlib.Path(args.logroot).expanduser().resolve()
    if args.smoke:
        logroot = logroot / "_smoke"
    logdir = logroot / folder

    options = dict(COMMON)
    options.update({
        "env.task": task,
        "seed": args.seed,
        "env.steps": args.steps or spec["steps"],
        "trainer.random_policy": spec["random"],
        "buffer.reward_slices": spec.get("reward_slices", 0),
        "model.act_entropy": spec.get("act_entropy", 3e-4),
        "model.train_actor_critic": True,  # the actor still learns, in imagination, from the expert's data
        "buffer.storage_device": "cpu" if args.buffer_device == "cpu" else "cuda:0",
        "logdir": logdir,
    })
    if spec.get("expert"):
        # Every episode of the replay buffer is played by the expert; the
        # evaluations (and episode/eval_score) still use the learned actor.
        options["trainer.data_mix.expert"] = 1
    if args.cpu:
        options.update({"device": "cpu", "buffer.storage_device": "cpu", "model.compile": False})
    if args.no_compile:
        options["model.compile"] = False
    if args.stop_at:
        options["trainer.stop_at"] = args.stop_at
    if args.no_save_buffer:
        options["trainer.save_buffer"] = False
    elif args.save_buffer_every:
        options["trainer.save_buffer_every"] = args.save_buffer_every
    if args.smoke:
        options.update({
            "env.steps": 1_500, "env.eval_episode_num": 1, "trainer.update_log_every": 500,
            "trainer.save_every": 1_000_000_000, "trainer.save_buffer": False, "buffer.max_size": 20_000,
        })

    # Extra overrides replace the ones above instead of repeating a key.
    for item in args.extra:
        key, _, value = item.partition("=")
        if key.startswith(("+", "~")):
            continue
        options[key] = value
    overrides = [f"{key}={fmt(value)}" for key, value in options.items()]
    overrides += [item for item in args.extra if item.startswith(("+", "~"))]
    expected = {
        "env.task": task, "trainer.random_policy": spec["random"], "model.train_actor_critic": True,
        "buffer.reward_slices": spec.get("reward_slices", 0), "model.act_entropy": spec.get("act_entropy", 3e-4),
        "env.steps": options["env.steps"], "seed": args.seed, "model.rep_loss": "dreamer",
    }
    if spec.get("expert"):
        expected.update({"trainer.data_mix.expert": 1, "trainer.data_mix.actor": 0, "trainer.data_mix.random": 0})
    if args.extra:  # the user knowingly overrides these
        extra_keys = {item.split("=", 1)[0].lstrip("+~") for item in args.extra}
        expected = {k: v for k, v in expected.items() if k not in extra_keys}
    return overrides, expected, logdir, folder


def venv_python():
    venv = pathlib.Path(os.environ.get("R2D_VENV", REPO / ".venv"))
    return venv / "bin" / "python"


def verify(overrides, expected, smoke=False):
    python = venv_python()
    if not python.exists():
        print(f"  (no se verifica la configuración: falta {python}; corre server/setup.sh)")
        return True
    done = subprocess.run(
        [str(python), "-c", CHECK, str(REPO), json.dumps(overrides), json.dumps(expected), json.dumps(SAME_AS_SAVED)] + (["smoke"] if smoke else []),
        cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print("  " + done.stdout.strip().replace("\n", "\n  "))
    return done.returncode == 0


def command(name, overrides, args, folder):
    inner = ["bash", str(RUN_GPU), "python", "-u", "train.py", *overrides]
    if args.no_ts:
        return inner, {}
    ts = [args.ts_bin]
    if not args.cpu and args.gpus:
        ts += ["-G", str(args.gpus)]
    ts += ["-L", f"{name}"]
    env = {}
    if args.gpu_id is not None:
        # CUDA_VISIBLE_DEVICES like the DreamerV3 guide; R2D_GPU because the
        # ts client overwrites that variable with the GPU the server picked.
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        env["R2D_GPU"] = str(args.gpu_id)
    return ts + inner, env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="show the experiments")
    run = sub.add_parser("run", help="check and enqueue experiments")
    run.add_argument("names", nargs="+", help="experiment names (see `list`), or `all`")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--steps", type=int, help="override the experiment's env.steps")
    run.add_argument("--logroot", default="~/logdir/r2dreamer/presentacion", help="folder that holds all the runs")
    run.add_argument("--gpus", type=int, default=1, help="GPUs requested from ts with -G (default 1)")
    run.add_argument("--gpu-id", type=int, help="force a GPU (sets CUDA_VISIBLE_DEVICES and R2D_GPU; see server/run_gpu.sh)")
    run.add_argument("--ts-bin", default=os.environ.get("TS", "ts"), help="task spooler binary (default `ts`)")
    run.add_argument("--no-ts", action="store_true", help="run in the foreground instead of queueing")
    run.add_argument("--cpu", action="store_true", help="train on CPU (device=cpu, no compile); only for tests")
    run.add_argument("--buffer-device", choices=["cpu", "cuda"], default="cpu",
                     help="where the 500k-step replay buffer lives: RAM (~11 GB) or the GPU (~11 GB of VRAM)")
    run.add_argument("--no-compile", action="store_true", help="model.compile=False (if torch.compile fails)")
    run.add_argument("--stop-at", type=int, help="stop and save at this step, to continue later")
    run.add_argument("--save-buffer-every", type=int, help="steps between replay buffer saves (default 100000; ~11 GB each)")
    run.add_argument("--no-save-buffer", action="store_true",
                     help="never write the replay buffer; a resumed run starts with an empty one")
    run.add_argument("--smoke", action="store_true", help="1,500 steps in <logroot>/_smoke (emptied first): checks that everything works")
    run.add_argument("--dry-run", action="store_true", help="check the configuration and print the commands only")
    run.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                     help="after this flag, extra Hydra overrides, e.g. --extra model.lr=2e-5")
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.action == "list":
        for name, spec in EXPERIMENTS.items():
            print(f"{name:12s} {spec['steps']:>9,} pasos  {spec['what']}")
        return 0

    names = list(EXPERIMENTS) if args.names == ["all"] else args.names
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(f"No existe: {', '.join(unknown)}. Los que hay: {', '.join(EXPERIMENTS)}", file=sys.stderr)
        return 2
    if not args.dry_run and not args.no_ts and shutil.which(args.ts_bin) is None:
        print(f"No encuentro `{args.ts_bin}` en el PATH. Usa --no-ts para correr en primer plano.", file=sys.stderr)
        return 2

    plan = []
    for name in names:
        overrides, expected, logdir, folder = build(name, args)
        print(f"[{name}] {logdir}")
        if not verify(overrides, expected, args.smoke):
            print(f"[{name}] configuración rechazada: NO se encola.", file=sys.stderr)
            return 1
        if args.extra:
            print(f"  AVISO: --extra cambia ajustes pero la carpeta sigue llamándose {folder}. Si es una variante nueva, "
                  "usa otra --logroot; solo se comprueban contra un run existente: " + ", ".join(SAME_AS_SAVED) + ".")
        plan.append((name, logdir, *command(name, overrides, args, folder)))

    for name, logdir, cmd, env in plan:
        if args.smoke and not args.dry_run and logdir.exists():
            # A finished smoke run would resume at its last step and test nothing.
            assert "_smoke" in logdir.parts, logdir
            shutil.rmtree(logdir)
        shown = " ".join(f"{k}={v}" for k, v in env.items()) + (" " if env else "")
        print(f"\n$ {shown}{' '.join(cmd)}")
        if args.dry_run:
            continue
        done = subprocess.run(cmd, cwd=REPO, env={**os.environ, **env})
        if done.returncode != 0:
            print(f"[{name}] terminó con código {done.returncode}", file=sys.stderr)
            return done.returncode
    if not args.dry_run and not args.no_ts:
        print("\nEncolado. Mira la cola con `ts`, el avance con `python3 server/status.py` y el log con `ts -t <id>`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
