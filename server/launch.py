"""Queue the presentation's cookie experiments on the server's task spooler.

Standard library only, so it runs with the server's system python3 (no venv):

    python3 server/launch.py list
    python3 server/launch.py run exp3-normal --dry-run     # show what would run
    python3 server/launch.py check                         # a short job: does the job see the GPU?
    python3 server/launch.py run exp3-normal exp3-random   # enqueue with Slurm (sbatch) or, without it, `ts -G 1`
    python3 server/launch.py run exp3-expert               # 1M steps, data only from the expert
    python3 server/launch.py run exp3-mix                  # 1M steps, 1/3 actor + 1/3 random + 1/3 expert
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
import shlex
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
# plays, through trainer.data_mix; `mix` is any other share of episodes between
# the "actor", "random" and "expert" sources (the weights are normalised).
# `steps` are the ones in the notebooks (the expert and mix runs have no
# notebook); override them with --steps.
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
    # The same run with the three behaviours in equal parts: a third of the
    # episodes played by the actor, a third by a random agent, a third by the expert.
    "exp3-mix": dict(view=5, random=False, mix={"actor": 1, "random": 1, "expert": 1}, steps=1_000_000,
                     what="vista 5x5, datos mezclados: 1/3 actor, 1/3 azar, 1/3 experto"),
    # A one-night test on a desktop GPU (8 GB, 16 GB of RAM): the same run with a smaller
    # replay buffer and a wall-clock limit as a safety net. See "Prueba local" in docs/servidor.md.
    "local-expert": dict(view=5, random=False, expert=True, steps=450_000, buffer=200_000, max_hours=7.75, suffix="local",
                         what="PRUEBA LOCAL de una noche (<8 h): vista 5x5, solo experto, buffer de 200k"),
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


def mix_of(spec):
    """The data_mix weights of an experiment ({} when the actor plays everything)."""
    return dict(spec.get("mix") or ({"expert": 1} if spec.get("expert") else {}))


def build(name, args):
    """The Hydra overrides, the expected effective values and the run's folder."""
    spec = EXPERIMENTS[name]
    task = "cookie_full_hall5" if spec["view"] is None else f"cookie_partial_hall5_view{spec['view']}"
    mix = mix_of(spec)
    case = ("mix" if len(mix) > 1 else next(iter(mix))) if mix else "random" if spec["random"] else "base"
    folder = f"{task}_s{args.seed}_{case}" + (f"_{spec['suffix']}" if spec.get("suffix") else "")
    logroot = pathlib.Path(args.logroot).expanduser().resolve()
    if args.smoke:
        logroot = logroot / "_smoke"
    logdir = logroot / folder

    options = dict(COMMON)
    options["buffer.max_size"] = spec.get("buffer", COMMON["buffer.max_size"])
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
    max_hours = getattr(args, "max_hours", None) or spec.get("max_hours")
    if max_hours:
        options["trainer.max_hours"] = max_hours  # stops cleanly, saving, after this long
    for source, weight in mix.items():
        # Who plays the episodes of the replay buffer; the evaluations (and
        # episode/eval_score) always use the learned actor.
        options[f"trainer.data_mix.{source}"] = weight
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
    if mix:
        expected.update({f"trainer.data_mix.{source}": mix.get(source, 0) for source in ("actor", "random", "expert")})
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
    return ts_command(name, inner, args)


def ts_command(name, inner, args):
    ts = [args.ts_bin]
    if not getattr(args, "cpu", False) and args.gpus:
        ts += ["-G", str(args.gpus)]
    ts += ["-L", f"{name}"]
    env = {}
    if args.gpu_id is not None:
        # CUDA_VISIBLE_DEVICES like the DreamerV3 guide; R2D_GPU because the
        # ts client overwrites that variable with the GPU the server picked.
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        env["R2D_GPU"] = str(args.gpu_id)
    return ts + inner, env


def pick_backend(args):
    """`slurm` when sbatch exists (the server moved from task spooler to Slurm), else `ts`,
    else `local`: no scheduler, the run starts right away on this machine."""
    if args.backend != "auto":
        return args.backend
    if shutil.which("sbatch"):
        return "slurm"
    return "ts" if shutil.which(args.ts_bin) else "local"


def slurm_script(name, argv, args, outdir, time_limit, mem):
    """The sbatch script that runs `argv` (a `bash server/run_gpu.sh ...` command).

    Python runs as a job step (srun), so that `scancel -s TERM <id>` and the
    `--signal` before the time limit reach Python itself, which then stops at
    the end of its step and saves, as it does for ts -k. --cpus-per-task is
    repeated for srun because since Slurm 22.05 it no longer inherits it from
    sbatch (the step would get one core for the main process and its workers).
    Steps get all the GPUs of the job by default.
    """
    directives = [
        f"--job-name={name}",
        f"--output={outdir}/%x-%j.out",
        f"--cpus-per-task={args.cpus}",
        f"--mem={mem}",
        f"--time={time_limit}",
    ]
    if not getattr(args, "cpu", False):
        directives.append(f"--gres={args.gres or 'gpu:%d' % args.gpus}")
    if args.grace:
        # Seconds before the time limit at which the steps get SIGTERM, enough
        # to save the agent and the replay buffer, so a run that does not fit
        # in the time limit stops cleanly and can be continued.
        directives.append(f"--signal=TERM@{args.grace}")
    for flag in ("partition", "account", "qos"):
        if getattr(args, flag):
            directives.append(f"--{flag}={getattr(args, flag)}")
    directives += list(args.sbatch_opt)
    command = " ".join(shlex.quote(str(a)) for a in argv)
    lines = ["#!/bin/bash", *(f"#SBATCH {d}" for d in directives), "",
             f"cd {shlex.quote(str(REPO))}",
             'echo "=== slurm job $SLURM_JOB_ID ($SLURM_JOB_NAME) en $(hostname); nodo(s): $SLURM_JOB_NODELIST"',
             f"srun --cpus-per-task={args.cpus} {command}", ""]
    return "\n".join(lines)


def submit_slurm(name, script, args, outdir, dependency=None):
    """sbatch the script; returns its job id."""
    outdir.mkdir(parents=True, exist_ok=True)  # Slurm does not create the folder of --output
    path = outdir / f"{name}.sbatch"
    path.write_text(script)
    cmd = ["sbatch", "--parsable"] + (["--test-only"] if args.validate else [])
    if dependency:
        cmd.append(f"--dependency=afterany:{dependency}")
    done = subprocess.run(cmd + [str(path)], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if done.stderr.strip():
        print("  " + done.stderr.strip().replace("\n", "\n  "))
    if done.returncode != 0:
        raise RuntimeError(f"sbatch rechazó el job {name} (código {done.returncode}): {done.stdout.strip()}")
    return done.stdout.strip().split(";")[0] or None


def dispatch(name, argv, args, time_limit, mem, outdir, replace_process=False):
    """Run or queue `argv` with the scheduler in use. Returns an exit code.

    Without a scheduler (or with --no-ts) the run goes in the foreground. With
    `replace_process` (a single run) this process becomes the training: its pid
    is the one to signal, and a SIGTERM stops it cleanly, saving, as `ts -k` does.
    """
    backend = pick_backend(args)
    if args.no_ts or backend == "local":
        print(f"\n$ {' '.join(argv)}")
        if args.dry_run:
            return 0
        if replace_process:
            print(f"[{name}] corriendo aquí, pid {os.getpid()}. Para detenerlo y guardar: kill -TERM {os.getpid()}", flush=True)
            os.chdir(REPO)
            os.execvp(argv[0], argv)
        return subprocess.run(argv, cwd=REPO).returncode
    if backend == "ts":
        cmd, env = ts_command(name, argv, args)
        shown = " ".join(f"{k}={v}" for k, v in env.items()) + (" " if env else "")
        print(f"\n$ {shown}{' '.join(cmd)}")
        if args.dry_run:
            return 0
        done = subprocess.run(cmd, cwd=REPO, env={**os.environ, **env})
        return done.returncode
    script = slurm_script(name, argv, args, outdir, time_limit, mem)
    print(f"\n--- script de sbatch para {name} ({outdir / (name + '.sbatch')}):\n{script}---")
    if args.dry_run:
        return 0
    previous = None
    for part in range(1, (1 if args.validate else args.chain) + 1):
        job = submit_slurm(name, script, args, outdir, dependency=previous)
        print(f"[{name}] " + ("(solo validado, no se encoló) " if args.validate else
              f"job {job} encolado" + (f", parte {part} de {args.chain}, después de {previous}" if previous else "")))
        previous = job
    return 0


def add_scheduler_options(parser):
    """The options that say where and how a job is queued (`run` and `check`)."""
    parser.add_argument("--logroot", default="~/logdir/r2dreamer/presentacion", help="folder that holds all the runs")
    parser.add_argument("--backend", choices=["auto", "slurm", "ts", "local"], default="auto",
                        help="scheduler: Slurm if sbatch exists, else task spooler, else (this machine) run right away")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs requested (Slurm --gres=gpu:N, or ts -G N); default 1")
    parser.add_argument("--no-ts", "--no-queue", dest="no_ts", action="store_true", help="run in the foreground instead of queueing")
    parser.add_argument("--dry-run", action="store_true", help="print the commands/script only")
    ts = parser.add_argument_group("task spooler (the previous scheduler)")
    ts.add_argument("--gpu-id", type=int, help="force a GPU (sets CUDA_VISIBLE_DEVICES and R2D_GPU; see server/run_gpu.sh)")
    ts.add_argument("--ts-bin", default=os.environ.get("TS", "ts"), help="task spooler binary (default `ts`)")
    sl = parser.add_argument_group("Slurm")
    sl.add_argument("--partition", "-p", help="sbatch --partition (see `sinfo`)")
    sl.add_argument("--account", "-A", help="sbatch --account")
    sl.add_argument("--qos", help="sbatch --qos")
    sl.add_argument("--gres", help="the GPU request as the cluster names it, e.g. gpu:a6000:1 (default gpu:<--gpus>; see `sinfo -o '%%P %%G'`)")
    sl.add_argument("--time", dest="time_limit", help="sbatch --time (default 1-00:00:00; 01:00:00 with --smoke); at most what the partition allows")
    sl.add_argument("--mem", help="sbatch --mem (default 40G: the 500k-step buffer takes ~11 GB of RAM; 16G with --smoke)")
    sl.add_argument("--cpus", type=int, default=8, help="sbatch --cpus-per-task (default 8)")
    sl.add_argument("--grace", type=int, default=600,
                    help="seconds before the time limit at which the run gets SIGTERM to stop and save (default 600; 0 = off)")
    sl.add_argument("--chain", type=int, default=1,
                    help="queue N jobs one after another (afterany) of the same run: each resumes the previous one's checkpoint, "
                         "for runs that need more than the time limit")
    sl.add_argument("--sbatch-opt", action="append", default=[], metavar="OPT",
                    help="any other sbatch option, repeated if needed: --sbatch-opt=--constraint=a6000")
    sl.add_argument("--validate", action="store_true", help="sbatch --test-only: Slurm checks the request and nothing is queued")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="show the experiments")
    check = sub.add_parser("check", help="queue server/check_env.py (GPU, cookie env, torch.compile) as a short job")
    add_scheduler_options(check)
    run = sub.add_parser("run", help="check and enqueue experiments")
    run.add_argument("names", nargs="+", help="experiment names (see `list`), or `all` (all but local-expert)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--steps", type=int, help="override the experiment's env.steps")
    add_scheduler_options(run)
    run.add_argument("--cpu", action="store_true", help="train on CPU (device=cpu, no compile); only for tests")
    run.add_argument("--buffer-device", choices=["cpu", "cuda"], default="cpu",
                     help="where the 500k-step replay buffer lives: RAM (~11 GB) or the GPU (~11 GB of VRAM)")
    run.add_argument("--no-compile", action="store_true", help="model.compile=False (if torch.compile fails)")
    run.add_argument("--stop-at", type=int, help="stop and save at this step, to continue later")
    run.add_argument("--max-hours", type=float, help="stop and save after this many hours of this session (trainer.max_hours)")
    run.add_argument("--save-buffer-every", type=int, help="steps between replay buffer saves (default 100000; ~11 GB each)")
    run.add_argument("--no-save-buffer", action="store_true",
                     help="never write the replay buffer; a resumed run starts with an empty one")
    run.add_argument("--smoke", action="store_true", help="1,500 steps in <logroot>/_smoke (emptied first): checks that everything works")
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

    backend = pick_backend(args)
    needed = {"slurm": "sbatch", "ts": args.ts_bin, "local": None}[backend]
    if needed and not args.dry_run and not args.no_ts and shutil.which(needed) is None:
        print(f"No encuentro `{needed}` en el PATH (scheduler: {backend}). Usa --backend o --no-ts.", file=sys.stderr)
        return 2
    if args.chain < 1:
        print("--chain debe ser al menos 1", file=sys.stderr)
        return 2
    logroot = pathlib.Path(args.logroot).expanduser().resolve()

    if args.action == "check":
        argv = ["bash", str(RUN_GPU), "python", "server/check_env.py"]
        args.cpu = False
        return dispatch("check_env", argv, args, args.time_limit or "00:20:00", args.mem or "16G", logroot / "slurm")

    # `all` is the server's experiments: local-expert is a test for a desktop PC.
    names = [n for n in EXPERIMENTS if n != "local-expert"] if args.names == ["all"] else args.names
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(f"No existe: {', '.join(unknown)}. Los que hay: {', '.join(EXPERIMENTS)}", file=sys.stderr)
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
        plan.append((name, logdir, overrides))

    for name, logdir, overrides in plan:
        if args.smoke and not args.dry_run and logdir.exists():
            # A finished smoke run would resume at its last step and test nothing.
            assert "_smoke" in logdir.parts, logdir
            shutil.rmtree(logdir)
        argv = ["bash", str(RUN_GPU), "python", "-u", "train.py", *overrides]
        time_limit = args.time_limit or ("01:00:00" if args.smoke else "1-00:00:00")
        mem = args.mem or ("16G" if args.smoke else "40G")
        code = dispatch(name, argv, args, time_limit, mem, logroot / "slurm", replace_process=len(plan) == 1)
        if code != 0:
            print(f"[{name}] terminó con código {code}", file=sys.stderr)
            return code
    if not args.dry_run and not args.no_ts and not args.validate and backend != "local":
        if backend == "slurm":
            print("\nEncolado. Mira la cola con `squeue -u $USER`, el avance con `python3 server/status.py` y el log con "
                  f"`tail -f {logroot}/slurm/<nombre>-<id>.out`.")
        else:
            print("\nEncolado. Mira la cola con `ts`, el avance con `python3 server/status.py` y el log con `ts -t <id>`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
