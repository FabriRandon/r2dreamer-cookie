import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Pointing a new run at a logdir that already holds a checkpoint continues
    # it, so a session cut short can be picked up where it stopped.
    checkpoint = logdir / "latest.pt"
    items = None
    if config.resume and checkpoint.exists():
        items = torch.load(checkpoint, map_location=config.device, weights_only=False)
    start_step = int(items["step"]) if items is not None else None

    # A resumed run is reseeded with its step: the original seed would replay
    # the episodes and random actions it already collected after the start.
    seed = config.seed if start_step is None else config.seed + start_step
    tools.set_seed_everywhere(seed)
    config.env.seed = seed
    if config.deterministic_run:
        tools.enable_deterministic_run()

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)

    if items is not None:
        agent.load_state_dict(items["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, items["optims_state_dict"])
        agent._scheduler.load_state_dict(items["scheduler_state_dict"])
        agent._scaler.load_state_dict(items["scaler_state_dict"])
        replay = logdir / "replay"
        if replay.exists():
            replay_buffer.load(replay)
        print(f"Resuming from {checkpoint} at step {start_step} with {replay_buffer.count()} transitions.")
        expected = items.get("buffer_count")
        if expected is not None and expected != replay_buffer.count():
            print(
                f"WARNING: the checkpoint was saved with {expected} transitions in the replay buffer, "
                f"but {replay_buffer.count()} were loaded. The buffer on disk is incomplete."
            )

    policy_trainer.begin(agent, start_step)
    policy_trainer.save(agent, policy_trainer.steps)


if __name__ == "__main__":
    main()
