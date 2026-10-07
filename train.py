import atexit
import pathlib
import signal
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
        # Checkpoints from before the buffer was saved on its own schedule
        # always saved it together with the agent.
        buffer_step = items.get("buffer_step", start_step)
        expected = items.get("buffer_count")
        policy_trainer.buffer_saved = (buffer_step, replay_buffer.count())
        if expected is not None and expected != replay_buffer.count():
            print(
                f"WARNING: the checkpoint was saved with {expected} transitions in the replay buffer, "
                f"but {replay_buffer.count()} were loaded. The buffer on disk is incomplete."
            )
        elif buffer_step is not None and buffer_step < start_step:
            print(
                f"The replay buffer on disk is from step {buffer_step}: the data collected between steps "
                f"{buffer_step} and {start_step} was lost when the run stopped, and is collected again."
            )
        elif replay_buffer.count() == 0:
            print("No replay buffer had been saved yet: the run goes on with an empty one and collects its data again.")

    # A first Ctrl+C or SIGTERM (what a notebook's stop button ends up sending)
    # lets the loop finish its step and save everything before quitting; a
    # second one quits at once. The handler only sets a flag, since printing
    # from it could interrupt another print.
    def request_stop(signum, frame):
        if policy_trainer.stop_requested:
            raise KeyboardInterrupt
        policy_trainer.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    final_step = policy_trainer.begin(agent, start_step)
    if final_step <= (start_step or 0):
        print(f"Nothing left to train: the run is at step {final_step} of {policy_trainer.steps}.")
        return
    if policy_trainer.stop_requested:
        print(f"Stopped at step {final_step}. Saving the agent and the replay buffer, which can take a few minutes...")
    policy_trainer.save(agent, final_step)
    if final_step >= policy_trainer.steps:
        print(f"Training finished at step {final_step}.")
    else:
        print(f"Saved at step {final_step}. Run the same command again to continue from there.")


if __name__ == "__main__":
    main()
