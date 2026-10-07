import datetime
import json
import shutil
import sys
import time

import torch
from tqdm import tqdm

import tools


def _every_from(every, start):
    """A tools.Every that fires at whole multiples of `every` after `start`.

    tools.Every fires on its first call. Spending that call here keeps a
    resumed session from saving the moment it starts, and keeps the saves on
    round steps however many times the run is resumed.
    """
    clock = tools.Every(every)
    if every:
        clock(start - start % every)
    return clock


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.logdir = logdir
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._save_every = int(config.save_every)
        self._save_buffer_every = int(config.save_buffer_every)
        self._save_buffer = bool(config.save_buffer)
        self._stop_at = int(config.stop_at) if config.get("stop_at") is not None else None
        self._action_repeat = config.action_repeat
        self._random_policy = bool(config.random_policy)
        self._progress_start = self._progress_start_step = self._progress_last = None
        # Set from a signal handler to stop at the end of the current step.
        self.stop_requested = False
        # Step and size of the replay buffer last written to disk.
        self.buffer_saved = (None, None)

    def save(self, agent, step, buffer=True):
        """Write a checkpoint this run can be resumed from.

        Written to temporary files and renamed, so a session killed mid-save
        (the common case on hosted notebooks) keeps the previous checkpoint.
        The replay buffer is most of it on disk and every save writes all of
        it, so it can be left out and saved less often than the agent. The
        checkpoint records when the buffer on disk was saved, so a resumed run
        knows which stretch of data it lost.
        """
        if buffer and self._save_buffer and self.buffer_saved[0] != step:
            start = time.time()
            tqdm.write(f"[{step}] Saving the replay buffer ({self.replay_buffer.count()} transitions)...")
            replay, tmp, old = (self.logdir / name for name in ("replay", "replay.tmp", "replay.old"))
            for path in (tmp, old):
                shutil.rmtree(path, ignore_errors=True)
            self.replay_buffer.save(tmp)
            if replay.exists():
                replay.rename(old)
            tmp.rename(replay)
            shutil.rmtree(old, ignore_errors=True)
            self.buffer_saved = (step, self.replay_buffer.count())
            tqdm.write(f"[{step}] Saved the replay buffer in {datetime.timedelta(seconds=int(time.time() - start))}.")
        buffer_step, buffer_count = self.buffer_saved
        items = {
            "step": step,
            "buffer_step": buffer_step,
            "buffer_count": buffer_count,
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "scheduler_state_dict": agent._scheduler.state_dict(),
            "scaler_state_dict": agent._scaler.state_dict(),
        }
        tmp = self.logdir / "latest.pt.tmp"
        torch.save(items, tmp)
        tmp.replace(self.logdir / "latest.pt")
        # A small summary notebooks can read without loading the checkpoint.
        status = {
            "step": step,
            "steps": self.steps,
            "buffer_step": buffer_step,
            "buffer_count": buffer_count,
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        tmp = self.logdir / "status.json.tmp"
        tmp.write_text(json.dumps(status))
        tmp.replace(self.logdir / "status.json")

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        For CPU-based environments (``ParallelEnv``), stepping is executed on
        CPU and observations are moved to GPU asynchronously.  For GPU-resident
        environments (``IsaacLabVecEnv``), no device transfer is needed —
        ``.to()`` is a no-op when source and target devices match.
        """
        tqdm.write("Evaluating the policy...")
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    def _report_progress(self, step, every=120):
        """Print how far the run is and how long it has left, every few minutes."""
        now = time.time()
        if self._progress_start is None:
            self._progress_start, self._progress_start_step, self._progress_last = now, step, now
            return
        if now - self._progress_last < every or step <= self._progress_start_step:
            return
        self._progress_last = now
        elapsed = now - self._progress_start
        speed = (step - self._progress_start_step) / elapsed
        left = (self.steps - step) / speed
        tqdm.write(
            f"[{step}] progress {100 * step / self.steps:.1f}% of {self.steps}"
            f" / elapsed {datetime.timedelta(seconds=int(elapsed))}"
            f" / left ~{datetime.timedelta(seconds=int(left))} / {speed:.1f} steps/s"
        )

    def begin(self, agent, start_step=None):
        """Main online training loop.

        For CPU-based environments the loop overlaps CPU stepping and GPU
        model execution via pinned-memory async H2D transfers.  For
        GPU-resident environments (IsaacLab) no transfer is needed —
        ``.to()`` is a no-op when the data is already on the target device.

        Returns the step it stopped at: the end of the run, trainer.stop_at,
        or wherever stop_requested was set.
        """
        envs = self.train_envs
        video_cache = []
        # A resumed run carries its own step: once the buffer is full its size
        # no longer tracks how many env steps have been taken.
        step = self.replay_buffer.count() * self._action_repeat if start_step is None else start_step
        first_step = step
        end = self.steps if self._stop_at is None else min(self.steps, self._stop_at)
        should_save = _every_from(self._save_every, step)
        should_save_buffer = _every_from(self._save_buffer_every, step)
        update_count = 0
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Kept constant so short episodes (< batch_length) remain sampable; RSSM resets via is_first.
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        # smoothing=0 bases the remaining time on the average speed of the
        # whole run, evaluations included, rather than on the last few steps.
        # The bar only shows in a terminal. A plain progress line is printed
        # every few minutes as well, since Colab's "!" commands may not draw
        # the bar. train.py mirrors sys.stderr to a log file, so it is the
        # original stream that tells whether there is a terminal.
        progress = tqdm(
            total=self.steps,
            initial=step,
            unit="step",
            smoothing=0,
            mininterval=2,
            ncols=100,
            disable=not sys.__stderr__.isatty(),
        )
        while step < end and not self.stop_requested:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0 and self.eval_envs is not None:
                progress.set_postfix_str("evaluating")
                self.eval(agent, step)
                progress.set_postfix_str("")
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
            new_steps = int((~done).sum()) * self._action_repeat
            step += new_steps  # step is based on env side
            progress.update(new_steps)
            self._report_progress(step)
            lengths += ~done

            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)
            if self._random_policy:
                # agent.act still tracks the latent state, but the action fed
                # back into it must be the one actually taken.
                act = agent.random_action(envs.env_num)
                agent_state["prev_action"] = act

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                video_cache.append(trans["image"][0])
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            # Update models after enough data has accumulated. Counted in the
            # buffer rather than in steps, since a run resumed without its
            # buffer starts far along with no data.
            if self.replay_buffer.count() // envs.env_num > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample()
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self.logger.write(step, fps=True)
            save_buffer = should_save_buffer(step)
            if should_save(step) or save_buffer:
                self.save(agent, step, buffer=bool(save_buffer))
        # The loop evaluates at the start of each pass, so it stops without
        # scoring the last stretch of training. Skipped when there was nothing
        # left to train, so rerunning a finished run does not evaluate again,
        # and when the run stops early, which should be quick.
        if step >= self.steps and step > first_step and self.eval_episode_num > 0 and self.eval_envs is not None:
            progress.set_postfix_str("evaluating")
            self.eval(agent, step)
        progress.close()
        return step
