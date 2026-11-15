"""Whole-grid pictures for the videos, and the expert with an agent that only sees 3x3."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cookie_env")
pytest.importorskip("minigrid")

from envs.cookie import VIEW_OUTLINE, Cookie  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def outline_box(frame):
    """Bounding box (x0, y0, x1, y1) of the outline-coloured pixels, and how many there are."""
    mask = (frame == np.array(VIEW_OUTLINE)).all(axis=-1)
    ys, xs = np.nonzero(mask)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), int(mask.sum())


@pytest.mark.parametrize("view", [3, 5])
def test_the_frame_is_the_whole_grid_with_the_view_outlined(view):
    env = Cookie(f"partial_hall5_view{view}", (64, 64), seed=0)
    env.reset()
    frame = env.frame(16)
    # 12 x 17 cells of 16 pixels, whatever the agent sees.
    assert frame.shape == (12 * 16, 17 * 16, 3) and frame.dtype == np.uint8
    # At the start the agent stands in the middle of a hallway, so its whole view is inside the grid.
    (x0, y0, x1, y1), count = outline_box(frame)
    assert (x1 - x0, y1 - y0) == (view * 16, view * 16)
    assert count > 0


def test_the_frame_follows_the_agent_and_is_clipped_at_the_edge_of_the_grid():
    env = Cookie("partial_hall5_view5", (64, 64), seed=0)
    env.reset()
    before = env.frame(16)
    for _ in range(10):
        env.step(env.expert_action())
    after = env.frame(16)
    assert not np.array_equal(before, after)
    # A view hanging over the grid's border is outlined only where it is inside.
    inner = env._env.unwrapped
    inner.agent_pos = (0, 0)
    inner.agent_dir = 2
    (x0, y0, x1, y1), _ = outline_box(env.frame(16))
    assert x0 >= 0 and y0 >= 0 and x1 <= 17 * 16 and y1 <= 12 * 16


def test_render_still_gives_a_picture_for_watch_py():
    env = Cookie("partial_hall5_view3", (64, 64), seed=0)
    env.reset()
    assert env.render().shape == (12 * env._env.unwrapped.tile_size, 17 * env._env.unwrapped.tile_size, 3)


def run_expert(view, seed=3):
    env = Cookie(f"partial_hall5_view{view}", (64, 64), seed=seed)
    env.reset()
    actions, cookies = [], 0
    done = False
    while not done:
        actions.append(env.expert_action())
        obs, reward, done, _ = env.step(actions[-1])
        cookies += int(reward > 0)
    return actions, cookies, obs["image"].shape


def test_the_expert_plays_the_same_with_a_3x3_agent_as_with_a_5x5_one():
    # The agent's view only changes the images it gets: the expert decides from its own 5x5 view.
    actions5, cookies5, shape5 = run_expert(5)
    actions3, cookies3, shape3 = run_expert(3)
    assert actions3 == actions5 and cookies3 == cookies5 >= 19
    assert shape3 == shape5 == (64, 64, 3)  # both are resized to the world model's input size
    assert run_expert(7)[0] == actions5


def test_the_expert_view_is_the_5x5_one_whatever_the_agent_sees():
    env = Cookie("partial_hall5_view3", (64, 64), seed=0)
    env.reset()
    inner = env._env.unwrapped
    assert inner.agent_view_size == 3 and inner.gen_obs()["image"].shape == (3, 3, 3)
    grid, visible = inner.gen_obs_grid(5)
    assert grid.encode(visible).shape == (5, 5, 3)
    env.expert_action()  # does not raise: it no longer needs a view of 5x5 or more


def test_parallel_envs_hand_out_the_frames_and_say_they_can():
    from omegaconf import OmegaConf

    from envs import make_envs

    config = OmegaConf.create(
        dict(task="cookie_partial_hall5_view3", env_num=1, eval_episode_num=0, action_repeat=1, time_limit=120,
             press_on_move=True, seed=0, device="cpu", size=[64, 64])
    )
    train_envs, _, _, _ = make_envs(config)
    try:
        assert train_envs.can_frame
        import torch

        train_envs.step(torch.zeros(1, 3), torch.ones(1, dtype=torch.bool))  # the first reset
        frame = train_envs.frame(0, 16)
        assert frame.shape == (12 * 16, 17 * 16, 3)
    finally:
        for env in train_envs.envs:
            env.close()


def trainer_config(**trainer):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="configs", overrides=["env=cookie"] + [f"trainer.{k}={v}" for k, v in trainer.items()])
        OmegaConf.resolve(cfg)
    return cfg.trainer


def test_video_render_needs_environments_that_can_draw_the_grid():
    from trainer import OnlineTrainer

    cannot = SimpleNamespace(env_num=1)
    can = SimpleNamespace(env_num=1, can_frame=True)
    with pytest.raises(AssertionError, match="cookie"):
        OnlineTrainer(trainer_config(video_render=True), None, None, None, cannot, None)
    trainer = OnlineTrainer(trainer_config(video_render=True, train_video_every=10, video_tile_size=12), None, None, None, can, can)
    assert (trainer._video_render, trainer._train_video_every, trainer._video_tile) == (True, 10, 12)
    # Off by default: the older runs keep logging the agent's observation.
    assert not OnlineTrainer(trainer_config(), None, None, None, cannot, None)._video_render
