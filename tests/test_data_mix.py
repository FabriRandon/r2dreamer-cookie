"""Which source (actor, random, expert) plays each collected episode."""

import numpy as np
import pytest

from trainer import ACTOR, EXPERT, RANDOM, DataMix


def test_nothing_set_means_the_actor_plays_everything():
    for weights in (None, {}, {"actor": 0, "random": 0, "expert": 0}):
        assert list(DataMix(weights).sample(5)) == [ACTOR] * 5


def test_the_old_random_policy_flag_is_all_random():
    assert list(DataMix(None, random_policy=True).sample(3)) == [RANDOM] * 3
    with pytest.raises(AssertionError):
        DataMix({"expert": 1}, random_policy=True)


def test_a_weight_alone_is_a_hundred_percent_of_it():
    assert list(DataMix({"expert": 1}).sample(4)) == [EXPERT] * 4
    assert list(DataMix({"expert": 0.3}).sample(4)) == [EXPERT] * 4


def test_a_single_source_does_not_touch_the_random_generator():
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    DataMix({"expert": 1}).sample(10)
    np.random.seed(0)
    assert (np.random.get_state()[1] == before).all()


def test_weights_are_normalised_and_followed():
    np.random.seed(1)
    mix = DataMix({"actor": 2, "random": 1, "expert": 1})
    assert mix.probs == pytest.approx([0.5, 0.25, 0.25])
    sources = mix.sample(20_000)
    for source, share in ((ACTOR, 0.5), (RANDOM, 0.25), (EXPERT, 0.25)):
        assert np.mean(sources == source) == pytest.approx(share, abs=0.02)


@pytest.mark.parametrize("weights", [{"nope": 1}, {"actor": -1, "expert": 2}])
def test_bad_mixes_are_refused(weights):
    with pytest.raises(AssertionError):
        DataMix(weights)


# The expert inside the real environment, in its worker processes.
@pytest.fixture
def cookie_envs():
    pytest.importorskip("cookie_env")
    import torch
    from omegaconf import OmegaConf

    from envs import make_envs

    config = OmegaConf.create(
        dict(
            task="cookie_partial_hall5_view5", env_num=2, eval_episode_num=0, action_repeat=1, time_limit=120,
            press_on_move=True, seed=0, device="cpu", size=[64, 64],
        )
    )
    train_envs, _, _, act_space = make_envs(config)
    yield train_envs, act_space.shape[0], torch
    for env in train_envs.envs:
        env.close()


def test_the_expert_plays_whole_episodes_through_the_parallel_envs(cookie_envs):
    envs, n_actions, torch = cookie_envs
    done = torch.ones(2, dtype=torch.bool)
    act = torch.zeros(2, n_actions)
    cookies = presses = episodes = 0
    for _ in range(500):
        restarting = done.numpy().copy()
        trans, done = envs.step(act, done)
        episodes += int(done.sum())
        cookies += int((trans["reward"][:, 0] > 0).sum())
        presses += int(trans["log_presses"].sum())
        # What the trainer does: the expert's action for the envs still running.
        running = [i for i in range(2) if not done[i]]
        act = torch.zeros(2, n_actions)
        if running:
            act[running] = torch.nn.functional.one_hot(torch.as_tensor(envs.expert_actions(running)), n_actions).float()
    assert episodes >= 6  # 120-step episodes, restarted by the trainer's rule
    # Each press is followed by a cookie, except the last cycle of an episode.
    assert cookies >= presses - episodes
    assert cookies >= 20


def test_taking_another_action_than_the_experts_is_an_error(cookie_envs):
    envs, n_actions, torch = cookie_envs
    done = torch.ones(2, dtype=torch.bool)
    envs.step(torch.zeros(2, n_actions), done)
    wanted = int(envs.expert_actions([0])[0])
    other = torch.zeros(2, n_actions)
    other[0, (wanted + 1) % n_actions] = 1
    other[1, 0] = 1
    with pytest.raises(Exception, match="expert asked"):
        envs.step(other, torch.zeros(2, dtype=torch.bool))
        envs.step(other, torch.zeros(2, dtype=torch.bool))
