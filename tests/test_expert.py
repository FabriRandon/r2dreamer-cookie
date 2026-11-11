"""The hand-written expert policy against the real cookie environment."""

import itertools

import pytest

pytest.importorskip("cookie_env")
pytest.importorskip("minigrid")

from envs.cookie import Cookie  # noqa: E402
from envs.expert import EAST, FORWARD, NORTH, START, STEP, ExpertPolicy, _first_action, _walkable  # noqa: E402

HALLWAY = 5
CORNERS = [(sx * (HALLWAY + dx), dy) for sx in (-1, 1) for dx in (-2, 2) for dy in (-2, 2)]
PRESS = (0, -(HALLWAY - 1), NORTH)


def walk(pose, cell, heading=None):
    """Steps of a shortest path and the pose it ends in (an independent check of the
    policy: it replays `_first_action` on a bare simulator of the three rooms)."""
    walkable, steps = _walkable(HALLWAY), 0
    while (action := _first_action(HALLWAY, pose, cell, heading)) is not None:
        x, y, d = pose
        if action == FORWARD:
            dx, dy = STEP[d]
            assert (x + dx, y + dy) in walkable
            pose = (x + dx, y + dy, d)
        else:
            pose = (x, y, (d + (1 if action == 1 else -1)) % 4)
        steps += 1
    return steps, pose


class Order:
    """A stand-in for the policy's random generator that picks the room order."""

    def __init__(self, order):
        self.order = list(order)

    def sample(self, population, k):
        return list(self.order)


def expected_cycle(corner, order):
    """Steps from one button press to the next, for a cookie at `corner` when the
    policy looks into the rooms in `order`; counts only the optimal walking."""
    side = 1 if corner[0] > 0 else -1
    policy = ExpertPolicy(HALLWAY)
    pose, total = PRESS, 0
    for room in order:
        x, y, heading = policy.vantage(room)
        steps, pose = walk(pose, (x, y), heading)
        total += steps
        if room == side:
            break
    steps, pose = walk(pose, corner)
    total += steps
    steps, _ = walk(pose, PRESS[:2], PRESS[2])
    return total + steps + 1  # the last step is the press


def run_cycle(env, policy, corner, order, origin):
    """One cycle from the first press to the next, with the cookie put at `corner`.
    Returns the steps it took and the cookies collected."""
    inner = env._env.unwrapped
    policy._rng = Order(order)
    pressed, steps, cookies = 0, 0, 0
    obs = None
    while pressed < 2:
        action = policy.act(inner.gen_obs()["image"])
        # What the policy believes must be what the simulator says.
        pos = (int(inner.agent_pos[0]) - origin[0], int(inner.agent_pos[1]) - origin[1], int(inner.agent_dir))
        assert policy.pose == pos
        obs, reward, done, _ = env.step(action)
        assert not done
        # The cycle is what happens after the first press, up to and including the second.
        if pressed:
            steps += 1
            cookies += int(reward > 0)
        if obs["log_presses"]:
            pressed += 1
            # The environment drops the cookie in a random corner; use the one under test.
            inner._remove_cookie()
            inner.put_obj(inner.cookie, origin[0] + corner[0], origin[1] + corner[1])
    return steps, cookies


@pytest.fixture(scope="module")
def env():
    return Cookie("partial_hall5_view5", (64, 64), seed=0)


@pytest.mark.parametrize("corner,order", list(itertools.product(CORNERS, [(-1, 1), (1, -1)])))
def test_every_cycle_gets_one_cookie_in_the_fewest_steps(env, corner, order):
    env.reset()
    inner = env._env.unwrapped
    origin = tuple(int(v) for v in inner.agent_pos)
    assert START == (0, 0, EAST)
    policy = ExpertPolicy(HALLWAY)
    policy.reset()
    # First press, with the cookie of that cycle put at the corner too.
    steps, cookies = run_cycle(env, policy, corner, order, origin)
    assert cookies == 1

    side = 1 if corner[0] > 0 else -1
    pose = walk(PRESS, corner)[1]
    oracle = walk(PRESS, corner)[0] + walk(pose, PRESS[:2], PRESS[2])[0] + 1  # knowing where the cookie is
    # Turning around at the button while facing west, the 5x5 view reaches the
    # room's inner corner (-3, -2) through the wall, and the policy takes it
    # right away: as fast as knowing where it is.
    glimpsed = corner == (-(HALLWAY - 2), -2)
    expected = oracle if order[0] == side or glimpsed else expected_cycle(corner, order)
    assert steps == expected
    assert oracle <= steps <= expected_cycle(corner, order)


def test_looks_into_a_room_from_its_doorway_and_sees_all_of_it(env):
    # The reason the policy stops in the doorway: a 5x5 view from there covers the whole room.
    env.reset()
    inner = env._env.unwrapped
    origin = tuple(int(v) for v in inner.agent_pos)
    policy = ExpertPolicy(HALLWAY)
    for side in (-1, 1):
        x, y, d = policy.vantage(side)
        for corner in CORNERS:
            if (corner[0] > 0) != (side > 0):
                continue
            inner._remove_cookie()
            inner.put_obj(inner.cookie, origin[0] + corner[0], origin[1] + corner[1])
            inner.agent_pos = (origin[0] + x, origin[1] + y)
            inner.agent_dir = d
            assert 12 in inner.gen_obs()["image"][..., 0]


def test_whole_episode_collects_a_cookie_each_cycle(env):
    env.reset()
    policy = ExpertPolicy(HALLWAY, seed=3)
    policy.reset()
    inner = env._env.unwrapped
    origin = tuple(int(v) for v in inner.agent_pos)
    cookies = presses = 0
    for _ in range(600):
        action = policy.act(inner.gen_obs()["image"])
        assert policy.pose == (int(inner.agent_pos[0]) - origin[0], int(inner.agent_pos[1]) - origin[1], int(inner.agent_dir))
        obs, reward, done, _ = env.step(action)
        cookies += int(reward > 0)
        presses += int(obs["log_presses"])
    # One cookie per press, the last cycle may be unfinished.
    assert presses - 1 <= cookies <= presses
    assert presses >= 15
