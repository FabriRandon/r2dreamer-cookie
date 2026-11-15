"""The expert for an agent that sees 3x3 cells (envs/expert.py: ExpertPolicy3x3)."""

import collections
import itertools
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cookie_env")
pytest.importorskip("minigrid")

from envs.cookie import Cookie  # noqa: E402
from envs.expert import (  # noqa: E402
    COOKIE, EAST, NORTH, SOUTH, STEP, WALL, WEST, ExpertPolicy, ExpertPolicy3x3, _walkable, make_expert,
)

ROOT = Path(__file__).resolve().parents[1]
HALLWAY = 5
BACK = HALLWAY + 1  # the back wall of a side room comes into view standing 6 cells from the start
# The only places the cookie appears: the outer corners of the two side rooms.
CORNERS = [(side * (HALLWAY + 2), dy) for side in (-1, 1) for dy in (-2, 2)]
PRESS = (0, -(HALLWAY - 1), NORTH)
WALK = _walkable(HALLWAY)


# ---- an independent shortest-path search over (cell, heading), to check the policy against ----
def shortest(start, cell, heading=None, via=None):
    """Fewest actions from the pose `start` to `cell` (with `heading`, if given), going through
    the cell `via` on the way if given. Turns and moves cost one action each."""
    first = (start, via is None or start[:2] == via)
    seen, queue = {first}, deque([(first, 0)])
    while queue:
        (pose, passed), steps = queue.popleft()
        if passed and pose[:2] == cell and heading in (None, pose[2]):
            return steps
        x, y, d = pose
        dx, dy = STEP[d]
        options = [(x, y, (d - 1) % 4), (x, y, (d + 1) % 4)]
        if (x + dx, y + dy) in WALK:
            options.append((x + dx, y + dy, d))
        for nxt in options:
            state = (nxt, passed or nxt[:2] == via)
            if state not in seen:
                seen.add(state)
                queue.append((state, steps + 1))
    raise AssertionError(f"{cell} cannot be reached from {start}")


class World:
    """A cookie environment with a 3x3 agent, and helpers to put things where a test wants them."""

    def __init__(self, seed=0, agent_view=3, expert_view=3):
        self.env = Cookie(f"partial_hall5_view{agent_view}", (64, 64), seed=seed, expert_view=expert_view)
        self.env.reset()
        self.inner = self.env._env.unwrapped
        self.origin = tuple(int(v) for v in self.inner.agent_pos)

    def real_pose(self):
        (x, y), d = self.inner.agent_pos, self.inner.agent_dir
        return int(x) - self.origin[0], int(y) - self.origin[1], int(d)

    def put_cookie(self, corner):
        self.inner._remove_cookie()
        self.inner.put_obj(self.inner.cookie, self.origin[0] + corner[0], self.origin[1] + corner[1])

    def view(self, pose, size=3):
        """The ids the agent would see standing at `pose` (relative to the start)."""
        self.inner.agent_pos = (self.origin[0] + pose[0], self.origin[1] + pose[1])
        self.inner.agent_dir = pose[2]
        grid, visible = self.inner.gen_obs_grid(size)
        return grid.encode(visible)[..., 0]

    def sees_cookie(self, pose, corner):
        self.put_cookie(corner)
        return bool((self.view(pose) == COOKIE).any())


@pytest.fixture(scope="module")
def world():
    return World()


# ---- the geometry the expert relies on, against the real environment ----
@pytest.mark.parametrize("corner", CORNERS)
def test_a_corner_comes_into_view_only_by_turning_at_the_back_of_its_room(world, corner):
    side = 1 if corner[0] > 0 else -1
    wanted, other = (NORTH, SOUTH) if corner[1] < 0 else (SOUTH, NORTH)
    into = EAST if side > 0 else WEST
    back = (side * BACK, 0)
    assert world.sees_cookie((*back, wanted), corner)
    assert not world.sees_cookie((*back, other), corner)  # the opposite corner is not in view
    assert not world.sees_cookie((*back, into), corner)  # walking in, it has to turn to see it
    # Before the back of the room, in the middle row, no heading shows it.
    for x in range(1, BACK):
        for heading in (NORTH, SOUTH, into):
            assert not world.sees_cookie((side * x, 0, heading), corner), (x, heading)


@pytest.mark.parametrize("side", [-1, 1])
def test_the_back_wall_comes_into_view_exactly_at_the_back_of_the_room(world, side):
    into = EAST if side > 0 else WEST
    for x in range(1, HALLWAY + 3):
        ids = world.view((side * x, 0, into))
        # Two cells ahead, in the middle of the view: the room's back wall.
        assert (ids[1, 0] == WALL) == (x >= BACK), x


def test_cookies_only_appear_in_the_four_outer_corners():
    seen = set()
    for seed in range(40):
        w = World(seed)
        for _ in range(120):
            obs, reward, done, _ = w.env.step(w.env.expert_action())
            if reward > 0:
                seen.add(w.real_pose()[:2])
    assert seen == set(CORNERS)


# ---- the expert's cycle, step by step, against the independent search ----
class Script:
    """Stands in for the expert's random generator to try every choice."""

    def __init__(self, rooms, verticals):
        self.rooms, self.verticals = list(rooms), [list(v) for v in verticals]

    def sample(self, population, k):
        return list(self.rooms) if tuple(population) == (-1, 1) else list(self.verticals.pop(0))


def expected_cycle(corner, rooms, verticals):
    """Steps from one button press to the next when it looks into `rooms` in that order, at the
    corners of each in the order of `verticals`, and the cookie is at `corner`."""
    side = 1 if corner[0] > 0 else -1
    wanted = "up" if corner[1] < 0 else "down"
    pose, total, found = PRESS, 0, False
    for room, order in zip(rooms, verticals):
        into, back = (EAST if room > 0 else WEST), (room * BACK, 0)
        total += shortest(pose, back, into)  # walk in until the back wall is in view
        pose = (*back, into)
        for vertical in order:
            facing = NORTH if vertical == "up" else SOUTH
            total += shortest(pose, back, facing)  # turn to the corner
            pose = (*back, facing)
            if room == side and vertical == wanted:
                found = True
                break
        if found:
            break
    assert found
    return total + shortest(pose, PRESS[:2], PRESS[2], via=corner) + 1  # fetch it, come back, press


def run_cycle(world, policy, corner):
    """One cycle from the first press to the second, the cookie put at `corner` after each press.
    The expert is only given the agent's own 3x3 observation."""
    world.env.reset()
    policy.reset()
    pressed = steps = cookies = 0
    while pressed < 2:
        action = policy.act(world.inner.gen_obs()["image"])
        assert world.inner.gen_obs()["image"].shape == (3, 3, 3)
        assert policy.pose == world.real_pose()  # what it believes is what the simulator says
        obs, reward, done, _ = world.env.step(action)
        assert not done
        if pressed:
            steps += 1
            cookies += int(reward > 0)
        if obs["log_presses"]:
            pressed += 1
            world.put_cookie(corner)
    return steps, cookies


CASES = list(itertools.product(CORNERS, [(1, -1), (-1, 1)], itertools.product([("up", "down"), ("down", "up")], repeat=2)))


@pytest.mark.parametrize("corner,rooms,verticals", CASES)
def test_every_cycle_takes_the_fewest_steps_its_search_order_allows(world, corner, rooms, verticals):
    policy = ExpertPolicy3x3(HALLWAY)
    policy._rng = Script(rooms, verticals)
    steps, cookies = run_cycle(world, policy, corner)
    assert cookies == 1
    assert steps == expected_cycle(corner, rooms, verticals)


def test_the_four_cycle_lengths():
    # 1st corner of the first room, 2nd corner of it (a U-turn more), and the same in the other room.
    lengths = {expected_cycle(c, r, v) for c, r, v in CASES}
    assert lengths == {35, 37, 51, 53}


# ---- whole episodes through the environment's own expert_action(), many seeds ----
def play_episode(seed):
    world = World(seed)
    env, inner = world.env, world.inner
    cookies, presses, bad_pose, steps, done = [], [], 0, 0, False
    search_poses, fetch_poses = [], []
    while not done:
        action = env.expert_action()
        bad_pose += env._expert.pose != world.real_pose()
        (search_poses if env._expert.phase == "search" else fetch_poses).append(world.real_pose())
        obs, reward, done, _ = env.step(action)
        steps += 1
        if reward > 0:
            cookies.append(world.real_pose()[:2])
        if obs["log_presses"]:
            presses.append(steps)
    return dict(steps=steps, cookies=cookies, presses=presses, bad_pose=bad_pose, search=search_poses)


@pytest.fixture(scope="module")
def episodes():
    return [play_episode(seed) for seed in range(30)]


def test_whole_episodes_are_clean(episodes):
    for e in episodes:
        assert e["steps"] == 816  # the time limit of this map, reached without incident
        assert e["bad_pose"] == 0
        assert 16 <= len(e["cookies"]) <= 21
        assert set(e["cookies"]) <= set(CORNERS)
        assert 0 <= len(e["presses"]) - len(e["cookies"]) <= 1  # one cookie per press, the last cycle may be unfinished


def test_cycle_lengths_and_corners_are_what_the_algorithm_says(episodes):
    gaps = collections.Counter(b - a for e in episodes for a, b in zip(e["presses"], e["presses"][1:]))
    assert set(gaps) == {35, 37, 51, 53}
    total = sum(gaps.values())
    for length, count in gaps.items():
        assert 0.15 < count / total < 0.35, (length, count / total)  # the four corners are equally likely
    assert {c for e in episodes for c in e["cookies"]} == set(CORNERS)


def test_it_walks_in_only_until_the_back_wall_is_in_view(episodes):
    # While searching it never goes past the back-wall cell of a room, never leaves the middle row
    # inside a room, and so only looks at the corners by turning.
    for e in episodes:
        for x, y, _ in e["search"]:
            if abs(x) >= 3:
                assert y == 0 and abs(x) <= BACK, (x, y)


def test_the_first_room_and_corner_are_chosen_at_random():
    firsts = collections.Counter()
    for seed in range(160):
        w = World(seed)
        for _ in range(80):
            w.env.step(w.env.expert_action())
            if w.env._expert.looks:
                firsts[w.env._expert.looks[0]] += 1
                break
    assert sum(firsts.values()) == 160
    assert set(firsts) == {(s, v) for s in (-1, 1) for v in ("up", "down")}
    for combo, count in firsts.items():
        assert 0.15 < count / 160 < 0.35, (combo, count)


def test_the_first_choice_does_not_know_where_the_cookie_appeared():
    # The environment draws the cookie's corner from a generator seeded with the env's seed; the
    # expert must not reuse that seed, or its first room would always be the right one.
    counts = collections.Counter()
    for seed in range(240):
        w = World(seed)
        for _ in range(60):
            _, _, _, _ = w.env.step(w.env.expert_action())
            if w.env._expert.rooms:  # it has just pressed the button: the cookie is on the grid
                cookie_x = w.inner.cookie.cur_pos[0] - w.origin[0]
                counts[(1 if cookie_x > 0 else -1) == w.env._expert.rooms[0]] += 1
                break
    assert sum(counts.values()) == 240
    assert 0.35 < counts[True] / 240 < 0.65, counts  # about half the time, as chance says


# ---- plumbing ----
def test_the_factory_and_the_views_each_expert_accepts():
    assert type(make_expert(3)) is ExpertPolicy3x3 and type(make_expert(5)) is ExpertPolicy
    with pytest.raises(ValueError):
        make_expert(4)
    with pytest.raises(AssertionError, match="5x5"):
        ExpertPolicy().act(np.zeros((3, 3, 3), dtype=np.uint8))
    with pytest.raises(AssertionError, match="3x3"):
        ExpertPolicy3x3().act(np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.mark.parametrize("agent_view", [3, 5])
def test_the_3x3_expert_plays_for_agents_with_any_view(agent_view):
    # What the expert sees is its own view; the agent's only changes the images the agent gets.
    w = World(seed=1, agent_view=agent_view, expert_view=3)
    cookies, done = 0, False
    while not done:
        obs, reward, done, _ = w.env.step(w.env.expert_action())
        cookies += int(reward > 0)
    assert 16 <= cookies <= 21 and obs["image"].shape == (64, 64, 3)


def press_gaps(**env_options):
    """Steps between button presses of the expert, played through the parallel envs the trainer uses."""
    import torch
    from omegaconf import OmegaConf

    from envs import make_envs

    config = OmegaConf.create(dict(task="cookie_partial_hall5_view3", env_num=1, eval_episode_num=0, action_repeat=1,
                                   time_limit=2088, press_on_move=True, seed=0, device="cpu", size=[64, 64], **env_options))
    envs, _, _, _ = make_envs(config)
    try:
        done, act, presses = torch.ones(1, dtype=torch.bool), torch.zeros(1, 3), []
        for step in range(780):  # shorter than an episode: no restart to handle
            trans, done = envs.step(act, done)
            if trans["log_presses"].item():
                presses.append(step)
            act = torch.zeros(1, 3)
            act[0, int(envs.expert_actions([0])[0])] = 1
    finally:
        for env in envs.envs:
            env.close()
    return {b - a for a, b in zip(presses, presses[1:])}


def test_the_expert_view_is_part_of_the_environment_config():
    # env.expert_view=3 selects the 3x3 expert; without the option (older configs) the 5x5 one is used.
    assert press_gaps(expert_view=3) <= {35, 37, 51, 53} and len(press_gaps(expert_view=3)) >= 3
    assert press_gaps() == {35, 43}


def test_the_demo_shows_the_four_cases_in_order():
    out = subprocess.run(
        [sys.executable, str(ROOT / "demo_expert.py"), "--expert-view", "3", "--showcase", "--out", str(ROOT / "expert_demo_3x3")],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    assert "cycles of [35, 37, 51, 53] steps" in out
