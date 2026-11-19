"""Solid abstract obstacles in every room (envs/obstacles.py), and the experts playing around them."""

import collections
import itertools

import numpy as np
import pytest

pytest.importorskip("cookie_env")
pytest.importorskip("minigrid")

from minigrid.core.constants import OBJECT_TO_IDX  # noqa: E402
from minigrid.core.grid import Grid  # noqa: E402

from envs import expert, obstacles  # noqa: E402
from envs.cookie import Cookie  # noqa: E402
from envs.obstacles import COLOURS, PALETTE, SHAPES, Obstacle  # noqa: E402

HALLWAY = 5
CORNERS = [(side * (HALLWAY + 2), dy) for side in (-1, 1) for dy in (-2, 2)]  # where cookies appear
DOORS = {"boton": (0, -HALLWAY + 2), "oeste": (-HALLWAY + 2, 0), "este": (HALLWAY - 2, 0)}
ROOM_CENTERS = {"boton": (0, -HALLWAY), "oeste": (-HALLWAY, 0), "este": (HALLWAY, 0)}


class World:
    def __init__(self, seed=0, view=3, expert_view=3, obstacles=1):
        self.env = Cookie(f"partial_hall5_view{view}", (64, 64), seed=seed, expert_view=expert_view, obstacles=obstacles)
        self.env.reset()
        self.inner = self.env._env.unwrapped
        self.origin = tuple(int(v) for v in self.inner.agent_pos)

    def rel(self, cell):
        return cell[0] - self.origin[0], cell[1] - self.origin[1]

    def layout(self):
        return [(room, self.rel(cell), shape, colour) for room, cell, shape, colour in self.env._placed]

    def real_pose(self):
        (x, y), d = self.inner.agent_pos, self.inner.agent_dir
        return (*self.rel((int(x), int(y))), int(d))

    def reach(self):
        """Cells (relative to the start) the agent can walk to, by an independent search over the real grid."""
        seen, queue = {self.origin}, [self.origin]
        while queue:
            x, y = queue.pop()
            for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                cell = self.inner.grid.get(*nxt)
                if nxt not in seen and (cell is None or cell.can_overlap()):
                    seen.add(nxt)
                    queue.append(nxt)
        return {self.rel(c) for c in seen}


def facing(world, index=2):
    """Put the agent on a free cell next to the obstacle `index`, looking at it; returns that obstacle."""
    room, cell, shape, colour = world.env._placed[index]
    for heading, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
        before = (cell[0] - dx, cell[1] - dy)
        if world.inner.grid.get(*before) is None:
            world.inner.agent_pos, world.inner.agent_dir = before, heading
            return room, cell, shape, colour, before
    raise AssertionError("an obstacle with no free neighbour")


# ---- where, what and how they are drawn ----
def test_one_obstacle_per_room_never_on_a_cookie_corner_or_a_doorway():
    world = World(seed=0)
    for _ in range(400):
        world.env.reset()
        placed = world.layout()
        assert [p[0] for p in placed] == ["boton", "oeste", "este"]
        for room, cell, shape, colour in placed:
            cx, cy = ROOM_CENTERS[room]
            assert max(abs(cell[0] - cx), abs(cell[1] - cy)) <= 2  # inside its own room
            assert cell not in CORNERS and cell != DOORS[room] and cell != (0, -HALLWAY)  # not the button's either
        # Different shapes and different colours while there are enough of them.
        assert len({p[2] for p in placed}) == 3 and len({p[3] for p in placed}) == 3


def test_the_cookie_corners_and_the_button_stay_reachable():
    world = World(seed=1)
    for _ in range(400):
        world.env.reset()
        reach = world.reach()
        assert all(corner in reach for corner in CORNERS)
        assert any(n in reach for n in ((1, -HALLWAY), (-1, -HALLWAY), (0, -HALLWAY + 1), (0, -HALLWAY - 1)))


def test_every_free_cell_gets_used_and_every_shape_and_colour():
    world = World(seed=2)
    cells, shapes, colours = collections.defaultdict(collections.Counter), set(), set()
    for _ in range(1500):
        world.env.reset()
        for room, cell, shape, colour in world.layout():
            cells[room][cell] += 1
            shapes.add(shape)
            colours.add(colour)
    assert shapes == set(SHAPES) and colours == set(COLOURS)
    sizes = {room: len(c) for room, c in cells.items()}
    assert sizes == {"boton": 23, "oeste": 22, "este": 22}  # 25 cells less the button/the corners and the doorway
    for room, counter in cells.items():
        expected = 1500 / len(counter)
        assert min(counter.values()) > 0.4 * expected and max(counter.values()) < 1.9 * expected, room  # about uniform


def test_a_new_layout_every_episode_and_a_reproducible_run():
    a, b, c = World(seed=5), World(seed=5), World(seed=6)
    seq = lambda w: [w.layout()] + [(w.env.reset(), w.layout())[1] for _ in range(30)]  # noqa: E731
    la, lb, lc = seq(a), seq(b), seq(c)
    assert la == lb  # the same seed gives the same run
    assert la != lc  # another seed, another one
    assert sum(x != y for x, y in zip(la, la[1:])) >= 29  # and it changes at every reset


def test_reseeding_the_env_replays_the_same_obstacles():
    # watch.py and the explorer reseed an env by setting its seed and `_seeded = False`.
    env = Cookie("partial_hall5_view3", (64, 64), seed=0, obstacles=1)
    runs = []
    for _ in range(2):
        env._seed, env._seeded = 5, False
        env.reset()
        runs.append([env._placed] + [(env.reset(), env._placed)[1] for _ in range(3)])
    assert runs[0] == runs[1]


def test_the_captions_of_the_demo_videos_fit_in_the_frame():
    from PIL import ImageFont

    import demo_expert

    font = demo_expert.font(14)
    width = 17 * 32  # the grid of this map, drawn with 32 pixels per cell, is as wide as the video
    for case in demo_expert.SHOWCASE + demo_expert.OBSTACLE_SHOWCASE:
        assert font.getlength(case["case"]) + 8 < width, case["case"]


def test_no_obstacles_changes_nothing_else():
    plain = Cookie("partial_hall5_view3", (64, 64), seed=4)
    zero = Cookie("partial_hall5_view3", (64, 64), seed=4, obstacles=0)
    one = Cookie("partial_hall5_view3", (64, 64), seed=4, obstacles=1)
    a, b, c = plain.reset(), zero.reset(), one.reset()
    assert set(a) == set(b) and "log_bumps" not in a and "log_bumps" in c
    assert np.array_equal(a["image"], b["image"])
    assert "log_bumps" not in plain.observation_space.spaces and "log_bumps" in one.observation_space.spaces


def tile(obj, size=32, agent=None):
    return np.clip(Grid.render_tile(obj, agent_dir=agent, tile_size=size), 0, 255)


def test_they_do_not_look_like_the_button_the_agent_or_the_cookie():
    from cookie_env.objects import Button, Cookie as CookieObj

    others = {"boton": tile(Button("blue")), "galleta": tile(CookieObj()), "agente": tile(None, agent=0)}
    seen = {}
    for shape, colour in itertools.product(SHAPES, COLOURS):
        picture = tile(Obstacle(shape, colour))
        seen[(shape, colour)] = picture
        for name, other in others.items():
            assert np.abs(picture - other).mean() > 12, (shape, colour, name)
    # Colours well away from blue (button), red (agent) and the cookie's own brown, measured on its picture.
    cookie = others["galleta"]
    cookie_colour = cookie[cookie.sum(axis=-1) > 150].mean(axis=0)
    forbidden = {"azul": (0, 0, 255), "rojo": (255, 0, 0), "galleta": cookie_colour}
    for colour, rgb in PALETTE.items():
        for name, ref in forbidden.items():
            assert np.linalg.norm(np.array(rgb) - np.array(ref)) > (100 if name == "galleta" else 140), (colour, name)
    # No circles or triangles: the silhouette of none of them overlaps much with the button's circle, the
    # cookie's disc or the agent's triangle (intersection over union of what each tile paints).
    def mask(picture):
        return picture[3:-3, 3:-3].sum(axis=-1) > 60  # inside the tile's grid lines

    def iou(a, b):
        return (a & b).sum() / (a | b).sum()

    for (shape, colour), picture in seen.items():
        for name, other in others.items():
            # A circle scores above 0.9 here; the most circular of the five shapes (the square) about 0.8.
            assert iou(mask(picture), mask(other)) < 0.85, (shape, colour, name)
    assert len({Obstacle(s, c).encode() for s, c in itertools.product(SHAPES, COLOURS)}) == len(SHAPES) * len(COLOURS)


def test_the_symbolic_id_is_new_and_the_experts_know_it():
    assert obstacles.OBSTACLE == expert.OBSTACLE == 13
    assert obstacles.OBSTACLE not in OBJECT_TO_IDX.values() and {11, 12}.isdisjoint({obstacles.OBSTACLE})
    world = World(seed=3)
    room, cell, shape, colour = world.env._placed[2]
    world.inner.agent_pos, world.inner.agent_dir = (cell[0] - 1, cell[1]), 0
    grid, visible = world.inner.gen_obs_grid(3)
    ids = grid.encode(visible)
    assert ids[1, 1, 0] == 13 and ids[1, 1, 2] == SHAPES.index(shape)  # the cell right ahead


def test_they_are_solid_and_the_bumps_are_counted():
    world = World(seed=3)
    env, inner = world.env, world.inner
    room, cell, shape, colour, before = facing(world)
    for _ in range(3):
        obs, *_ = env.step(2)  # forward, into it
        assert tuple(inner.agent_pos) == before and float(obs["log_bumps"]) == 1  # it does not move
    # At the start, with nothing ahead, forward does move it and is no bump.
    env.reset()
    start = tuple(inner.agent_pos)
    obs, *_ = env.step(2)
    assert tuple(inner.agent_pos) != start and float(obs["log_bumps"]) == 0
    # Turning in front of an obstacle is not a bump either.
    world2 = World(seed=3)
    facing(world2)
    obs, *_ = world2.env.step(0)
    assert float(obs["log_bumps"]) == 0


def test_the_agent_sees_them_in_its_image():
    # What it gets with the obstacle in front of it differs, in a patch of the obstacle's colour,
    # from what it gets with the same view and the obstacle taken away.
    for seed in range(6):
        world = World(seed=seed)
        room, cell, shape, colour, before = facing(world, index=seed % 3)
        world.env.step(1)  # turn right and back left: a fresh observation of the same view
        with_it = world.env.step(0)[0]["image"].astype(int)
        world.inner.grid.set(*cell, None)
        without = world.env.step(1)[0]["image"].astype(int)
        without = world.env.step(0)[0]["image"].astype(int)
        changed = np.abs(with_it - without).sum(axis=-1) > 60
        assert changed.sum() > 60, (seed, room, shape, colour, int(changed.sum()))  # many pixels differ
        # ... and the pixels that differ are, on average, the obstacle's colour (it replaced the black floor).
        assert np.abs(with_it[changed].mean(axis=0) - np.array(PALETTE[colour])).max() < 110, (shape, colour)


# ---- the experts around them ----
def play(world, steps=816):
    """Play with the expert through the environment; returns (cookies, bumps, bad poses, steps)."""
    env = world.env
    cookies = bumps = bad = n = 0
    done = False
    while not done and n < steps:
        action = env.expert_action()
        bad += env._expert.pose != world.real_pose()
        obs, reward, done, _ = env.step(action)
        n += 1
        cookies += int(reward > 0)
        bumps += int(obs.get("log_bumps", 0))
    return cookies, bumps, bad, n


@pytest.mark.parametrize("view,expert_view,low", [(3, 3, 15), (5, 5, 18), (3, 5, 18)])
def test_the_experts_play_whole_episodes_around_random_obstacles(view, expert_view, low):
    for seed in range(12):
        world = World(seed, view=view, expert_view=expert_view)
        cookies, bumps, bad, n = play(world)
        assert (bumps, bad, n) == (0, 0, 816), (seed, bumps, bad, n)
        assert cookies >= low, (seed, cookies)


def put(world, cells, shapes=itertools.cycle(SHAPES)):
    """Replace the episode's obstacles by obstacles at these cells (relative to the start)."""
    for _, cell, *_ in world.env._placed:
        world.inner.grid.set(*cell, None)
    world.env._placed = []
    for cell, shape in zip(cells, shapes):
        world.inner.put_obj(Obstacle(shape, "green"), world.origin[0] + cell[0], world.origin[1] + cell[1])


ADVERSARIAL = {
    "al fondo de la sala este, donde el experto 3x3 se para": [(6, 0)],
    "al fondo de la sala oeste": [(-6, 0)],
    "en la fila del medio, camino al fondo": [(4, 0), (-5, 0)],
    "justo a la puerta, a los lados": [(3, 1), (3, -1), (-3, 1), (-3, -1)],
    "delante del botón": [(0, -4)],
    "al lado del botón": [(1, -5), (-1, -5)],
    "rodeando el botón por tres lados": [(0, -4), (1, -5), (-1, -5)],
    "contra las esquinas de la galleta": [(6, -2), (6, 2), (-6, -2), (-6, 2)],
    "lo peor a la vez": [(6, 0), (-6, 0), (0, -4)],
    "en el camino de ida y vuelta": [(5, 0), (-5, 0), (0, -4)],
}


@pytest.mark.parametrize("view,expert_view", [(3, 3), (5, 5)])
@pytest.mark.parametrize("name", list(ADVERSARIAL))
def test_the_experts_get_around_obstacles_in_the_worst_places(name, view, expert_view):
    world = World(0, view=view, expert_view=expert_view)
    put(world, ADVERSARIAL[name])
    assert all(c in world.reach() for c in CORNERS)
    cookies, bumps, bad, n = play(world, steps=816)
    assert (bumps, bad) == (0, 0)
    assert cookies >= 10, (name, cookies)


def test_obstacles_come_from_the_environment_config():
    import torch
    from omegaconf import OmegaConf

    from envs import make_envs

    config = OmegaConf.create(dict(task="cookie_partial_hall5_view3", env_num=1, eval_episode_num=0, action_repeat=1,
                                   time_limit=2088, press_on_move=True, seed=0, device="cpu", size=[64, 64], obstacles=1))
    envs, _, obs_space, _ = make_envs(config)
    try:
        trans, _ = envs.step(torch.zeros(1, 3), torch.ones(1, dtype=torch.bool))
        assert "log_bumps" in trans.keys() and "log_bumps" in obs_space.spaces
    finally:
        for env in envs.envs:
            env.close()
