import random

import gymnasium as gym
import numpy as np
from PIL import Image

# MiniGrid's action indices for moving forward and for toggling the object in
# front of the agent.
FORWARD, TOGGLE = 2, 5

# Variants of the cookie domain (cookie_env.envs.ThreeRooms), mirroring the
# ones used in the JAX DreamerV3 study this fork replicates. "full" shows the
# whole grid; the others expose only the 3x3 patch in front of the agent, which
# is the partially observable setting Dreamer struggles with.
VARIANTS = {
    "partial": dict(full_obs=False),
    "deterministic": dict(full_obs=False, spawner="deterministic_corner"),
    "norespawn": dict(full_obs=False, respawn=False),
    "full": dict(full_obs=True),
    "fullfixed": dict(full_obs=True, respawn=False),
}



def _shorter_hallways(hallway, fixed_corner=False):
    """ThreeRooms with hallways of `hallway` cells instead of 11.

    The three 5x5 rooms stay as they are, and the cookie still appears in an
    outer corner of a side room, so only the distances shrink. cookie-env
    hardcodes the corners for the 18x29 grid, which is why just shrinking the
    grid lands the cookie in the wrong places. Returns the class and the
    arguments to build it with.
    """
    from cookie_env.envs import ThreeRooms
    from cookie_env.objects import Button
    from minigrid.core.grid import Grid

    # The hallways meet where the agent starts; the button's room sits
    # `hallway` cells above, and the side rooms as far to each side.
    cx = cy = hallway + 3
    corners = [(x, y) for x in (cx - hallway - 2, cx + hallway + 2) for y in (cy - 2, cy + 2)]

    class Rooms(ThreeRooms):
        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            self._fill_with_walls()
            for x, y in ((cx - hallway, cy), (cx + hallway, cy), (cx, cy - hallway)):
                self._generate_room(x, y)
            self._generate_hallway(cx - hallway, cy, cx + hallway, cy)
            self._generate_hallway(cx, cy - hallway, cx, cy)
            self.put_obj(Button("blue", self.spawn_cookie), cx, cy - hallway)
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir

    # cookie-env picks the corner with Python's global random, like this.
    spawner = (lambda: corners[0]) if fixed_corner else (lambda: random.choice(corners))
    options = dict(width=2 * hallway + 7, height=hallway + 7, agent_start_pos=(cx, cy), cookie_spawner=spawner)
    return Rooms, options


# Per-step signals the trainer adds up over each episode and logs as
# episode/<name>, so a run shows progress long before the agent gets cookies:
# button presses, distinct cells visited, and steps spent in the button's room.
LOGS = ("presses", "cells", "button_room")


class Cookie(gym.Env):
    metadata = {}

    def __init__(self, task, size=(64, 64), seed=0, press_on_move=True):
        from cookie_env.envs import ThreeRooms
        from cookie_env.objects import Button
        from cookie_env.utils import spawner
        from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper

        # "partial", "partial_18x29" or "partial_hall5": the optional suffix
        # overrides the grid, or shortens the hallways (see _shorter_hallways).
        variant, _, suffix = task.partition("_")
        assert variant in VARIANTS, (variant, tuple(VARIANTS))
        options = dict(VARIANTS[variant])
        full_obs = options.pop("full_obs")
        rooms = ThreeRooms
        if suffix.startswith("hall"):
            fixed_corner = options.pop("spawner", None) == "deterministic_corner"
            rooms, layout = _shorter_hallways(int(suffix[len("hall"):]), fixed_corner)
            options.update(layout)
        else:
            if "spawner" in options:
                options["cookie_spawner"] = getattr(spawner, options.pop("spawner"))
            if suffix:
                height, width = (int(side) for side in suffix.split("x"))
                options.update(height=height, width=width)

        # max_steps=None picks MiniGrid's 4 * height * width, the episode
        # length the JAX runs used; ThreeRooms would default to 10_000.
        env = rooms(render_mode="rgb_array", max_steps=None, **options)
        # MiniGrid observations are symbolic; the world model wants images.
        self._env = (RGBImgObsWrapper if full_obs else RGBImgPartialObsWrapper)(env)
        # With press_on_move the agent only turns and moves, and moving into
        # the button presses it, so it no longer needs the other four MiniGrid
        # actions, which a random policy mostly wastes steps on.
        self._press_on_move = press_on_move
        self._button = Button
        self._size = tuple(size)
        self._seed = seed
        self._seeded = False
        self._visited = set()
        self._button_pos = None

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, self._size + (3,), np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            **{f"log_{name}": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32) for name in LOGS},
        })

    @property
    def action_space(self):
        if self._press_on_move:
            return gym.spaces.Discrete(3)
        return gym.spaces.Discrete(self._env.action_space.n)

    def step(self, action):
        env = self._env.unwrapped
        facing_button = isinstance(env.grid.get(*env.front_pos), self._button)
        if self._press_on_move and action == FORWARD and facing_button:
            action = TOGGLE
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        cell = tuple(int(v) for v in env.agent_pos)
        new_cell = cell not in self._visited
        self._visited.add(cell)
        obs = {
            "image": self._image(obs),
            "is_first": False,
            "is_last": done,
            "is_terminal": terminated,
            **self._logs(presses=action == TOGGLE and facing_button, cells=new_cell, button_room=self._in_room(cell)),
        }
        return obs, np.float32(reward), done, info

    def reset(self):
        # Gymnasium seeds through reset(), so only the first one is seeded;
        # seeding every episode would replay the same cookie layout forever.
        if not self._seeded:
            # cookie-env picks the cookie's corner with Python's global random,
            # which the env worker processes would otherwise seed from the OS.
            random.seed(self._seed)
        obs, _ = self._env.reset(seed=None if self._seeded else self._seed)
        self._seeded = True
        env = self._env.unwrapped
        cell = tuple(int(v) for v in env.agent_pos)
        self._visited = {cell}
        self._button_pos = next(
            ((x, y) for x in range(env.width) for y in range(env.height) if isinstance(env.grid.get(x, y), self._button)),
            None,
        )
        return {
            "image": self._image(obs),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            # The starting cell counts as visited.
            **self._logs(presses=False, cells=True, button_room=self._in_room(cell)),
        }

    def render(self):
        return self._env.render()

    def _in_room(self, cell, radius=2):
        # ThreeRooms builds each room as the 5x5 square around its center, and
        # the button sits at the center of its room.
        if self._button_pos is None:
            return False
        return max(abs(cell[0] - self._button_pos[0]), abs(cell[1] - self._button_pos[1])) <= radius

    @staticmethod
    def _logs(**values):
        return {f"log_{name}": np.float32(values[name]) for name in LOGS}

    def _image(self, obs):
        image = obs["image"]
        if image.shape[:2] != self._size:
            image = Image.fromarray(image).resize((self._size[1], self._size[0]), Image.BILINEAR)
            image = np.asarray(image, dtype=np.uint8)
        return image
