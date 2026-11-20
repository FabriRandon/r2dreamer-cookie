import random

import gymnasium as gym
import numpy as np
from PIL import Image

# MiniGrid's action indices for moving forward and for toggling the object in
# front of the agent.
FORWARD, TOGGLE = 2, 5

# Colour of the outline drawn around what the agent sees in the videos of the whole grid.
VIEW_OUTLINE = (255, 200, 0)

# Variants of the cookie domain (cookie_env.envs.ThreeRooms), mirroring the
# ones used in the JAX DreamerV3 study this fork replicates. "full" shows the
# whole grid; the others expose only the patch in front of the agent, which is
# the partially observable setting Dreamer struggles with. The patch is 3x3, or
# NxN when the task ends in "_viewN".
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
# With obstacles there is one more: the times it walked into one.
LOGS = ("presses", "cells", "button_room")
OBSTACLE_LOGS = ("bumps",)


class Cookie(gym.Env):
    metadata = {}

    def __init__(self, task, size=(64, 64), seed=0, press_on_move=True, expert_view=5, obstacles=0):
        from cookie_env.envs import ThreeRooms
        from cookie_env.objects import Button
        from cookie_env.utils import spawner
        from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper

        # "partial", "partial_18x29" or "partial_hall5": the optional suffix
        # overrides the grid, or shortens the hallways (see _shorter_hallways).
        # Any of them can end in "_view5" to see 5x5 cells instead of 3x3.
        rest, _, last = task.rpartition("_")
        view = 3
        if last.startswith("view"):
            task, view = rest, int(last[len("view"):])
        # The agent sits at the middle of the view's back edge.
        assert view % 2 == 1 and view >= 3, f"the view must be odd and at least 3, got {view}"
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
        # ThreeRooms fixes the view at 3x3 when it builds MiniGrid, which reads
        # it again on every step.
        env.agent_view_size = view
        env.observation_space["image"] = gym.spaces.Box(0, 255, (view, view, 3), np.uint8)
        # MiniGrid observations are symbolic; the world model wants images.
        self._env = (RGBImgObsWrapper if full_obs else RGBImgPartialObsWrapper)(env)
        self._full_obs = full_obs
        # With press_on_move the agent only turns and moves, and moving into
        # the button presses it, so it no longer needs the other four MiniGrid
        # actions, which a random policy mostly wastes steps on.
        self._press_on_move = press_on_move
        # Obstacles (envs/obstacles.py): a number per room, drawn again at every reset, or one of the fixed
        # layouts of the first experiments ("same", "distinct"). They need the map with hallways.
        from envs.obstacles import parse

        self._obstacles, self._fixed_obstacles = parse(obstacles)
        assert not (self._obstacles or self._fixed_obstacles) or suffix.startswith("hall"), (
            "obstacles are placed on the map with hallways (hallN)"
        )
        self._obstacle_rng = None
        self._placed = []  # (room, cell, shape, colour) of the obstacles of the current episode
        self._log_names = LOGS + (OBSTACLE_LOGS if self._obstacles or self._fixed_obstacles else ())
        self._button = Button
        self._size = tuple(size)
        self._seed = seed
        self._seeded = False
        self._visited = set()
        self._button_pos = None
        self._hallway = int(suffix[len("hall"):]) if suffix.startswith("hall") else 5
        self._view = view
        # The hand-written expert (envs/expert.py), built when first asked.
        self._expert = None
        # The expert decides from a view of this many cells per side: 5 is the original
        # one, 3 the one for an agent that only sees 3x3 (it has to walk to see a corner).
        self._expert_view = int(expert_view)
        self._expert_action = None

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, self._size + (3,), np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            **{f"log_{name}": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32) for name in self._log_names},
        })

    @property
    def action_space(self):
        if self._press_on_move:
            return gym.spaces.Discrete(3)
        return gym.spaces.Discrete(self._env.action_space.n)

    def expert_action(self):
        """The action of the hand-written perfect policy for the current state.

        The expert decides from a symbolic view of its own (the one MiniGrid builds before
        it is drawn as an image) of `expert_view` x `expert_view` cells, whatever the
        agent's view is. It remembers where it is from the actions it asked for, so the
        caller must take exactly the action returned, at every step of the episode
        (step() checks it). A new episode restarts it.
        """
        if self._expert is None:
            from envs.expert import make_expert

            assert self._press_on_move, "the expert presses the button by moving into it (press_on_move)"
            assert not self._fixed_obstacles, "the experts know the random obstacles (env.obstacles=N), not the fixed ones"
            assert self._hallway == 5, f"the expert knows the map with hallways of 5 cells, not {self._hallway}"
            self._expert = make_expert(self._expert_view, hallway=self._hallway, seed=self._seed)
            self._expert.reset()
        env = self._env.unwrapped
        grid, visible = env.gen_obs_grid(self._expert_view)
        self._expert_action = self._expert.act(grid.encode(visible))
        return self._expert_action

    def step(self, action):
        if self._expert_action is not None and int(action) != self._expert_action:
            raise RuntimeError(
                f"the expert asked for action {self._expert_action} but {int(action)} was taken; "
                "its memory of where it is would be wrong"
            )
        self._expert_action = None
        env = self._env.unwrapped
        facing_button = isinstance(env.grid.get(*env.front_pos), self._button)
        if self._press_on_move and action == FORWARD and facing_button:
            action = TOGGLE
        bumped = False
        if (self._obstacles or self._fixed_obstacles) and action == FORWARD:
            from envs.obstacles import SOLID

            bumped = isinstance(env.grid.get(*env.front_pos), SOLID)  # solid: it stays where it is
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
            **self._logs(presses=action == TOGGLE and facing_button, cells=new_cell, button_room=self._in_room(cell), bumps=bumped),
        }
        return obs, np.float32(reward), done, info

    def reset(self):
        # Gymnasium seeds through reset(), so only the first one is seeded;
        # seeding every episode would replay the same cookie layout forever.
        if not self._seeded:
            # cookie-env picks the cookie's corner with Python's global random,
            # which the env worker processes would otherwise seed from the OS.
            random.seed(self._seed)
            # Whoever reseeds the env (watch.py, the explorer: _seeded = False) gets the same obstacles too.
            self._obstacle_rng = None
        obs, _ = self._env.reset(seed=None if self._seeded else self._seed)
        self._seeded = True
        self._expert_action = None
        if self._expert is not None:
            self._expert.reset()
        env = self._env.unwrapped
        if self._fixed_obstacles:
            from envs import obstacles

            obstacles.place_fixed(env, self._fixed_obstacles, self._hallway)
            obs = self._env.observation(env.gen_obs())  # the picture MiniGrid made at reset lacks them
        if self._obstacles:
            from envs import obstacles

            if self._obstacle_rng is None:
                # Its own generator (not the global random of the cookie's corner, nor MiniGrid's),
                # seeded with the env's seed: a run is reproducible, and every episode is different.
                self._obstacle_rng = np.random.default_rng([int(self._seed), 0x0B57AC1E])
            self._placed = obstacles.place(env, self._obstacle_rng, self._hallway, self._obstacles)
            # The picture MiniGrid made when it reset does not have them yet.
            obs = self._env.observation(env.gen_obs())
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
        # The grid for people to watch, with the agent's view shaded. With full
        # vision it sees the whole grid, so nothing is shaded, although the
        # image the agent gets still has MiniGrid's shading of the view.
        env = self._env.unwrapped
        return self.frame(env.tile_size)

    def frame(self, tile_size=16):
        """The whole grid as an image of (height * tile_size, width * tile_size) pixels,
        with what the agent sees lit up (the view_size x view_size cells in front of
        it, turned like the agent). Used for the training and evaluation videos."""
        env = self._env.unwrapped
        frame = env.get_frame(highlight=not self._full_obs, tile_size=tile_size)
        if not self._full_obs:
            # MiniGrid only lightens the cells in view, which is faint on the black floor:
            # outline the whole view window as well (the part inside the grid).
            left, top, right, bottom = env.get_view_exts()
            x0, y0 = max(left, 0) * tile_size, max(top, 0) * tile_size
            x1, y1 = min(right, env.width) * tile_size, min(bottom, env.height) * tile_size
            edge = max(1, tile_size // 8)
            frame = frame.copy()
            for rows, cols in ((slice(y0, y0 + edge), slice(x0, x1)), (slice(y1 - edge, y1), slice(x0, x1)),
                               (slice(y0, y1), slice(x0, x0 + edge)), (slice(y0, y1), slice(x1 - edge, x1))):
                frame[rows, cols] = VIEW_OUTLINE
        return frame

    def _in_room(self, cell, radius=2):
        # ThreeRooms builds each room as the 5x5 square around its center, and
        # the button sits at the center of its room.
        if self._button_pos is None:
            return False
        return max(abs(cell[0] - self._button_pos[0]), abs(cell[1] - self._button_pos[1])) <= radius

    def _logs(self, **values):
        return {f"log_{name}": np.float32(values.get(name, 0)) for name in self._log_names}

    def _image(self, obs):
        image = obs["image"]
        if image.shape[:2] != self._size:
            image = Image.fromarray(image).resize((self._size[1], self._size[0]), Image.BILINEAR)
            image = np.asarray(image, dtype=np.uint8)
        return image
