import gymnasium as gym
import numpy as np
from PIL import Image

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


class Cookie(gym.Env):
    metadata = {}

    def __init__(self, task, size=(64, 64), seed=0):
        from cookie_env.envs import ThreeRooms
        from cookie_env.utils import spawner
        from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper

        # "partial" or "partial_18x29": the optional suffix overrides the grid.
        variant, _, grid = task.partition("_")
        assert variant in VARIANTS, (variant, tuple(VARIANTS))
        options = dict(VARIANTS[variant])
        full_obs = options.pop("full_obs")
        if "spawner" in options:
            options["cookie_spawner"] = getattr(spawner, options.pop("spawner"))
        if grid:
            height, width = (int(side) for side in grid.split("x"))
            options.update(height=height, width=width)

        # max_steps=None picks MiniGrid's 4 * height * width, the episode
        # length the JAX runs used; ThreeRooms would default to 10_000.
        env = ThreeRooms(render_mode="rgb_array", max_steps=None, **options)
        # MiniGrid observations are symbolic; the world model wants images.
        self._env = (RGBImgObsWrapper if full_obs else RGBImgPartialObsWrapper)(env)
        self._size = tuple(size)
        self._seed = seed
        self._seeded = False

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, self._size + (3,), np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(self._env.action_space.n)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        obs = {
            "image": self._image(obs),
            "is_first": False,
            "is_last": done,
            "is_terminal": terminated,
        }
        return obs, np.float32(reward), done, info

    def reset(self):
        # Gymnasium seeds through reset(), so only the first one is seeded;
        # seeding every episode would replay the same cookie layout forever.
        obs, _ = self._env.reset(seed=None if self._seeded else self._seed)
        self._seeded = True
        return {"image": self._image(obs), "is_first": True, "is_last": False, "is_terminal": False}

    def render(self):
        return self._env.render()

    def _image(self, obs):
        image = obs["image"]
        if image.shape[:2] != self._size:
            image = Image.fromarray(image).resize((self._size[1], self._size[0]), Image.BILINEAR)
            image = np.asarray(image, dtype=np.uint8)
        return image
