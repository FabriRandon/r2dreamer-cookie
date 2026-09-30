"""Record a video of a trained agent playing one episode of the cookie domain.

Loads the agent from the last checkpoint of a run and plays with its learned
policy, as in evaluation (the most likely action at every step), so it can be
watched long after training has finished:

    python3 watch.py --logdir ./logdir/run --out agent.mp4
"""

import argparse
import pathlib
import random
import sys
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from tensordict import TensorDict

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))

from dreamer import Dreamer  # noqa: E402
from envs import make_env  # noqa: E402

# Names of the actions the agent can take with env.press_on_move, which is how
# the cookie domain is set up by default.
ACTION_NAMES = ["izquierda", "derecha", "avanzar"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", required=True, type=pathlib.Path, help="run folder holding latest.pt")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="video file to write (.mp4)")
    parser.add_argument("--seed", type=int, default=0, help="changes where the cookies appear")
    parser.add_argument("--fps", type=int, default=30, help="frames per second of the video")
    parser.add_argument("--max-steps", type=int, default=None, help="stop earlier than the episode's time limit")
    return parser.parse_args()


def load_agent(logdir, device):
    # Hydra stores the run's config next to the checkpoint.
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    config.device = device
    config.model.compile = False  # compiling only pays off when training
    env = make_env(config.env, 0)
    agent = Dreamer(config.model, env.observation_space, env.action_space).to(device)
    checkpoint = torch.load(logdir / "latest.pt", map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    return config, env, agent, int(checkpoint["step"])


def to_batch(obs, device):
    """Shape one observation like the trainer does: a batch of one env."""
    data = {k: torch.as_tensor(np.asarray(v))[None] for k, v in obs.items()}
    data["reward"] = torch.zeros(1)
    for key, value in data.items():
        if value.ndim == 1:
            data[key] = value.unsqueeze(-1)
    return TensorDict(data, batch_size=(1,)).to(device)


def frame(grid, view, step, cookies, action, trained_steps, map_label, font):
    """The whole grid on the left, what the agent sees on the right."""
    height, width = grid.shape[:2]
    view = Image.fromarray(view).resize((height, height), Image.NEAREST)
    header, footer, gap = 22, 20, 8
    image = Image.new("RGB", (width + gap + height, header + height + footer))
    image.paste(Image.fromarray(grid), (0, header))
    image.paste(view, (width + gap, header))
    draw = ImageDraw.Draw(image)
    text = f"paso {step}   galletas {cookies}   jugada: {action}   (agente entrenado {trained_steps:,} pasos)"
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    draw.text((6, header + height + 3), map_label, fill=(180, 180, 180), font=font)
    draw.text((width + gap + 4, header + height + 3), "lo que recibe el agente", fill=(180, 180, 180), font=font)
    return np.asarray(image)


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, env, agent, trained_steps = load_agent(args.logdir, device)
    # The Cookie env under the wrappers draws the whole grid, highlighting the
    # part the agent can see.
    cookie = env.unwrapped
    cookie._env.unwrapped.tile_size = 16
    cookie._seed = args.seed
    # cookie-env picks the cookie's corner with Python's global random module.
    random.seed(args.seed)
    names = ACTION_NAMES if env.action_space.shape[0] == len(ACTION_NAMES) else None

    font = ImageFont.load_default(size=13)
    partial = type(cookie._env).__name__ == "RGBImgPartialObsWrapper"
    map_label = "el mapa (lo iluminado es lo que ve el agente)" if partial else "el mapa (el agente lo ve completo)"

    obs = env.reset()
    state = agent.get_initial_state(1)
    cookies, step, done, frames = 0, 0, False, []
    limit = args.max_steps or int(config.env.time_limit)
    while not done and step < limit:
        action, state = agent.act(to_batch(obs, device), state, eval=True)
        index = int(action[0].argmax())
        name = names[index] if names else index
        frames.append(frame(cookie.render(), obs["image"], step, cookies, name, trained_steps, map_label, font))
        obs, reward, done, _ = env.step(action[0].cpu().numpy())
        cookies += int(reward > 0)
        step += 1

    from moviepy.editor import ImageSequenceClip

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ImageSequenceClip(frames, fps=args.fps).write_videofile(str(args.out), codec="libx264", audio=False, logger=None)
    print(f"{step} pasos, {cookies} galletas. Video en {args.out}")


if __name__ == "__main__":
    main()
