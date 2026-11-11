"""Watch the hand-written expert solve the cookie domain.

    python demo_expert.py                       # 2 cycles, writes expert_demo/
    python demo_expert.py --steps 300 --seed 3  # a longer episode
    python demo_expert.py --stats 200           # only numbers, 200 episodes of the time limit

Writes expert_demo/expert.mp4 and expert.gif (the whole grid with the agent's
5x5 view shaded, and the step, phase, action and cookies on top) and
expert_keyframes.png (one frame per step of the first cycle, as a sheet).
"""

import argparse
import pathlib
import statistics

import numpy as np
from PIL import Image, ImageDraw

from envs.cookie import Cookie
from envs.expert import EAST, START, ExpertPolicy

ACTIONS = {0: "left", 1: "right", 2: "forward"}


def symbolic_view(env):
    """The 5x5 view of object ids the agent gets, before it is drawn as an image."""
    return env._env.unwrapped.gen_obs()["image"]


def true_pose(env, origin):
    """The simulator's pose relative to the starting cell, to check the policy's own."""
    inner = env._env.unwrapped
    (x, y), heading = inner.agent_pos, inner.agent_dir
    return int(x) - origin[0], int(y) - origin[1], int(heading)


def run_episode(env, policy, steps, check=True):
    """Run one episode and return its frames and a log of every step."""
    env.reset()
    policy.reset()
    inner = env._env.unwrapped
    origin = tuple(int(v) for v in inner.agent_pos)
    frames, log, cookies = [env.render()], [], 0
    for t in range(steps):
        view = symbolic_view(env)
        action = policy.act(view)
        if check:
            # Believed vs real pose, before acting.
            assert policy.pose == true_pose(env, origin), (t, policy.pose, true_pose(env, origin))
        phase = policy.phase
        obs, reward, done, _ = env.step(action)
        cookies += int(reward > 0)
        log.append(dict(t=t + 1, action=action, phase=phase, reward=float(reward), pressed=bool(obs["log_presses"]), cookies=cookies))
        frames.append(env.render())
        if done:
            break
    return frames, log


def annotate(frame, entry):
    """The frame with a one-line caption above it."""
    h, w = frame.shape[:2]
    out = Image.new("RGB", (w, h + 26), (20, 20, 20))
    out.paste(Image.fromarray(frame), (0, 26))
    if entry is None:
        text = "start"
    else:
        press = "  PRESS" if entry["pressed"] else ""
        text = f"step {entry['t']:3d}  {entry['phase']:<6}  {ACTIONS[entry['action']]:<7} cookies {entry['cookies']}{press}"
        if entry["reward"] > 0:
            text += "  +1"
    ImageDraw.Draw(out).text((8, 7), text, fill=(240, 240, 240))
    return np.asarray(out)


def cycles(log):
    """Steps of each button-to-button cycle: [first press, second press, ...] gaps."""
    presses = [e["t"] for e in log if e["pressed"]]
    return [b - a for a, b in zip(presses, presses[1:])]


def save_keyframes(frames, log, path, columns=6):
    """One captioned frame per step up to the end of the first cycle."""
    presses = [e["t"] for e in log if e["pressed"]]
    last = presses[1] if len(presses) > 1 else len(log)
    shots = [annotate(frames[t + 1], log[t]) for t in range(last)]
    shots = [Image.fromarray(s).resize((s.shape[1] // 2, s.shape[0] // 2), Image.LANCZOS) for s in shots]
    w, h = shots[0].size
    rows = -(-len(shots) // columns)
    sheet = Image.new("RGB", (columns * w, rows * h), (0, 0, 0))
    for k, shot in enumerate(shots):
        sheet.paste(shot, ((k % columns) * w, (k // columns) * h))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="partial_hall5_view5", help="Cookie task (the hallway length must be 5 for now)")
    parser.add_argument("--steps", type=int, default=75, help="steps to record")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="expert_demo")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--stats", type=int, default=0, help="run this many full episodes and print statistics, nothing else")
    parser.add_argument("--time-limit", type=int, default=2088)
    args = parser.parse_args()

    env = Cookie(args.task, (64, 64), seed=args.seed)
    policy = ExpertPolicy(hallway=5, seed=args.seed)

    if args.stats:
        totals, gaps = [], []
        for _ in range(args.stats):
            _, log = run_episode(env, policy, args.time_limit)
            totals.append(log[-1]["cookies"])
            gaps += cycles(log)
        print(f"{args.stats} episodes of {args.time_limit} steps")
        print(f"cookies per episode: mean {statistics.mean(totals):.2f}, min {min(totals)}, max {max(totals)}")
        print(f"steps per cycle: mean {statistics.mean(gaps):.2f}, min {min(gaps)}, max {max(gaps)}")
        return

    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)
    frames, log = run_episode(env, policy, args.steps)
    captioned = [annotate(frames[0], None)] + [annotate(f, e) for f, e in zip(frames[1:], log)]
    import imageio.v2 as imageio

    imageio.mimsave(out / "expert.mp4", captioned, fps=args.fps, macro_block_size=1)
    imageio.mimsave(out / "expert.gif", captioned, duration=1000 / args.fps, loop=0)
    save_keyframes(frames, log, out / "expert_keyframes.png")
    print(f"{len(log)} steps, {log[-1]['cookies']} cookies, cycles of {cycles(log)} steps")
    print(f"wrote {out}/expert.mp4, expert.gif and expert_keyframes.png")


if __name__ == "__main__":
    main()
