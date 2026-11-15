"""Watch the hand-written expert solve the cookie domain.

    python demo_expert.py                                 # the 5x5 expert, 2 cycles -> expert_demo/
    python demo_expert.py --expert-view 3                 # the 3x3 expert, as it plays -> expert_demo_3x3/
    python demo_expert.py --expert-view 3 --showcase      # the 4 ways a cycle can go, one after another
    python demo_expert.py --expert-view 3 --stats 40      # only numbers, 40 full episodes

Writes <out>/expert.mp4 and expert.gif (the whole grid with what the agent sees lit up and
outlined in yellow, and on top the step, cookies, what the expert is doing and its action) and
expert_keyframes.png (one frame per step of the first cycle, as a sheet). With --showcase the
files are called expert_showcase.*: the expert's random choices and the cookie's corner are
forced so that each of the four cycles shows one case.
"""

import argparse
import collections
import pathlib
import random
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.cookie import Cookie
from envs.expert import make_expert

ACTIONS = {0: "girar a la izquierda", 1: "girar a la derecha", 2: "avanzar"}


def symbolic_view(env, view):
    """The `view` x `view` view of object ids the expert decides from, before it is drawn as an image."""
    grid, visible = env._env.unwrapped.gen_obs_grid(view)
    return grid.encode(visible)


def true_pose(env, origin):
    """The simulator's pose relative to the starting cell, to check the policy's own."""
    inner = env._env.unwrapped
    (x, y), heading = inner.agent_pos, inner.agent_dir
    return int(x) - origin[0], int(y) - origin[1], int(heading)


class Script:
    """Stands in for the expert's random generator, so a demo can show every case.

    `rooms` holds, for each cycle, the order in which the expert looks into the side
    rooms (-1 west, 1 east); `verticals` the order of the corners ("up", "down") of each
    room it looks into, in the order it gets there."""

    def __init__(self, rooms, verticals, seed=0):
        self.rooms, self.verticals = [list(r) for r in rooms], [list(v) for v in verticals]
        self._random = random.Random(seed)  # once the script is over, the choices are random again

    def sample(self, population, k):
        queue = self.rooms if tuple(population) == (-1, 1) else self.verticals
        return queue.pop(0) if queue else self._random.sample(population, k)


def run_episode(env, policy, steps, view, check=True, forced=(), cases=()):
    """Run one episode and return its frames and a log of every step.

    `forced` are the corners (relative to the start) where the cookie is put after the
    1st, 2nd... press, instead of the random ones; `cases` their descriptions."""
    env.reset()
    policy.reset()
    inner = env._env.unwrapped
    origin = tuple(int(v) for v in inner.agent_pos)
    frames, log, cookies, presses = [env.render()], [], 0, 0
    for t in range(steps):
        action = policy.act(symbolic_view(env, view))
        if check:
            # Believed vs real pose, before acting.
            assert policy.pose == true_pose(env, origin), (t, policy.pose, true_pose(env, origin))
        phase, status = policy.phase, policy.status
        obs, reward, done, _ = env.step(action)
        cookies += int(reward > 0)
        pressed = bool(obs["log_presses"])
        case = cases[presses - 1] if 0 < presses <= len(cases) else ""
        if pressed:
            presses += 1
            if presses <= len(forced):
                # The environment dropped the cookie in a random corner; use the one asked for.
                inner._remove_cookie()
                inner.put_obj(inner.cookie, origin[0] + forced[presses - 1][0], origin[1] + forced[presses - 1][1])
        log.append(dict(t=t + 1, action=action, phase=phase, status=status, reward=float(reward), pressed=pressed,
                        cookies=cookies, case=case))
        frames.append(env.render())
        if done:
            break
    return frames, log


def font(size=14):
    """A font with accents (the captions are in Spanish): DejaVu Sans, which the system or
    matplotlib has; PIL's default one draws boxes for the accented letters."""
    for name in ("DejaVuSans.ttf",):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        import os

        import matplotlib

        return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def annotate(frame, entry, header=44):
    """The frame with a caption above it: the case (if any), the step and what the expert does."""
    h, w = frame.shape[:2]
    out = Image.new("RGB", (w, h + header), (20, 20, 20))
    out.paste(Image.fromarray(frame), (0, header))
    draw = ImageDraw.Draw(out)
    if entry is None:
        lines = ["", "inicio"]
    else:
        extra = "   ¡apretó el botón!" if entry["pressed"] else ("   +1 galleta" if entry["reward"] > 0 else "")
        lines = [entry["case"], f"paso {entry['t']:3d}   galletas {entry['cookies']}   {entry['status']}   [{ACTIONS[entry['action']]}]{extra}"]
    draw.text((8, 4), lines[0], fill=(255, 200, 0), font=font(14))
    draw.text((8, 24), lines[1], fill=(240, 240, 240), font=font(13))
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


# The four ways a cycle of the 3x3 expert can go, with the corner the cookie is put in, the
# order in which the expert looks into the rooms and at the corners of each room it enters.
SHOWCASE = [
    dict(case="Caso 1 de 4: la galleta está en la primera esquina que revisa", corner=(7, -2), rooms=(1, -1), verticals=[("up", "down")]),
    dict(case="Caso 2 de 4: está en la segunda esquina de la misma sala", corner=(7, 2), rooms=(1, -1), verticals=[("up", "down")]),
    dict(case="Caso 3 de 4: la primera sala está vacía; está en la otra", corner=(-7, -2), rooms=(1, -1), verticals=[("up", "down"), ("up", "down")]),
    dict(case="Caso 4 de 4: sala vacía y, en la otra, en la segunda esquina", corner=(7, 2), rooms=(-1, 1), verticals=[("down", "up"), ("up", "down")]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expert-view", type=int, default=5, choices=[3, 5], help="cells per side of the view the expert decides from")
    parser.add_argument("--task", default=None, help="Cookie task (hallways of 5). Default: partial_hall5_view<expert-view>")
    parser.add_argument("--steps", type=int, default=None, help="steps to record (default 75 for 5x5, 400 for 3x3)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="folder (default expert_demo, or expert_demo_3x3)")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--showcase", action="store_true", help="3x3 only: force the 4 ways a cycle can go, one per cycle")
    parser.add_argument("--stats", type=int, default=0, help="run this many full episodes and print statistics, nothing else")
    parser.add_argument("--time-limit", type=int, default=2088)
    args = parser.parse_args()
    view = args.expert_view
    assert not args.showcase or view == 3, "--showcase is for the 3x3 expert"

    env = Cookie(args.task or f"partial_hall5_view{view}", (64, 64), seed=args.seed, expert_view=view)
    policy = make_expert(view, hallway=5, seed=args.seed)

    if args.stats:
        totals, gaps, lengths = [], [], set()
        for _ in range(args.stats):
            _, log = run_episode(env, policy, args.time_limit, view)
            totals.append(log[-1]["cookies"])
            lengths.add(len(log))
            gaps += cycles(log)
        # The environment ends the episode before --time-limit (816 steps on this map).
        print(f"{args.stats} episodes of {'/'.join(str(n) for n in sorted(lengths))} steps")
        print(f"cookies per episode: mean {statistics.mean(totals):.2f}, min {min(totals)}, max {max(totals)}")
        print(f"steps per cycle: mean {statistics.mean(gaps):.2f}, min {min(gaps)}, max {max(gaps)}")
        print("cycle lengths (steps: how many): " + ", ".join(f"{k}: {v}" for k, v in sorted(collections.Counter(gaps).items())))
        return

    out = pathlib.Path(args.out or ("expert_demo" if view == 5 else "expert_demo_3x3"))
    out.mkdir(exist_ok=True)
    kwargs, name = {}, "expert"
    if args.showcase:
        policy._rng = Script([c["rooms"] for c in SHOWCASE], [v for c in SHOWCASE for v in c["verticals"]])
        kwargs = dict(forced=[c["corner"] for c in SHOWCASE], cases=[c["case"] for c in SHOWCASE])
        name = "expert_showcase"
    steps = args.steps or (75 if view == 5 else 400)
    frames, log = run_episode(env, policy, steps, view, **kwargs)
    if args.showcase:
        # Stop a moment after the 4th cycle ends, and not later.
        presses = [e["t"] for e in log if e["pressed"]]
        end = min(len(log), presses[len(SHOWCASE)] + 3) if len(presses) > len(SHOWCASE) else len(log)
        frames, log = frames[: end + 1], log[:end]
    captioned = [annotate(frames[0], None)] + [annotate(f, e) for f, e in zip(frames[1:], log)]
    import imageio.v2 as imageio

    fps = args.fps or (4 if view == 5 else 6)
    imageio.mimsave(out / f"{name}.mp4", captioned, fps=fps, macro_block_size=1)
    imageio.mimsave(out / f"{name}.gif", captioned, duration=1000 / fps, loop=0)
    save_keyframes(frames, log, out / f"{name}_keyframes.png")
    print(f"{len(log)} steps, {log[-1]['cookies']} cookies, cycles of {cycles(log)} steps")
    print(f"wrote {out}/{name}.mp4, {name}.gif and {name}_keyframes.png")


if __name__ == "__main__":
    main()
