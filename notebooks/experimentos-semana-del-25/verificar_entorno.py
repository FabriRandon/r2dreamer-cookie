"""Validate the three colored obstacles without training or using a GPU."""

from collections import deque
import importlib.util
from itertools import product
from pathlib import Path
import sys


def figuras(raw):
    return tuple(
        (x, y, obj.shape, obj.color)
        for x in range(raw.width)
        for y in range(raw.height)
        if (obj := raw.grid.get(x, y)) is not None and hasattr(obj, "shape")
    )


def alcanzables(raw):
    seen, queue = {(8, 8)}, deque([(8, 8)])
    while queue:
        x, y = queue.popleft()
        for pos in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if pos in seen or not (0 <= pos[0] < raw.width and 0 <= pos[1] < raw.height):
                continue
            obj = raw.grid.get(*pos)
            if obj is None or obj.can_overlap():
                seen.add(pos)
                queue.append(pos)
    walkable = {
        (x, y) for x in range(raw.width) for y in range(raw.height)
        if (obj := raw.grid.get(x, y)) is None or obj.can_overlap()
    }
    assert seen == walkable, "Una figura deja casillas transitables aisladas"
    return seen


def verificar(module):
    assert module.OBSTACLE_LAYOUT == "three_colored_per_episode_v1", "Se cargó otra versión del entorno"
    corners = {(1, 6), (1, 10), (15, 6), (15, 10)}
    button_access = {(8, 2), (8, 4), (7, 3), (9, 3)}
    side_positions = [
        {(cx + dx, 8 + dy) for dx in range(-2, 3) for dy in (-2, -1, 1, 2)} - corners
        for cx in (3, 13)
    ]
    center_positions = {
        (8 + dx, 3 + dy) for dx in (-2, -1, 1, 2) for dy in range(-2, 3)
        if abs(dx) + abs(dy) > 1
    }
    assert all(len(positions) == 18 for positions in (*side_positions, center_positions))

    def secuencia(mode, view, seed):
        env = module.Cookie(f"partial_hall5_view{view}", obstacles=mode, seed=seed)
        layouts, cookies = [], []
        try:
            for _ in range(24):
                obs = env.reset()
                raw = env._env.unwrapped
                assert obs["image"].shape == (64, 64, 3)
                assert raw.agent_view_size == view
                layout = figuras(raw)
                assert len(layout) == (0 if mode == "none" else 3)
                if layout:
                    left, center, right = layout
                    assert left[:2] in side_positions[0] and right[:2] in side_positions[1]
                    assert center[:2] in center_positions
                    if mode == "distinct":
                        assert center[2:] == ("diamond", "yellow")
                        assert {left[2:], right[2:]} == {("square", "green"), ("triangle", "purple")}
                        assert len({row[2] for row in layout}) == len({row[3] for row in layout}) == 3
                    else:
                        assert all(row[2:] == ("square", "green") for row in layout)
                seen = alcanzables(raw)
                assert corners | button_access <= seen
                assert raw.grid.get(8, 3).__class__.__name__ == "Button"
                assert all(raw.grid.get(8, y) is None for y in range(4, 9))

                for action in (0, 1, 0, 1):
                    env.step(action)
                    assert figuras(raw) == layout
                # Every approach to the button remains usable.
                for pos, direction in (((8, 4), 3), ((8, 2), 1), ((7, 3), 0), ((9, 3), 2)):
                    raw.agent_pos, raw.agent_dir = pos, direction
                    obs, _, _, _ = env.step(module.FORWARD)
                    assert obs["log_presses"] == 1 and raw.cookie.cur_pos is not None
                    cookies.append(tuple(raw.cookie.cur_pos))
                    assert figuras(raw) == layout
                spawner = raw.spawner
                for corner in sorted(corners):
                    raw.spawner = lambda pos=corner: pos
                    raw.spawn_cookie()
                    x, y = corner
                    for direction, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
                        pos = (x - dx, y - dy)
                        if pos in seen and raw.grid.get(*pos) is None:
                            raw.agent_pos, raw.agent_dir = pos, direction
                            _, reward, _, _ = env.step(module.FORWARD)
                            assert reward == 1, "La galleta no se puede recoger"
                            break
                    else:
                        raise AssertionError("No hay acceso a una esquina")
                    assert figuras(raw) == layout
                raw.spawner = spawner
                layouts.append(layout)
        finally:
            env._env.close()
        return layouts, cookies

    for view in (3, 5):
        for seed in (0, 7):
            _, no_cookies = secuencia("none", view, seed)
            same, same_cookies = secuencia("same", view, seed)
            distinct, distinct_cookies = secuencia("distinct", view, seed)
            assert [tuple(row[:2] for row in layout) for layout in same] == [
                tuple(row[:2] for row in layout) for layout in distinct
            ], "same y distinct deben compartir geometría"
            assert no_cookies == same_cookies == distinct_cookies, "Cambió el sorteo de galletas"
            for room in range(3):
                assert len({layout[room][:2] for layout in distinct}) > 1
            for room in (0, 2):
                assert {layout[room][2] for layout in distinct} == {"square", "triangle"}
            assert secuencia("distinct", view, seed) == (distinct, distinct_cookies)
            print(f"OK: vista {view}×{view}, semilla {seed}, 24 episodios por condición y repetición")

    # Exhaust every possible placement, including combinations not drawn above.
    env = module.Cookie("partial_hall5_view5", obstacles="distinct", seed=0)
    try:
        env.reset()
        raw = env._env.unwrapped
        original = figuras(raw)
        objects = [raw.grid.get(*row[:2]) for row in original]
        for row in original:
            raw.grid.set(*row[:2], None)
        count = 0
        for positions in product(sorted(side_positions[0]), sorted(center_positions), sorted(side_positions[1])):
            for pos, obj in zip(positions, objects):
                raw.grid.set(*pos, obj)
            assert corners | button_access <= alcanzables(raw)
            for pos in positions:
                raw.grid.set(*pos, None)
            count += 1
        assert count == 18 ** 3
        print(f"OK: las {count} distribuciones posibles mantienen todas las casillas transitables conectadas")
    finally:
        env._env.close()
    print("Verificado: tres figuras, colores distintos, posiciones variables y objetivos accesibles.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).resolve()
        spec = importlib.util.spec_from_file_location("cookie_semana25", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from envs import cookie as module
    verificar(module)
