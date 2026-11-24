"""Check the November 24 variant, without training or using a GPU.

From the repository root:
    .venv/bin/python notebooks/experimentos-semana-del-24/verificar_entorno.py \
        notebooks/experimentos-semana-del-24/cookie.py

Without a file argument, check envs.cookie in the isolated Colab checkout.
"""

from collections import deque
import importlib.util
from pathlib import Path
import sys


def figuras(raw):
    return tuple(
        (x, y, obj.triangle)
        for x in range(raw.width)
        for y in range(raw.height)
        if (obj := raw.grid.get(x, y)) is not None and hasattr(obj, "triangle")
    )


def verificar(module):
    assert module.OBSTACLE_LAYOUT == "random_per_episode_v1", "Se cargó otra versión del entorno"
    corners = {(1, 6), (1, 10), (15, 6), (15, 10)}

    def secuencia(mode, view, seed):
        env = module.Cookie(f"partial_hall5_view{view}", obstacles=mode, seed=seed)
        layouts, cookies = [], []
        try:
            for _ in range(24):
                obs = env.reset()
                raw = env._env.unwrapped
                assert obs["image"].shape == (64, 64, 3)
                assert raw.agent_view_size == view
                assert {"log_cookie_visible", "log_cookie_in_front"} <= obs.keys()
                layout = figuras(raw)
                assert len(layout) == (0 if mode == "none" else 2)
                if layout:
                    assert 1 <= layout[0][0] <= 5 and 11 <= layout[1][0] <= 15
                    assert all(6 <= y <= 10 and y != 8 for _, y, _ in layout)
                    assert all((x, y) not in corners for x, y, _ in layout)
                    assert sum(triangle for _, _, triangle in layout) == (mode == "distinct")

                # Reachability of the tile facing the button and every corner.
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
                assert corners | {(8, 4)} <= seen, "Una figura impide llegar al objetivo"

                # Turning and pressing the button must not regenerate figures.
                for action in (0, 1, 0, 1):
                    env.step(action)
                    assert figuras(raw) == layout
                raw.agent_pos, raw.agent_dir = (8, 4), 3
                env.step(module.FORWARD)
                assert raw.cookie.cur_pos is not None
                assert figuras(raw) == layout
                for _ in range(4):
                    raw.spawn_cookie()
                    cookies.append(tuple(raw.cookie.cur_pos))
                    assert tuple(raw.cookie.cur_pos) in corners
                    assert figuras(raw) == layout

                # Every corner still accepts a cookie that the agent can collect.
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
            assert len(set(same)) > 1 and len(set(distinct)) > 1, "Las figuras siguen fijas"
            assert [tuple((x, y) for x, y, _ in row) for row in same] == [
                tuple((x, y) for x, y, _ in row) for row in distinct
            ], "same y distinct deben compartir geometría"
            assert no_cookies == same_cookies == distinct_cookies, "Las figuras alteraron el sorteo de galletas"
            for room in (0, 1):
                assert len({row[room][:2] for row in distinct}) > 1
                assert {row[room][2] for row in distinct} == {False, True}, "La forma identifica siempre el mismo lado"
            assert secuencia("distinct", view, seed) == (distinct, distinct_cookies), "La semilla no es reproducible"
            print(f"OK: vista {view}×{view}, semilla {seed}, 24 episodios por condición y repetición")
    print("Verificado: ubicaciones variables entre episodios, estables dentro de cada episodio y objetivos accesibles.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).resolve()
        spec = importlib.util.spec_from_file_location("cookie_semana24", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from envs import cookie as module
    verificar(module)
