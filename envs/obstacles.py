"""Abstract obstacles for the cookie domain (map with hallways: `hallN`).

Each episode, every room (the button's and the two where cookies appear) gets one solid
obstacle: a figure of a random shape and colour in a random free cell. They are drawn by
MiniGrid like any other object, so they show in the images the agent gets and in the videos,
and they are encoded in the symbolic view with the object id `OBSTACLE`.

What an obstacle may look like is limited so that it cannot be taken for what matters in the
task: no circles (the button is a blue circle, the cookie a brown one with chips) and no
triangles (the agent), and no blue, red or brown colours. So there are five shapes (star,
diamond, square, plus, cross) in four colours (yellow, white, green, magenta); the obstacles of
one episode have different shapes and different colours while there are enough of them.

Where they can go: any free cell of a room except where cookies appear (the two outer corners of
each side room) and the doorway of the room (the cell where its hallway enters it): an obstacle
there would seal the room. A layout is also discarded if any cookie corner, or the button, would
not be reachable from the start.
"""

import math
from collections import deque

from minigrid.core.constants import COLORS
from minigrid.core.world_object import Wall, WorldObj
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_triangle, rotate_fn

# Object id in the symbolic view; cookie-env uses 11 (button) and 12 (cookie). envs/expert.py
# repeats it (it must not import this module), and a test checks both agree.
OBSTACLE = 13

SHAPES = ("star", "diamond", "square", "plus", "cross")
PALETTE = {
    "yellow": (255, 215, 0),
    "white": (255, 255, 255),
    "green": (0, 200, 0),
    "magenta": (230, 0, 230),
}
COLOURS = tuple(PALETTE)


def _star(x, y, points=5, outer=0.44, inner=0.18):
    """Point in a star with `points` tips, one pointing up (y grows downwards)."""
    angles = [math.pi / 2 + k * math.pi / points for k in range(2 * points)]
    radii = [outer if k % 2 == 0 else inner for k in range(2 * points)]
    polygon = [(0.5 + r * math.cos(a), 0.5 - r * math.sin(a)) for r, a in zip(radii, angles)]
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):  # ray casting
        (xi, yi), (xj, yj) = polygon[i], polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _diamond(x, y):
    # A tall rhombus: a regular one is too close to a circle (the button) at the size the agent sees.
    return abs(x - 0.5) / 0.28 + abs(y - 0.5) / 0.44 <= 1


_square = point_in_rect(0.17, 0.83, 0.17, 0.83)
_bar_h, _bar_v = point_in_rect(0.12, 0.88, 0.37, 0.63), point_in_rect(0.37, 0.63, 0.12, 0.88)


def _plus(x, y):
    return _bar_h(x, y) or _bar_v(x, y)


_cross = rotate_fn(_plus, cx=0.5, cy=0.5, theta=math.pi / 4)

SHAPE_FUNCTIONS = {"star": _star, "diamond": _diamond, "square": _square, "plus": _plus, "cross": _cross}


class Obstacle(WorldObj):
    """A solid figure. It is typed "ball" (a type MiniGrid knows, as cookie-env's button), and
    its encoding and its drawing are its own."""

    def __init__(self, shape, colour):
        assert shape in SHAPES and colour in PALETTE, (shape, colour)
        super().__init__("ball", "yellow")
        self.shape, self.colour = shape, colour

    def can_overlap(self):
        return False

    def encode(self):
        # (object id, colour, shape): MiniGrid caches the picture of a tile by this tuple, so
        # it has to tell every shape and colour apart.
        return (OBSTACLE, COLOURS.index(self.colour), SHAPES.index(self.shape))

    def render(self, img):
        fill_coords(img, SHAPE_FUNCTIONS[self.shape], PALETTE[self.colour])

    def __repr__(self):
        return f"Obstacle({self.shape}, {self.colour})"


class FixedObstacle(Wall):
    """The fixed obstacles of the first experiments (`env.obstacles=same|distinct`): a green block that
    is a wall as far as the symbolic view goes, drawn as a square or as a triangle. Kept so that runs
    trained with them can be loaded and explored with the exact same pictures."""

    def __init__(self, triangle):
        super().__init__("green")
        self.triangle = triangle

    def encode(self):
        # Grid.render_tile caches by encoding, so each shape needs its own key.
        kind, color, _ = super().encode()
        return kind, color, int(self.triangle)

    def render(self, img):
        shape = (point_in_triangle((0.5, 0.08), (0.9, 0.88), (0.1, 0.88)) if self.triangle
                 else point_in_rect(0.22, 0.78, 0.22, 0.78))
        fill_coords(img, shape, COLORS[self.color])


FIXED_MODES = ("same", "distinct")
SOLID = (Obstacle, FixedObstacle)


def parse(value):
    """What `env.obstacles` asks for: (random obstacles per room, fixed mode). It is a number of random
    obstacles per room (0 or "none": no obstacles), or one of the fixed layouts "same" (two squares) and
    "distinct" (a square and a triangle) of the first experiments."""
    if value in (None, False, "none", ""):
        return 0, None
    if value in FIXED_MODES:
        return 0, value
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"unknown value for env.obstacles: {value!r} (a number per room, or none, same, distinct)") from None
    if count < 0:
        raise ValueError(f"env.obstacles cannot be negative, got {count}")
    return count, None


def place_fixed(env, mode, hallway):
    """The fixed layout: a square in the west room (one row above its middle) and, in the east room (one
    row below its middle), another square (`same`) or a triangle (`distinct`)."""
    cx, cy = (int(v) for v in env.agent_pos)
    env.put_obj(FixedObstacle(triangle=False), cx - hallway, cy - 1)
    env.put_obj(FixedObstacle(triangle=mode == "distinct"), cx + hallway, cy + 1)


def layout(start, hallway):
    """Rooms of the map with hallways of `hallway` cells, for an agent that starts at `start`:
    the 5x5 room cells, the doorway and, for the side rooms, the corners where cookies appear."""
    cx, cy = start
    h = hallway
    rooms = {
        "boton": dict(center=(cx, cy - h), door=(cx, cy - h + 2), corners=[]),
        "oeste": dict(center=(cx - h, cy), door=(cx - h + 2, cy), corners=[(cx - h - 2, cy - 2), (cx - h - 2, cy + 2)]),
        "este": dict(center=(cx + h, cy), door=(cx + h - 2, cy), corners=[(cx + h + 2, cy - 2), (cx + h + 2, cy + 2)]),
    }
    for room in rooms.values():
        rx, ry = room["center"]
        room["cells"] = [(rx + dx, ry + dy) for dx in range(-2, 3) for dy in range(-2, 3)]
    return rooms


def reachable(env, start, blocked):
    """Cells the agent can walk to from `start` on the grid, with `blocked` cells added to the
    walls and the other objects that block."""
    seen, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt in seen or nxt in blocked or not (0 <= nxt[0] < env.width and 0 <= nxt[1] < env.height):
                continue
            cell = env.grid.get(*nxt)
            if cell is None or cell.can_overlap():
                seen.add(nxt)
                queue.append(nxt)
    return seen


def place(env, rng, hallway, per_room=1, tries=200):
    """Put `per_room` new obstacles in each room of `env` (just reset, the agent at the start),
    at random. Returns a list of (room, cell, shape, colour)."""
    start = tuple(int(v) for v in env.agent_pos)
    rooms = layout(start, hallway)
    button = next((x, y) for x in range(env.width) for y in range(env.height)
                  if type(env.grid.get(x, y)).__name__ == "Button")
    must_reach = [c for room in rooms.values() for c in room["corners"]]
    candidates = {
        name: [c for c in room["cells"] if env.grid.get(*c) is None and c != room["door"] and c not in room["corners"]]
        for name, room in rooms.items()
    }
    for _ in range(tries):
        chosen = [(name, cells[int(rng.integers(len(cells)))])
                  for name, cells in candidates.items() for _ in range(per_room)]
        if len({cell for _, cell in chosen}) < len(chosen):
            continue  # two on the same cell
        seen = reachable(env, start, {cell for _, cell in chosen})
        around_button = [(button[0] + dx, button[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        if all(c in seen for c in must_reach) and any(c in seen for c in around_button):
            break
    else:
        raise RuntimeError("no layout of obstacles leaves the cookie corners and the button reachable")
    # Different shapes and colours while there are enough of them.
    shapes = [SHAPES[i] for i in _distinct(rng, len(SHAPES), len(chosen))]
    colours = [COLOURS[i] for i in _distinct(rng, len(COLOURS), len(chosen))]
    placed = []
    for (name, cell), shape, colour in zip(chosen, shapes, colours):
        env.put_obj(Obstacle(shape, colour), *cell)
        placed.append((name, cell, shape, colour))
    return placed


def _distinct(rng, options, count):
    """`count` indices below `options`: all different until they run out, then repeating."""
    order = []
    while len(order) < count:
        order += [int(i) for i in rng.permutation(options)]
    return order[:count]
