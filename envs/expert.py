"""Hand-written "perfect" policy for the cookie domain (ThreeRooms).

The policy only looks at the agent's own 5x5 partial view (the symbolic one
MiniGrid builds, before it is drawn as an image) and at a small memory of its
own movements; it never reads the simulator's state. Every cycle goes like this:

  1. press   walk to the button and bump into it (the cookie appears in a
             random corner of one of the two side rooms);
  2. search  go back down and look into one of the two side rooms, picked at
             random, from the doorway: a 5x5 view standing in the doorway sees
             the whole 5x5 room. If the cookie is not there, look into the
             other room;
  3. fetch   walk to the cookie as soon as it is seen;
  4. press   return through the starting cell and press the button again,
             until the episode's time limit.

The map is known by the policy (it is a hand-written expert), but the cookie's
position is not: it has to be seen. Positions are tracked relative to the
starting cell by dead reckoning, so the policy works with any grid origin.

Actions are the ones of `envs.cookie.Cookie` with `press_on_move`: 0 turn left,
1 turn right, 2 forward (against the button, forward presses it).
"""

import random
from collections import deque
from functools import lru_cache

import numpy as np

LEFT, RIGHT, FORWARD = 0, 1, 2

# MiniGrid's object ids as they appear in the symbolic view (cookie-env adds
# the button and the cookie).
WALL, BUTTON, COOKIE = 2, 11, 12

# Headings in the order MiniGrid turns right (clockwise on screen, y grows
# downwards): a right turn is +1 and a left turn is -1.
EAST, SOUTH, WEST, NORTH = range(4)
STEP = ((1, 0), (0, 1), (-1, 0), (0, -1))

# The agent starts at the junction of the hallways, looking east.
START = (0, 0, EAST)


def _walkable(hallway):
    """Cells the agent can stand on, relative to the starting cell.

    Three 5x5 rooms whose centers are `hallway` cells away from the start (the
    button's room to the north, the cookie rooms to the east and west), joined
    through the start by two straight hallways. The button's cell is a room's
    center and blocks the way.
    """
    cells = set()
    for cx, cy in ((0, -hallway), (-hallway, 0), (hallway, 0)):
        cells.update((cx + dx, cy + dy) for dx in range(-2, 3) for dy in range(-2, 3))
    cells.update((0, y) for y in range(-hallway, 1))
    cells.update((x, 0) for x in range(-hallway, hallway + 1))
    cells.discard((0, -hallway))
    return frozenset(cells)


@lru_cache(maxsize=None)
def _first_action(hallway, pose, cell, heading):
    """First action of a shortest path from `pose` to `cell` (with `heading`
    if given), or None when the agent is already there. Turns and moves all
    cost one step, as in the environment."""
    walkable = _walkable(hallway)
    if pose[:2] == cell and heading in (None, pose[2]):
        return None
    seen, queue = {pose}, deque([(pose, None)])
    while queue:
        (x, y, d), first = queue.popleft()
        dx, dy = STEP[d]
        moves = [(FORWARD, (x + dx, y + dy, d)), (LEFT, (x, y, (d - 1) % 4)), (RIGHT, (x, y, (d + 1) % 4))]
        for action, nxt in moves:
            if nxt in seen or nxt[:2] not in walkable:
                continue
            seen.add(nxt)
            first_action = action if first is None else first
            if nxt[:2] == cell and heading in (None, nxt[2]):
                return first_action
            queue.append((nxt, first_action))
    raise ValueError(f"{cell} cannot be reached from {pose}")


class ExpertPolicy:
    """The perfect policy. Call `reset()` when an episode starts, then
    `act(view)` with the symbolic 5x5 view (shape [5, 5, 3] or [5, 5], whose
    first channel is the object id) to get the next action."""

    def __init__(self, hallway=5, seed=0):
        self.hallway = hallway
        self._rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.pose = START
        self.phase = "press"
        self.cookie = None  # where the cookie was seen, relative to the start
        self.rooms = []  # the order in which the side rooms are looked into
        self._last = None  # (action, whether the cell ahead was blocked)

    @property
    def press_pose(self):
        # Right below the button, looking at it.
        return (0, -(self.hallway - 1), NORTH)

    def vantage(self, side):
        """Pose in the doorway of a side room (side -1 is west, +1 is east)
        from which a 5x5 view covers the whole room."""
        return (side * (self.hallway - 2), 0, EAST if side > 0 else WEST)

    def act(self, view):
        view = np.asarray(view)
        ids = view[..., 0] if view.ndim == 3 else view
        assert ids.shape[0] == ids.shape[1] >= 5, f"the expert needs a view of at least 5x5, got {ids.shape}"
        self._update_pose()
        self._see_cookie(ids)
        action = self._decide()
        size = ids.shape[0]
        # The agent sits at the middle of the view's near edge, and the cell it
        # faces is the one right above it.
        self._last = (action, ids[size // 2, size - 2] in (WALL, BUTTON))
        return action

    def _update_pose(self):
        if self._last is None:
            return
        action, blocked = self._last
        x, y, d = self.pose
        if action == LEFT:
            self.pose = (x, y, (d - 1) % 4)
        elif action == RIGHT:
            self.pose = (x, y, (d + 1) % 4)
        elif not blocked:
            dx, dy = STEP[d]
            self.pose = (x + dx, y + dy, d)

    def _see_cookie(self, ids):
        size = ids.shape[0]
        seen = np.argwhere(ids == COOKIE)
        if not len(seen):
            return
        i, j = seen[0]
        # In the view, i grows to the agent's right and j towards the agent.
        ahead, right = size - 1 - j, i - size // 2
        x, y, d = self.pose
        (fx, fy), (rx, ry) = STEP[d], STEP[(d + 1) % 4]
        self.cookie = (x + ahead * fx + right * rx, y + ahead * fy + right * ry)

    def _go(self, cell, heading=None):
        return _first_action(self.hallway, self.pose, cell, heading)

    def _decide(self):
        if self.phase == "press":
            action = self._go(self.press_pose[:2], self.press_pose[2])
            if action is None:
                # Bump into the button; the cookie shows up in a side room.
                self.phase = "search"
                self.cookie = None
                self.rooms = self._rng.sample((-1, 1), 2)
                return FORWARD
            return action

        if self.phase == "fetch" and self.pose[:2] == self.cookie:
            self.phase, self.cookie = "press", None
            return self._decide()

        if self.phase == "search":
            if self.cookie is not None:
                self.phase = "fetch"
                return self._decide()
            x, y, d = self.vantage(self.rooms[0])
            action = self._go((x, y), d)
            if action is None:
                # In the doorway and no cookie in sight: it is in the other room.
                self.rooms.reverse()
                x, y, d = self.vantage(self.rooms[0])
                action = self._go((x, y), d)
            return action

        return self._go(self.cookie)
