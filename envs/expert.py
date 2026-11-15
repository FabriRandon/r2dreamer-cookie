"""Hand-written "perfect" policy for the cookie domain (ThreeRooms).

There are two experts, chosen with `make_expert(view)` (env.expert_view): `ExpertPolicy`
decides from a 5x5 partial view of its own and `ExpertPolicy3x3` from a 3x3 one. They only
look at that symbolic view (the one MiniGrid builds, before it is drawn as an image) and at
a small memory of their own movements; they never read the simulator's state. Every cycle
goes like this:

  1. press   walk to the button and bump into it (the cookie appears in one of the
             outer corners of one of the two side rooms);
  2. search  the 5x5 expert goes back down and looks into one of the two side rooms,
             picked at random, from the doorway: a 5x5 view standing in the doorway sees
             the whole 5x5 room; if the cookie is not there, it looks into the other room.
             The 3x3 expert cannot see a room from its doorway: it walks into a room
             (picked at random) until the back wall is in view and turns to look at each of
             the two corners, up and down in a random order, then does the same in the other
             room if the cookie was not there;
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

# Side of the square view the expert decides from: 5 is the original expert, which
# sees a whole side room from its doorway; 3 is the expert with the same small view
# as a 3x3 agent, which has to walk to the back of a room to see its corners.
EXPERT_VIEW = 5
EXPERT_VIEWS = (3, 5)

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
    """The perfect policy for a 5x5 view. Call `reset()` when an episode starts, then
    `act(view)` with the symbolic view (shape [n, n, 3] or [n, n] with n >= MIN_VIEW, whose
    first channel is the object id) to get the next action."""

    MIN_VIEW = 5

    def __init__(self, hallway=5, seed=0):
        self.hallway = hallway
        self._rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.pose = START
        self.phase = "press"
        self.cookie = None  # where the cookie was seen, relative to the start
        self.rooms = []  # the order in which the side rooms are looked into
        self.status = "va al botón"  # what it is doing, in words (captions of the demo videos)
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
        assert ids.shape[0] == ids.shape[1] >= self.MIN_VIEW, (
            f"this expert needs a view of at least {self.MIN_VIEW}x{self.MIN_VIEW}, got {ids.shape}"
        )
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
                self._start_search()
                self.status = "aprieta el botón"
                return FORWARD
            self.status = "va al botón"
            return action

        if self.phase == "fetch" and self.pose[:2] == self.cookie:
            self.phase, self.cookie = "press", None
            return self._decide()

        if self.phase == "search":
            if self.cookie is not None:
                self.phase = "fetch"
                return self._decide()
            return self._search()

        self.status = "va por la galleta"
        return self._go(self.cookie)

    def _start_search(self):
        """The button was just pressed: the cookie is in one of the side rooms."""
        self.rooms = self._rng.sample((-1, 1), 2)

    def _search(self):
        x, y, d = self.vantage(self.rooms[0])
        action = self._go((x, y), d)
        if action is None:
            # In the doorway and no cookie in sight: it is in the other room.
            self.rooms.reverse()
            x, y, d = self.vantage(self.rooms[0])
            action = self._go((x, y), d)
        self.status = f"mira la sala {'este' if self.rooms[0] > 0 else 'oeste'} desde la entrada"
        return action


class ExpertPolicy3x3(ExpertPolicy):
    """The expert for an agent that only sees 3x3 cells: two ahead and one to each side.

    It cannot see into a side room from its doorway, and the cookie always appears
    in one of the room's two outer corners (the ones at the back, to the left and to
    the right of its middle row). So, for each side room it looks into, picked at
    random:

      1. walk in along the middle row until the back wall comes into view (two
         cells ahead is the wall: at 6 cells from the start);
      2. pick at random whether to check the corner "up" or "down" first, turn
         to that side and go towards it until the corner is in view. A 3x3 view
         reaches two cells ahead, so at the back of the room turning is enough;
      3. if the cookie is not there, turn around and do the same for the other corner;
      4. if it is in neither, the cookie is in the other room: do the same there.

    The cookie is fetched as soon as it is seen (also if it shows up on the way), and
    then it goes back to the button, as the 5x5 expert does. It only reads 3x3 views
    and its own dead reckoning, like the agent it plays for.
    """

    MIN_VIEW = 3

    def __init__(self, hallway=5, seed=0):
        super().__init__(hallway, seed)
        # Not the seed the environment seeds the cookie's corner with: with the same one, the
        # expert's first choice of room would always be the cookie's room in the first cycle.
        self._rng = random.Random(f"expert-3x3/{seed}")

    def reset(self):
        super().reset()
        self.looks = []  # (side, "up" | "down"): the corners still to look at in the room being searched

    def _start_search(self):
        super()._start_search()
        self.looks = []

    def back_pose(self, side):
        """Where the back wall of a side room comes into view: in its middle row, with the
        wall two cells ahead (the room's last cell is at `hallway + 2`, the wall right behind)."""
        return (side * (self.hallway + 1), 0)

    def _search(self):
        for _ in range(8):  # one pass is enough: this only repeats to move on to the next corner or room
            if not self.looks:
                if not self.rooms:
                    # Both rooms were looked at without finding the cookie. It cannot happen on
                    # the real map; start over rather than stand still.
                    self.rooms = self._rng.sample((-1, 1), 2)
                side = self.rooms.pop(0)
                self.looks = [(side, vertical) for vertical in self._rng.sample(("up", "down"), 2)]
            side, vertical = self.looks[0]
            back = self.back_pose(side)
            room = "este" if side > 0 else "oeste"
            if self.pose[:2] != back:
                self.status = f"entra a la sala {room} hasta ver el fondo"
                return self._go(back, EAST if side > 0 else WEST)
            action = self._go(back, NORTH if vertical == "up" else SOUTH)
            if action is not None:
                self.status = f"gira para mirar {'arriba' if vertical == 'up' else 'abajo'} en la sala {room}"
                return action
            # Facing the corner with nothing in sight (else the cookie would be seen): next.
            self.status = f"no hay galleta {'arriba' if vertical == 'up' else 'abajo'} en la sala {room}"
            self.looks.pop(0)
        return FORWARD  # unreachable on the real map


def make_expert(view, hallway=5, seed=0):
    """The expert that decides from a view of `view` x `view` cells (see EXPERT_VIEWS)."""
    if view not in EXPERT_VIEWS:
        raise ValueError(f"there is an expert for views {EXPERT_VIEWS}, not {view}")
    return (ExpertPolicy3x3 if view == 3 else ExpertPolicy)(hallway=hallway, seed=seed)
