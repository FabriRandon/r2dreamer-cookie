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
# the button and the cookie; envs/obstacles.py the obstacles).
WALL, BUTTON, COOKIE, OBSTACLE = 2, 11, 12, 13

# Headings in the order MiniGrid turns right (clockwise on screen, y grows
# downwards): a right turn is +1 and a left turn is -1.
EAST, SOUTH, WEST, NORTH = range(4)
STEP = ((1, 0), (0, 1), (-1, 0), (0, -1))

# The agent starts at the junction of the hallways, looking east.
START = (0, 0, EAST)


@lru_cache(maxsize=8)
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


def _moves(pose):
    x, y, d = pose
    dx, dy = STEP[d]
    return [(FORWARD, (x + dx, y + dy, d)), (LEFT, (x, y, (d - 1) % 4)), (RIGHT, (x, y, (d + 1) % 4))]


@lru_cache(maxsize=50_000)
def _first_action(hallway, pose, cell, heading, blocked=frozenset()):
    """First action of a shortest path from `pose` to `cell` (with `heading`
    if given), or None when the agent is already there. Turns and moves all
    cost one step, as in the environment. `blocked` are cells with an obstacle."""
    walkable = _walkable(hallway) - blocked
    if pose[:2] == cell and heading in (None, pose[2]):
        return None
    seen, queue = {pose}, deque([(pose, None)])
    while queue:
        pose_now, first = queue.popleft()
        for action, nxt in _moves(pose_now):
            if nxt in seen or nxt[:2] not in walkable:
                continue
            seen.add(nxt)
            first_action = action if first is None else first
            if nxt[:2] == cell and heading in (None, nxt[2]):
                return first_action
            queue.append((nxt, first_action))
    raise ValueError(f"{cell} cannot be reached from {pose}")


@lru_cache(maxsize=50_000)
def _first_action_any(hallway, pose, goals, blocked=frozenset()):
    """Like _first_action, towards the nearest of several poses (a frozenset of (x, y, heading))."""
    walkable = _walkable(hallway) - blocked
    if pose in goals:
        return None
    seen, queue = {pose}, deque([(pose, None)])
    while queue:
        pose_now, first = queue.popleft()
        for action, nxt in _moves(pose_now):
            if nxt in seen or nxt[:2] not in walkable:
                continue
            seen.add(nxt)
            first_action = action if first is None else first
            if nxt in goals:
                return first_action
            queue.append((nxt, first_action))
    raise ValueError(f"none of {sorted(goals)} can be reached from {pose}")


def _in_view(pose, cell, size):
    """Whether `cell` is inside the size x size view of an agent at `pose` (the agent is at the
    middle of the near edge: `size - 1` cells ahead, `size // 2` to each side)."""
    x, y, d = pose
    (fx, fy), (rx, ry) = STEP[d], STEP[(d + 1) % 4]
    dx, dy = cell[0] - x, cell[1] - y
    return 0 <= dx * fx + dy * fy <= size - 1 and abs(dx * rx + dy * ry) <= size // 2


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
        self.obstacles = set()  # cells where it has seen an obstacle this episode, relative to the start
        self.status = "va al botón"  # what it is doing, in words (captions of the demo videos)
        self._last = None  # (action, whether the cell ahead was blocked)

    @property
    def press_pose(self):
        # Right below the button, looking at it.
        return (0, -(self.hallway - 1), NORTH)

    def press_poses(self):
        """Poses next to the button, looking at it, that it can stand on: the cell below it
        (where it goes when nothing is in the way) and the ones at its sides and above."""
        h = self.hallway
        poses = [(0, -(h - 1), NORTH), (-1, -h, EAST), (1, -h, WEST), (0, -(h + 1), SOUTH)]
        walkable = _walkable(h)
        return [p for p in poses if p[:2] in walkable and p[:2] not in self.obstacles]

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
        self._see_obstacles(ids)
        self._see_cookie(ids)
        self._observe(ids)
        action = self._decide()
        size = ids.shape[0]
        # The agent sits at the middle of the view's near edge, and the cell it
        # faces is the one right above it.
        self._last = (action, ids[size // 2, size - 2] in (WALL, BUTTON, OBSTACLE))
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

    def _cell_in_view(self, i, j, size):
        """Where, relative to the start, the cell [i, j] of the view is."""
        # In the view, i grows to the agent's right and j towards the agent.
        ahead, right = size - 1 - j, i - size // 2
        x, y, d = self.pose
        (fx, fy), (rx, ry) = STEP[d], STEP[(d + 1) % 4]
        return (x + ahead * fx + right * rx, y + ahead * fy + right * ry)

    def _see_cookie(self, ids):
        seen = np.argwhere(ids == COOKIE)
        if len(seen):
            self.cookie = self._cell_in_view(*seen[0], ids.shape[0])

    def _see_obstacles(self, ids):
        """Remember the obstacles in view: they stay where they are for the whole episode."""
        for i, j in np.argwhere(ids == OBSTACLE):
            self.obstacles.add(self._cell_in_view(i, j, ids.shape[0]))

    def _observe(self, ids):
        """Hook for what a subclass learns from the view besides cookies and obstacles."""

    def _blocked(self):
        return frozenset(self.obstacles)

    def _go(self, cell, heading=None):
        """First action towards `cell`, walking around the obstacles it has seen."""
        try:
            return _first_action(self.hallway, self.pose, cell, heading, self._blocked())
        except ValueError:  # cannot happen on a real layout: do not stand still if it does
            return _first_action(self.hallway, self.pose, cell, heading)

    def _go_any(self, goals):
        """First action towards the nearest of the poses `goals`, or None if it is at one."""
        goals = frozenset(goals)
        try:
            return _first_action_any(self.hallway, self.pose, goals, self._blocked())
        except ValueError:
            return _first_action_any(self.hallway, self.pose, goals)

    def _decide(self):
        if self.phase == "press":
            poses = self.press_poses()
            action = self._go_any(poses)
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

    Steps 1 and 2 are really "go to the nearest pose from which the corner is in view": with
    the room empty that is the middle row up to the back wall and a turn, as described, and
    with an obstacle in the way it is the next best pose. A corner that came into view on the
    way, with no cookie in it, counts as looked at. The cookie is fetched as soon as it is
    seen, and then it goes back to the button, as the 5x5 expert does. Obstacles are
    solid and nothing says where they are: it remembers the ones it sees (also at the sides
    of its path) and plans around them. It only reads 3x3 views and its own dead reckoning,
    like the agent it plays for.
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
        self.empty = set()  # corners seen without a cookie since the button was pressed

    def _start_search(self):
        super()._start_search()
        self.looks, self.empty = [], set()

    def corner(self, side, vertical):
        """Where the cookie can be: the outer corners of the side room `side` (-1 west, 1 east)."""
        return (side * (self.hallway + 2), -2 if vertical == "up" else 2)

    def _observe(self, ids):
        # With no cookie in sight, any corner inside the view is empty: the cookie does not move.
        if self.phase != "search" or self.cookie is not None:
            return
        for side in (-1, 1):
            for vertical in ("up", "down"):
                if _in_view(self.pose, self.corner(side, vertical), ids.shape[0]):
                    self.empty.add(self.corner(side, vertical))

    def look_poses(self, corner):
        """Poses it can stand on (avoiding the obstacles it has seen) with `corner` in its 3x3 view."""
        return [(x, y, d) for x, y in sorted(_walkable(self.hallway)) if (x, y) not in self.obstacles
                for d in (EAST, SOUTH, WEST, NORTH) if _in_view((x, y, d), corner, self.MIN_VIEW)]

    def _search(self):
        for _ in range(8):  # only repeats to move on to the next corner or room
            self.looks = [look for look in self.looks if self.corner(*look) not in self.empty]
            if not self.looks:
                if not self.rooms:
                    # Both rooms were looked at without finding the cookie. It cannot happen on
                    # the real map; start over rather than stand still.
                    self.rooms, self.empty = self._rng.sample((-1, 1), 2), set()
                side = self.rooms.pop(0)
                self.looks = [(side, vertical) for vertical in self._rng.sample(("up", "down"), 2)]
                continue
            side, vertical = self.looks[0]
            goals = self.look_poses(self.corner(side, vertical))
            action = self._go_any(goals)
            room = "este" if side > 0 else "oeste"
            if action is not None:
                turning = any(g[:2] == self.pose[:2] for g in goals)
                self.status = (f"gira para mirar {'arriba' if vertical == 'up' else 'abajo'} en la sala {room}" if turning
                               else f"entra a la sala {room} hasta ver el fondo")
                return action
            self.empty.add(self.corner(side, vertical))  # standing where it sees it: nothing there
        return FORWARD  # unreachable on the real map


def make_expert(view, hallway=5, seed=0):
    """The expert that decides from a view of `view` x `view` cells (see EXPERT_VIEWS)."""
    if view not in EXPERT_VIEWS:
        raise ValueError(f"there is an expert for views {EXPERT_VIEWS}, not {view}")
    return (ExpertPolicy3x3 if view == 3 else ExpertPolicy)(hallway=hallway, seed=seed)
