from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    STATE_TO_IDX,
)
from minigrid.core.world_object import *
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_circle

OBJECT_TO_IDX.update(
    {
        "remote_toggle_door": 11,
        "remote_toggle": 12,
        "checkpoint": 13,
    }
)

IDX_TO_OBJECT.update(dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys())))


STATE_TO_IDX.update({"on": 3, "off": 4})

IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))


def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
    """Create an object from a 3-tuple state description"""

    obj_type = IDX_TO_OBJECT[type_idx]
    color = IDX_TO_COLOR[color_idx]

    if obj_type == "empty" or obj_type == "unseen" or obj_type == "agent":
        return None

    # State, 0: open, 1: closed, 2: locked
    is_open = state == 0
    is_locked = state == 2

    if obj_type == "wall":
        v = Wall(color)
    elif obj_type == "floor":
        v = Floor(color)
    elif obj_type == "ball":
        v = Ball(color)
    elif obj_type == "key":
        v = Key(color)
    elif obj_type == "box":
        v = Box(color)
    elif obj_type == "door":
        v = Door(color, is_open, is_locked)
    elif obj_type == "goal":
        v = Goal()
    elif obj_type == "lava":
        v = Lava()
    elif obj_type == "remote_toggle_door":
        v = RemoteToggleDoor(color, is_open=is_open, is_locked=is_locked)
    elif obj_type == "checkpoint":
        v = Checkpoint(color)
    else:
        assert False, "unknown object type in decode '%s'" % obj_type

    return v


WorldObj.decode = decode


class Checkpoint(WorldObj):

    def __init__(self, color: str):
        super().__init__("checkpoint", color)

    def can_overlap(self) -> bool:
        return True

    def can_pickup(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class RemoteToggleDoor(Door):
    def __init__(self, color: str, is_open: bool = False, is_locked: bool = False):
        WorldObj.__init__(self, "remote_toggle_door", color)
        self.is_open = is_open
        self.is_locked = is_locked

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            toggles = [
                o
                for o in env.grid.grid
                if isinstance(o, RemoteToggle) and o.color == self.color
            ]
            if all(t.is_on for t in toggles):
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True


class RemoteToggle(WorldObj):
    def __init__(self, color: str, is_on: bool = False):
        super().__init__("remote_toggle", color=color)
        self.is_on = is_on
        self.outer_color = "red" if not is_on else "green"

    def can_pickup(self):
        return False

    def toggle(self, env: MiniGridEnv, pos: tuple[int, int]) -> bool:
        if not self.is_on:
            self.outer_color = "green"
            self.is_on = True
        # else:
        #     self.outer_color = "red"
        #     self.is_on = False
        return True

    def render(self, img):
        fill_coords(
            img, point_in_circle(cx=0.5, cy=0.5, r=0.31), COLORS[self.outer_color]
        )
        fill_coords(img, point_in_circle(cx=0.5, cy=0.5, r=0.25), COLORS[self.color])

    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (
            OBJECT_TO_IDX[self.type],
            COLOR_TO_IDX[self.color],
            STATE_TO_IDX[("on" if self.is_on else "off")],
        )
