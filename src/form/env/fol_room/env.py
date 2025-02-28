import re
from functools import cached_property, lru_cache

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Door, Goal, Key, Lava
from minigrid.minigrid_env import TILE_PIXELS, MiniGridEnv

from ..wrappers import LabelingFunctionWrapper
from .objects import OBJECT_TO_IDX, STATE_TO_IDX, Checkpoint
from .rooms import Room


class FOLMultiRoomsEnv(MiniGridEnv):
    def __init__(
        self,
        rooms,
        agent_start_room,
        has_lava: bool = True,
        max_steps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        self.rooms = rooms
        self.agent_start_room = agent_start_room
        self.has_lava = has_lava

        self.agent_start_pos = None
        self.agent_start_dir = 0

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 20 * width**2

        super().__init__(
            mission_space=mission_space,
            max_steps=max_steps,
            width=width,
            height=height,
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
            screen_size=screen_size,
            highlight=highlight,
            tile_size=tile_size,
            agent_pov=agent_pov,
        )

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
            "remote_toggle": "T",
            "checkpoint": "C",
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        output = ""

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    output += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                tile = self.grid.get(i, j)

                if tile is None:
                    output += "  "
                    continue

                if tile.type == "door":
                    if tile.is_open:
                        output += "__"
                    elif tile.is_locked:
                        output += "L" + tile.color[0].upper()
                    else:
                        output += "D" + tile.color[0].upper()
                    continue

                oid = self.objects.index(tile) if tile in self.objects else -1
                output += OBJECT_TO_STR[tile.type] + tile.color[0].upper() + str(oid)

            if j < self.grid.height - 1:
                output += "\n"

        return output

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        self.objects = []

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for r in self.rooms:
            r.build_contours(self)

        for r in self.rooms:
            r.gen_room(self)
            self.objects.extend(r.objects)

        if self.has_lava:
            for _ in range(5):

                def _reject(self_, pos):
                    pos = np.array(pos)
                    # avoid placing lava in front of a door
                    for delta in (
                        np.array([0, -1]),
                        np.array([0, 1]),
                        np.array([-1, 0]),
                        np.array([1, 0]),
                    ):
                        adj_cell = pos + delta
                        if isinstance(self_.grid.get(adj_cell[0], adj_cell[1]), Door):
                            return True
                    return False

                lava = Lava()
                self.place_obj(lava, reject_fn=_reject)
                self.objects.append(lava)

        # Place a goal square in the bottom-right corner
        goal = Goal()
        self.objects.append(goal)
        self.put_obj(goal, width - 2, height - 2)

        # Place the agent
        if self.agent_start_room:
            self.place_agent(
                (self.agent_start_room.x, self.agent_start_room.y),
                (self.agent_start_room.w, self.agent_start_room.h),
            )
        else:
            self.place_agent()

        self.mission = self._gen_mission()

    @staticmethod
    def _gen_mission():
        return "Go to Goal"

    def step(self, action):
        obs, reward, terminated, truncated, infos = super().step(action)
        infos["step"] = self.step_count
        return obs, reward, terminated, truncated, infos

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1.0


class FOLMultiRoomLabelingFunctionWrapper(LabelingFunctionWrapper):
    def __init__(self, env: gym.Env, delta: bool = False):
        super().__init__(env)

        self.delta = delta
        self.prev_obs = None

    @staticmethod
    def _get_position(obs, obj_ix):
        pos = list(zip(*np.where(obs[:, :, 3] == obj_ix)))
        if pos:
            return pos[0]

    def _get_labels_from_obs(self, obs: dict):
        labels = []

        for k_ix, key in self.keys.items():
            if obs["carried_object"][3] == k_ix:
                labels.append((f"{key.color}_key(o{k_ix})",))
                # labels.append((f"key(o{k_ix})", f"color({key.color})"))

        for b_ix, ball in self.balls.items():
            if obs["carried_object"][3] == b_ix:
                labels.append((f"{ball.color}_ball(o{b_ix})",))
                # labels.append((f"ball(o{b_ix})", f"color({ball.color})"))

        for c_ix, checkpoint in self.checkpoints.items():
            if self._get_position(obs["image"], c_ix) == tuple(obs["agent_position"]):
                labels.append((f"{checkpoint.color}_checkpoint(o{c_ix})",))

        for d_ix, door in self.doors.items():
            pos = self._get_position(obs["image"], d_ix)
            if pos:
                if obs["image"][pos][2] != STATE_TO_IDX["locked"]:
                    labels.append((f"{door.color}_door_unlocked(o{d_ix})",))
                else:
                    labels.append((f"{door.color}_door_locked(o{d_ix})",))

        for g_ix in self.goals:
            if self._get_position(obs["image"], g_ix) == tuple(obs["agent_position"]):
                labels.append(("goal",))

        for l_ix in self.lava:
            if self._get_position(obs["image"], l_ix) == tuple(obs["agent_position"]):
                labels.append(("lava",))

        return labels

    def get_labels(self, obs: dict, prev_obs: dict):
        labels = self._get_labels_from_obs(obs)
        prev_labels = self._get_labels_from_obs(prev_obs) if prev_obs else []

        if self.delta:
            prev_labels, labels = self._process_door_labels(prev_labels, labels)
            prev_labels, labels = self._process_object_labels(
                "key", prev_labels, labels
            )
            prev_labels, labels = self._process_object_labels(
                "ball", prev_labels, labels
            )
            prev_labels, labels = self._process_checkpoint_labels(prev_labels, labels)

        return [l for ls in labels for l in ls]

    @cached_property
    def _door_locked_pattern(self):
        return re.compile("(?P<color>[a-z]+)_door_locked\((?P<oid>.*)\)")

    @cached_property
    def _door_unlocked_pattern(self):
        return re.compile("(?P<color>[a-z]+)_door_unlocked\((?P<oid>.*)\)")

    def _process_door_labels(self, prev_labels, labels):
        filter_prev_labels = [ls for ls in prev_labels for l in ls if "_door_" not in l]
        filter_labels = [ls for ls in labels for l in ls if "_door_" not in l]

        door_prev_labels = [ls for ls in prev_labels for l in ls if "_door_" in l]
        door_labels = [ls for ls in labels for l in ls if "_door_" in l]

        locked_prev_labels = {
            m.groups()
            for m in (
                self._door_locked_pattern.match(l)
                for ls in door_prev_labels
                for l in ls
            )
            if m
        }
        unlocked_labels = {
            m.groups()
            for m in (
                self._door_unlocked_pattern.match(l) for ls in door_labels for l in ls
            )
            if m
        }

        for color, oid in locked_prev_labels.intersection(unlocked_labels):
            filter_labels.append((f"{color}_door_unlocked({oid})",))

        return filter_prev_labels, filter_labels

    @lru_cache
    def _object_pattern(self, type):
        return re.compile(f"(?P<color>[a-z]+)_{type}\((?P<oid>.*)\)")

    def _process_object_labels(self, type, prev_labels, labels):
        filter_prev_labels = [
            ls for ls in prev_labels for l in ls if f"_{type}" not in l
        ]
        filter_labels = [ls for ls in labels for l in ls if f"_{type}" not in l]

        type_prev_labels = [ls for ls in prev_labels for l in ls if f"_{type}" in l]
        type_labels = [ls for ls in labels for l in ls if f"_{type}" in l]

        type_prev_labels = {
            m.groups()
            for m in (
                self._object_pattern(type).match(l)
                for ls in type_prev_labels
                for l in ls
            )
            if m
        }
        type_labels = {
            m.groups()
            for m in (
                self._object_pattern(type).match(l) for ls in type_labels for l in ls
            )
            if m
        }

        for color, oid in type_labels.difference(type_prev_labels):
            filter_labels.append((f"{color}_{type}_collected({oid})",))

        for color, oid in type_prev_labels.difference(type_labels):
            filter_labels.append((f"{color}_{type}_dropped({oid})",))

        return filter_prev_labels, filter_labels

    @cached_property
    def _checkpoint_pattern(self):
        return re.compile("(?P<color>[a-z]+)_checkpoint\((?P<oid>.*)\)")

    def _process_checkpoint_labels(self, prev_labels, labels):
        filter_prev_labels = [
            ls for ls in prev_labels for l in ls if "_checkpoint" not in l
        ]
        filter_labels = [ls for ls in labels for l in ls if "_checkpoint" not in l]

        checkpoint_prev_labels = [
            ls for ls in prev_labels for l in ls if "_checkpoint" in l
        ]
        checkpoint_labels = [ls for ls in labels for l in ls if "_checkpoint" in l]

        checkpoint_prev_labels = {
            m.groups()
            for m in (
                self._checkpoint_pattern.match(l)
                for ls in checkpoint_prev_labels
                for l in ls
            )
            if m
        }
        checkpoint_labels = {
            m.groups()
            for m in (
                self._checkpoint_pattern.match(l)
                for ls in checkpoint_labels
                for l in ls
            )
            if m
        }

        for color, oid in checkpoint_labels.difference(checkpoint_prev_labels):
            filter_labels.append((f"{color}_checkpoint({oid})",))

        return filter_prev_labels, filter_labels

    @property
    def goals(self):
        return {
            i: o for i, o in enumerate(self.unwrapped.objects) if isinstance(o, Goal)
        }

    @property
    def lava(self):
        return {
            i: o for i, o in enumerate(self.unwrapped.objects) if isinstance(o, Lava)
        }

    @property
    def checkpoints(self):
        return {
            i: o
            for i, o in enumerate(self.unwrapped.objects)
            if isinstance(o, Checkpoint)
        }

    @property
    def keys(self):
        return {
            i: o for i, o in enumerate(self.unwrapped.objects) if isinstance(o, Key)
        }

    @property
    def balls(self):
        return {
            i: o for i, o in enumerate(self.unwrapped.objects) if isinstance(o, Ball)
        }

    @property
    def doors(self):
        return {
            i: o for i, o in enumerate(self.unwrapped.objects) if isinstance(o, Door)
        }

    def get_all_labels(self):
        return list(
            set(
                [
                    f"{key.color}_key_collected(o{k_ix})"
                    for k_ix, key in self.keys.items()
                    if self.delta
                ]
                + [
                    f"{key.color}_key_dropped(o{k_ix})"
                    for k_ix, key in self.keys.items()
                    if self.delta
                ]
                + [
                    f"{ball.color}_ball_collected(o{b_ix})"
                    for b_ix, ball in self.balls.items()
                    if self.delta
                ]
                + [
                    f"{ball.color}_ball_dropped(o{b_ix})"
                    for b_ix, ball in self.balls.items()
                    if self.delta
                ]
                + [
                    f"{key.color}_key(o{k_ix})"
                    for k_ix, key in self.keys.items()
                    if not self.delta
                ]
                + [
                    f"{ball.color}_ball(o{b_ix})"
                    for b_ix, ball in self.balls.items()
                    if not self.delta
                ]
                + [
                    f"{checkpoint.color}_checkpoint(o{c_ix})"
                    for c_ix, checkpoint in self.checkpoints.items()
                    # if not self.delta
                ]
                + [
                    f"{door.color}_door_unlocked(o{d_ix})"
                    for d_ix, door in self.doors.items()
                ]
                + [
                    f"{door.color}_door_locked(o{d_ix})"
                    for d_ix, door in self.doors.items()
                ]
                + ["goal", "lava"]
            )
        )


class IdentifierAndStateObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable grid with a symbolic state representation.
    """

    def __init__(self, env, full_view=True):
        super().__init__(env)

        self.full_view = full_view

        objects_position_space = gym.spaces.Box(
            low=-1,
            high=255,
            shape=(
                env.width,
                env.height,
                4,
            ),
            dtype=np.float32,
        )
        agent_position_space = gym.spaces.MultiDiscrete(
            [self.unwrapped.width, self.unwrapped.height]
        )
        carried_object_space = gym.spaces.Box(
            low=-1,
            high=255,
            shape=(4,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {
                **self.observation_space.spaces,
                "image": objects_position_space,
                "agent_position": agent_position_space,
                "carried_object": carried_object_space,
            }
        )

    def observation(self, obs):

        grid = self.unwrapped.grid

        def _get_object_id(o):
            if o in self.env.objects:
                return self.env.objects.index(o)
            return -1

        objects = np.array(
            [
                (
                    np.array(o.encode() + (_get_object_id(o),))
                    if o is not None
                    else np.array(
                        (
                            -1,
                            -1,
                            -1,
                            -1,
                        )
                    )
                )
                for o in grid.grid
            ]
        ).reshape(grid.height, grid.width, 4)

        objects = np.swapaxes(objects, 0, 1)

        for i in range(grid.width):
            for j in range(grid.height):
                if not self.full_view and not self.unwrapped.in_view(i, j):
                    objects[i, j, :] = np.array((-1, -1, -1, -1))

        agent_pos = self.env.agent_pos
        objects[agent_pos[0], agent_pos[1], 0] = OBJECT_TO_IDX["agent"]

        obs["image"] = objects
        obs["agent_position"] = np.array(self.unwrapped.agent_pos)

        carried_object = self.unwrapped.carrying
        obs["carried_object"] = (
            np.array(carried_object.encode() + (_get_object_id(carried_object),))
            if carried_object
            else np.array(
                (
                    -1,
                    -1,
                    -1,
                    -1,
                )
            )
        )

        return obs


class IdentifierObsWrapper(gym.core.ObservationWrapper):

    def __init__(self, env, full_view=True):
        super().__init__(env)

        self.full_view = full_view

        objects_position_space = gym.spaces.Box(
            low=-1,
            high=255,
            shape=(
                env.width,
                env.height,
                3,
            ),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {
                **self.observation_space.spaces,
                "image": objects_position_space,
            }
        )

    def observation(self, obs):
        grid = self.unwrapped.grid

        def _get_object_id(o):
            if o in self.env.objects:
                return self.env.objects.index(o)
            return -1

        objects = np.array(
            [
                (
                    np.array(o.encode()[:-1] + (_get_object_id(o),))
                    if o is not None
                    else np.array(
                        (
                            -1,
                            -1,
                            -1,
                        )
                    )
                )
                for o in grid.grid
            ]
        ).reshape(grid.height, grid.width, 3)

        objects = np.swapaxes(objects, 0, 1)

        for i in range(grid.width):
            for j in range(grid.height):
                if not self.full_view and not self.unwrapped.in_view(i, j):
                    objects[i, j, :] = np.array((-1, -1, -1))

        agent_pos = self.env.agent_pos
        objects[agent_pos[0], agent_pos[1], 0] = OBJECT_TO_IDX["agent"]

        obs["image"] = objects
        return obs


class PositiveTraceWrapper(gym.Wrapper):

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["is_positive_trace"] = terminated and reward != 0.0
        return obs, reward, terminated, truncated, info


def room_env(**kwargs):
    height = 13
    width = 13

    room = Room(
        x=0,
        y=0,
        w=width,
        h=height,
        checkpoints_configuration={
            "yellow": kwargs.pop("yellow_checkpoint", 2),
            "red": kwargs.pop("red_checkpoint", 2),
            "blue": kwargs.pop("blue_checkpoint", 2),
            "purple": kwargs.pop("purple_checkpoint", 2),
            "grey": kwargs.pop("grey_checkpoint", 2),
            "green": kwargs.pop("green_checkpoint", 2),
        },
    )
    return FOLMultiRoomsEnv(
        rooms=[room],
        agent_start_room=room,
        width=width,
        height=height,
        **kwargs,
    )

