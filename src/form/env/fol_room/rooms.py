import abc

from .objects import Checkpoint


class _Room(abc.ABC):
    def __init__(self, x: int, y: int, w: int, h: int, description: str = ""):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.description = description

    def build_contours(self, env):
        env.grid.wall_rect(self.x, self.y, self.w, self.h)

    @abc.abstractmethod
    def gen_room(self, env):
        raise NotImplementedError("gen_room")


class Room(_Room):
    def __init__(
        self,
        *args,
        checkpoints_configuration: dict = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.checkpoints_configuration = checkpoints_configuration or {}

    def gen_room(self, env):
        self.objects = []

        env.grid.vert_wall(self.x + 6, self.y + 1, 2)
        env.grid.vert_wall(self.x + 6, self.y + 4, 6)
        env.grid.vert_wall(self.x + 6, self.y + 11, 1)
        env.grid.horz_wall(self.x + 1, self.y + 6, 1)
        env.grid.horz_wall(self.x + 3, self.y + 6, 3)
        env.grid.horz_wall(self.x + 7, self.y + 7, 2)
        env.grid.horz_wall(self.x + 10, self.y + 7, 2)

        for c in ["yellow", "red", "blue", "purple", "grey", "green"]:
            for _ in range(self.checkpoints_configuration.get(c, 0)):
                checkpoint = Checkpoint(c)
                self.objects.append(checkpoint)
                env.place_obj(
                    checkpoint,
                    top=(self.x + 1, self.y + 1),
                    size=(self.w - 3, self.h - 3),
                )
