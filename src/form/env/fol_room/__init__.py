import numpy as np
from gymnasium.envs.registration import register
from gymnasium.wrappers import FilterObservation
from minigrid.wrappers import (
    FullyObsWrapper,
    ImgObsWrapper,
    RGBImgObsWrapper,
    SymbolicObsWrapper,
)

from .env import (
    FOLMultiRoomLabelingFunctionWrapper,
    FOLMultiRoomsEnv,
    IdentifierAndStateObsWrapper,
    IdentifierObsWrapper,
    PositiveTraceWrapper,
    room_env,
)


def _additional_wrappers(full_view=True, delta=False):
    return (
        IdentifierAndStateObsWrapper.wrapper_spec(full_view=full_view),
        FOLMultiRoomLabelingFunctionWrapper.wrapper_spec(delta=delta),
        FilterObservation.wrapper_spec(filter_keys=("direction", "image"))
    )

register(
    id="form/FOLRoom-Blue-AllYellow-7",
    entry_point="form.env.fol_room:room_env",
    disable_env_checker=True,
    additional_wrappers=_additional_wrappers(full_view=True, delta=True),
    kwargs=dict(
        has_lava=False,
        yellow_checkpoint=2,
        blue_checkpoint=2,
        see_through_walls=False,
        agent_view_size=3,
    ),
)

register(
    id="form/FOLRoom-GreenButOne-NoLava",
    entry_point="form.env.fol_room:room_env",
    disable_env_checker=True,
    additional_wrappers=_additional_wrappers(full_view=True, delta=True),
    kwargs=dict(
        has_lava=True,
        green_checkpoint=3,
        see_through_walls=False,
        agent_view_size=3,
    ),
)

register(
    id="form/FOLRoom-AllYellow-2",
    entry_point="form.env.fol_room:room_env",
    disable_env_checker=True,
    additional_wrappers=_additional_wrappers(full_view=True, delta=True),
    kwargs=dict(
        has_lava=False,
        yellow_checkpoint=2,
        see_through_walls=False,
        agent_view_size=3,
    ),
)

register(
    id="form/FOLRoom-AllYellow-4",
    entry_point="form.env.fol_room:room_env",
    disable_env_checker=True,
    additional_wrappers=_additional_wrappers(full_view=True, delta=True),
    kwargs=dict(
        has_lava=False,
        yellow_checkpoint=4,
        see_through_walls=False,
        agent_view_size=3,
    ),
)

register(
    id="form/FOLRoom-AllYellow-6",
    entry_point="form.env.fol_room:room_env",
    disable_env_checker=True,
    additional_wrappers=_additional_wrappers(full_view=True, delta=True),
    kwargs=dict(
        has_lava=False,
        yellow_checkpoint=6,
        see_through_walls=False,
        agent_view_size=3,
    ),
)