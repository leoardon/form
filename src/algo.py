import copy
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import ray
from model import PPORMLearningCatalog, PPORMLearningSharedLayersCatalog
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo import LEARNER_RESULTS_KL_KEY
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module import INFERENCE_ONLY
from ray.rllib.core.rl_module.marl_module import (
    MultiAgentRLModule,
    MultiAgentRLModuleConfig,
    MultiAgentRLModuleSpec,
    RLModule,
)
from ray.rllib.core.rl_module.rl_module import (
    RLMODULE_STATE_DIR_NAME,
    RLModuleConfig,
    SingleAgentRLModuleSpec,
)
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import PublicAPI, override
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    ENV_RUNNER_RESULTS,
    ENV_RUNNER_SAMPLING_TIMER,
    LEARNER_ADDITIONAL_UPDATE_TIMER,
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED_LIFETIME,
    NUM_EPISODES,
    NUM_EPISODES_LIFETIME,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TIMERS,
    TRAINING_ITERATION_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.serialization import (
    deserialize_type,
    space_from_dict,
    space_to_dict,
)
from ray.rllib.utils.typing import EnvType, ModuleID, PolicyID, ResultDict
from form import RewardMachine

logger = logging.getLogger(__name__)

LEARNER_RM_TIMER = "rm_learning"

@ray.remote
class SharedTracesBuffer:

    SHARED_TRACES_BUFFER_REF = "actor:shared_traces_buffer"

    def __init__(self, top_k: int = 1e6) -> None:
        self.top_k = int(top_k)

        self.incomplete_examples = set()
        self.positive_examples = set()
        self.dend_examples = set()

        self._prev_positive = set()
        self._prev_dend = set()
        self._prev_incomplete = set()

    def add_positive(self, example):
        self.positive_examples.add(example)

    def add_incomplete(self, example):
        self.incomplete_examples.add(example)

    def add_dend(self, example):
        self.dend_examples.add(example)

    def get_all_examples(self):
        return (
            list(self.positive_examples),
            list(self.dend_examples),
            list(self.incomplete_examples),
        )

    def get_examples(self):
        positive_examples = sorted(self.positive_examples)[: self.top_k]
        dend_examples = sorted(self.dend_examples)[: self.top_k]
        incomplete_examples = sorted(self.incomplete_examples)[: self.top_k]
        if (
            set(positive_examples) != self._prev_positive
            or set(dend_examples) != self._prev_dend
            or set(incomplete_examples) != self._prev_incomplete
        ):
            self._prev_positive = set(positive_examples)
            self._prev_dend = set(dend_examples)
            self._prev_incomplete = set(incomplete_examples)
            return (positive_examples, dend_examples, incomplete_examples)

        return [], [], []

    def get_positive(self):
        return self.positive_examples

    def get_incomplete(self):
        return self.incomplete_examples

    def get_dend(self):
        return self.dend_examples

class PPORMTorchRLModule(PPOTorchRLModule):
    
    def save_to_checkpoint(self, checkpoint_dir_path: Union[str, pathlib.Path]) -> None:
        """Saves the module to a checkpoint directory.

        Args:
            checkpoint_dir_path: The directory to save the checkpoint to.

        Raises:
            ValueError: If dir_path is not an absolute path.
        """
        path = pathlib.Path(checkpoint_dir_path)
        path.mkdir(parents=True, exist_ok=True)
        module_state_dir = path / RLMODULE_STATE_DIR_NAME
        module_state_dir.mkdir(parents=True, exist_ok=True)
        self.save_state(module_state_dir)
        self._save_module_metadata(path, PPORMSingleAgentRLModuleSpec)

class PPORMMultiAgentRLModuleConfig(MultiAgentRLModuleConfig):

    def to_dict(self):
        return {
            "modules": {
                module_id: module_spec.to_dict()
                for module_id, module_spec in self.modules.items()
            }
        }

    @classmethod
    def from_dict(cls, d) -> "PPORMMultiAgentRLModuleConfig":
        return cls(
            modules={
                module_id: PPORMSingleAgentRLModuleSpec.from_dict(module_spec)
                for module_id, module_spec in d["modules"].items()
            }
        )

class PPORMMultiAgentRLModule(MultiAgentRLModule):

    def __init__(self, config: Optional["MultiAgentRLModuleConfig"] = None) -> None:
        """Initializes a MultiagentRLModule instance.

        Args:
            config: The MultiAgentRLModuleConfig to use.
        """
        super().__init__(config or PPORMMultiAgentRLModuleConfig())

    @override(RLModule)
    def save_to_checkpoint(self, checkpoint_dir_path: Union[str, pathlib.Path]) -> None:
        path = pathlib.Path(checkpoint_dir_path)
        path.mkdir(parents=True, exist_ok=True)
        self.save_state(path)
        self._save_module_metadata(path, PPORMMultiAgentRLModuleSpec)

    @override(RLModule)
    def load_state(
        self,
        path: Union[str, pathlib.Path],
        modules_to_load: Optional[Set[ModuleID]] = None,
    ) -> None:
        """Loads the weights of an MultiAgentRLModule from dir.

        NOTE:
            If you want to load a module that is not already
            in this MultiAgentRLModule, you should add it to this MultiAgentRLModule
            before loading the checkpoint.

        Args:
            path: The path to the directory to load the state from.
            modules_to_load: The modules whose state is to be loaded from the path. If
                this is None, all modules that are checkpointed will be loaded into this
                marl module.


        """
        path = pathlib.Path(path)
        if not modules_to_load:
            modules_to_load = set(self._rl_modules.keys())
        path.mkdir(parents=True, exist_ok=True)
        for submodule_id in modules_to_load:
            if submodule_id not in self._rl_modules:
                raise ValueError(
                    f"Module {submodule_id} from `modules_to_load`: "
                    f"{modules_to_load} not found in this MultiAgentRLModule."
                )
            submodule = self._rl_modules[submodule_id]
            submodule_weights_dir = path / submodule_id / RLMODULE_STATE_DIR_NAME
            if not submodule_weights_dir.exists():
                raise ValueError(
                    f"Submodule {submodule_id}'s module state directory: "
                    f"{submodule_weights_dir} not found in checkpoint dir {path}."
                )
            # --LEO_DEBUG
            try:
                submodule.load_state(submodule_weights_dir)
            except Exception as e:
                continue
            # LEO_DEBUG--

@dataclass
class PPORMMultiAgentRLModuleSpec(MultiAgentRLModuleSpec):
    marl_module_class: Type[MultiAgentRLModule] = PPORMMultiAgentRLModule

    def get_marl_config(self) -> "PPORMMultiAgentRLModuleConfig":
        """Returns the MultiAgentRLModuleConfig for this spec."""
        return PPORMMultiAgentRLModuleConfig(modules=self.module_specs)

    @classmethod
    def from_module(self, module: PPORMMultiAgentRLModule) -> "PPORMMultiAgentRLModuleSpec":
        """Creates a MultiAgentRLModuleSpec from a MultiAgentRLModule.

        Args:
            module: The MultiAgentRLModule to create the spec from.

        Returns:
            The MultiAgentRLModuleSpec.
        """
        # we want to get the spec of the underlying unwrapped module that way we can
        # easily reconstruct it. The only wrappers that we expect to support today are
        # wrappers that allow us to do distributed training. Those will be added back
        # by the learner if necessary.
        module_specs = {
            module_id: PPORMSingleAgentRLModuleSpec.from_module(rl_module.unwrapped())
            for module_id, rl_module in module._rl_modules.items()
        }
        marl_module_class = module.__class__
        return PPORMMultiAgentRLModuleSpec(
            marl_module_class=marl_module_class, module_specs=module_specs
        )

    @classmethod
    def from_dict(cls, d) -> "PPORMMultiAgentRLModuleSpec":
        """Creates a MultiAgentRLModuleSpec from a dictionary."""
        return PPORMMultiAgentRLModuleSpec(
            marl_module_class=deserialize_type(d["marl_module_class"]),
            module_specs={
                module_id: PPORMSingleAgentRLModuleSpec.from_dict(module_spec)
                for module_id, module_spec in d["module_specs"].items()
            },
        )

class PPORMSingleAgentRLModuleSpec(SingleAgentRLModuleSpec):
    def __post_init__(self):
        pass

    @classmethod
    def from_dict(cls, d):
        """Returns a single agent RLModule spec from a serialized representation."""
        module_class = deserialize_type(d["module_class"])

        module_config = RLModuleConfig.from_dict(d["module_config"])
        observation_space = module_config.observation_space
        action_space = module_config.action_space
        model_config_dict = module_config.model_config_dict
        catalog_class = module_config.catalog_class

        spec = PPORMSingleAgentRLModuleSpec(
            module_class=module_class,
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
            catalog_class=catalog_class,
        )
        return spec

    @classmethod
    def from_module(cls, module: "PPORMTorchRLModule") -> "PPORMSingleAgentRLModuleSpec":

        if isinstance(module, MultiAgentRLModule):
            raise ValueError(
                "MultiAgentRLModule cannot be converted to PPORMSingleAgentRLModuleSpec."
            )

        return PPORMSingleAgentRLModuleSpec(
            module_class=type(module),
            observation_space=module.config.observation_space,
            action_space=module.config.action_space,
            model_config_dict=module.config.model_config_dict,
            catalog_class=module.config.catalog_class,
        )
    
    def as_multi_agent(self) -> "PPORMMultiAgentRLModuleSpec":

        return PPORMMultiAgentRLModuleSpec(
            module_specs={DEFAULT_MODULE_ID: self},
            load_state_path=self.load_state_path,
        )

class PPORMConfig(PPOConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class or PPORM)

        self.shared_hidden_layers: tuple = tuple() # (0, 1)

    def get_catalog(self):
        return PPORMLearningSharedLayersCatalog if self.shared_hidden_layers else PPORMLearningCatalog

    def get_marl_module_spec(
        self,
        *,
        policy_dict: Optional[Dict[str, PolicySpec]] = None,
        single_agent_rl_module_spec: Optional[SingleAgentRLModuleSpec] = None,
        env: Optional[EnvType] = None,
        spaces: Optional[Dict[PolicyID, Tuple[gym.Space, gym.Space]]] = None,
        inference_only: bool = False,
    ) -> PPORMMultiAgentRLModuleSpec:
        """Returns the MultiAgentRLModule spec based on the given policy spec dict.

        policy_dict could be a partial dict of the policies that we need to turn into
        an equivalent multi-agent RLModule spec.

        Args:
            policy_dict: The policy spec dict. Using this dict, we can determine the
                inferred values for observation_space, action_space, and config for
                each policy. If the module spec does not have these values specified,
                they will get auto-filled with these values obtrained from the policy
                spec dict. Here we are relying on the policy's logic for infering these
                values from other sources of information (e.g. environement)
            single_agent_rl_module_spec: The SingleAgentRLModuleSpec to use for
                constructing a MultiAgentRLModuleSpec. If None, the already
                configured spec (`self._rl_module_spec`) or the default RLModuleSpec for
                this algorithm (`self.get_default_rl_module_spec()`) will be used.
            env: An optional env instance, from which to infer the different spaces for
                the different SingleAgentRLModules. If not provided, will try to infer
                from `spaces`. Otherwise from `self.observation_space` and
                `self.action_space`. If no information on spaces can be infered, will
                raise an error.
            spaces: Optional dict mapping policy IDs to tuples of 1) observation space
                and 2) action space that should be used for the respective policy.
                These spaces were usually provided by an already instantiated remote
                EnvRunner. If not provided, will try to infer from `env`. Otherwise
                from `self.observation_space` and `self.action_space`. If no
                information on spaces can be inferred, will raise an error.
            inference_only: If `True`, the module spec will be used in either
                sampling or inference and can be built in its light version (if
                available), i.e. it contains only the networks needed for acting in the
                environment (no target or critic networks).
        """
        # TODO (Kourosh,sven): When we replace policy entirely there will be no need for
        #  this function to map policy_dict to marl_module_specs anymore. The module
        #  spec will be directly given by the user or inferred from env and spaces.
        if policy_dict is None:
            policy_dict, _ = self.get_multi_agent_setup(env=env, spaces=spaces)

        # TODO (Kourosh): Raise an error if the config is not frozen
        # If the module is single-agent convert it to multi-agent spec

        # The default RLModuleSpec (might be multi-agent or single-agent).
        default_rl_module_spec = self.get_default_rl_module_spec()
        # The currently configured RLModuleSpec (might be multi-agent or single-agent).
        # If None, use the default one.
        current_rl_module_spec = self._rl_module_spec or default_rl_module_spec

        # Algorithm is currently setup as a single-agent one.
        if isinstance(current_rl_module_spec, SingleAgentRLModuleSpec):
            # Use either the provided `single_agent_rl_module_spec` (a
            # SingleAgentRLModuleSpec), the currently configured one of this
            # AlgorithmConfig object, or the default one.
            single_agent_rl_module_spec = (
                single_agent_rl_module_spec or current_rl_module_spec
            )
            # Now construct the proper MultiAgentRLModuleSpec.
            marl_module_spec = PPORMMultiAgentRLModuleSpec(
                module_specs={
                    k: copy.deepcopy(single_agent_rl_module_spec)
                    for k in policy_dict.keys()
                },
            )

        # Algorithm is currently setup as a multi-agent one.
        else:
            # The user currently has a MultiAgentSpec setup (either via
            # self._rl_module_spec or the default spec of this AlgorithmConfig).
            assert isinstance(current_rl_module_spec, MultiAgentRLModuleSpec)

            # Default is single-agent but the user has provided a multi-agent spec
            # so the use-case is multi-agent.
            if isinstance(default_rl_module_spec, SingleAgentRLModuleSpec):
                # The individual (single-agent) module specs are defined by the user
                # in the currently setup MultiAgentRLModuleSpec -> Use that
                # SingleAgentRLModuleSpec.
                if isinstance(
                    current_rl_module_spec.module_specs, SingleAgentRLModuleSpec
                ):
                    single_agent_spec = single_agent_rl_module_spec or (
                        current_rl_module_spec.module_specs
                    )
                    module_specs = {
                        k: copy.deepcopy(single_agent_spec) for k in policy_dict.keys()
                    }

                # The individual (single-agent) module specs have not been configured
                # via this AlgorithmConfig object -> Use provided single-agent spec or
                # the the default spec (which is also a SingleAgentRLModuleSpec in this
                # case).
                else:
                    single_agent_spec = (
                        single_agent_rl_module_spec or default_rl_module_spec
                    )
                    module_specs = {
                        k: copy.deepcopy(
                            current_rl_module_spec.module_specs.get(
                                k, single_agent_spec
                            )
                        )
                        for k in policy_dict.keys()
                    }

                # Now construct the proper MultiAgentRLModuleSpec.
                # We need to infer the multi-agent class from `current_rl_module_spec`
                # and fill in the module_specs dict.
                marl_module_spec = current_rl_module_spec.__class__(
                    marl_module_class=current_rl_module_spec.marl_module_class,
                    module_specs=module_specs,
                    modules_to_load=current_rl_module_spec.modules_to_load,
                    load_state_path=current_rl_module_spec.load_state_path,
                )

            # Default is multi-agent and user wants to override it -> Don't use the
            # default.
            else:
                # Use has given an override SingleAgentRLModuleSpec -> Use this to
                # construct the individual RLModules within the MultiAgentRLModuleSpec.
                if single_agent_rl_module_spec is not None:
                    pass
                # User has NOT provided an override SingleAgentRLModuleSpec.
                else:
                    # But the currently setup multi-agent spec has a SingleAgentRLModule
                    # spec defined -> Use that to construct the individual RLModules
                    # within the MultiAgentRLModuleSpec.
                    if isinstance(
                        current_rl_module_spec.module_specs, SingleAgentRLModuleSpec
                    ):
                        # The individual module specs are not given, it is given as one
                        # SingleAgentRLModuleSpec to be re-used for all
                        single_agent_rl_module_spec = (
                            current_rl_module_spec.module_specs
                        )
                    # The currently setup multi-agent spec has NO
                    # SingleAgentRLModuleSpec in it -> Error (there is no way we can
                    # infer this information from anywhere at this point).
                    else:
                        raise ValueError(
                            "We have a MultiAgentRLModuleSpec "
                            f"({current_rl_module_spec}), but no "
                            "`SingleAgentRLModuleSpec`s to compile the individual "
                            "RLModules' specs! Use "
                            "`AlgorithmConfig.get_marl_module_spec("
                            "policy_dict=.., single_agent_rl_module_spec=..)`."
                        )

                # Now construct the proper MultiAgentRLModuleSpec.
                marl_module_spec = current_rl_module_spec.__class__(
                    marl_module_class=current_rl_module_spec.marl_module_class,
                    module_specs={
                        k: copy.deepcopy(single_agent_rl_module_spec)
                        for k in policy_dict.keys()
                    },
                    modules_to_load=current_rl_module_spec.modules_to_load,
                    load_state_path=current_rl_module_spec.load_state_path,
                )

        # Make sure that policy_dict and marl_module_spec have similar keys
        if set(policy_dict.keys()) != set(marl_module_spec.module_specs.keys()):
            raise ValueError(
                "Policy dict and module spec have different keys! \n"
                f"policy_dict keys: {list(policy_dict.keys())} \n"
                f"module_spec keys: {list(marl_module_spec.module_specs.keys())}"
            )

        # Fill in the missing values from the specs that we already have. By combining
        # PolicySpecs and the default RLModuleSpec.

        for module_id in policy_dict:
            policy_spec = policy_dict[module_id]
            module_spec = marl_module_spec.module_specs[module_id]
            if module_spec.module_class is None:
                if isinstance(default_rl_module_spec, SingleAgentRLModuleSpec):
                    module_spec.module_class = default_rl_module_spec.module_class
                elif isinstance(
                    default_rl_module_spec.module_specs, SingleAgentRLModuleSpec
                ):
                    module_class = default_rl_module_spec.module_specs.module_class
                    # This should be already checked in validate() but we check it
                    # again here just in case
                    if module_class is None:
                        raise ValueError(
                            "The default rl_module spec cannot have an empty "
                            "module_class under its SingleAgentRLModuleSpec."
                        )
                    module_spec.module_class = module_class
                elif module_id in default_rl_module_spec.module_specs:
                    module_spec.module_class = default_rl_module_spec.module_specs[
                        module_id
                    ].module_class
                else:
                    raise ValueError(
                        f"Module class for module {module_id} cannot be inferred. "
                        f"It is neither provided in the rl_module_spec that "
                        "is passed in nor in the default module spec used in "
                        "the algorithm."
                    )
            if module_spec.catalog_class is None:
                if isinstance(default_rl_module_spec, SingleAgentRLModuleSpec):
                    module_spec.catalog_class = default_rl_module_spec.catalog_class
                elif isinstance(
                    default_rl_module_spec.module_specs, SingleAgentRLModuleSpec
                ):
                    catalog_class = default_rl_module_spec.module_specs.catalog_class
                    module_spec.catalog_class = catalog_class
                elif module_id in default_rl_module_spec.module_specs:
                    module_spec.catalog_class = default_rl_module_spec.module_specs[
                        module_id
                    ].catalog_class
                else:
                    raise ValueError(
                        f"Catalog class for module {module_id} cannot be inferred. "
                        f"It is neither provided in the rl_module_spec that "
                        "is passed in nor in the default module spec used in "
                        "the algorithm."
                    )
            # TODO (sven): Find a good way to pack module specific parameters from
            # the algorithms into the `model_config_dict`.
            if module_spec.observation_space is None:
                module_spec.observation_space = policy_spec.observation_space
            if module_spec.action_space is None:
                module_spec.action_space = policy_spec.action_space
            # In case the `RLModuleSpec` does not have a model config dict, we use the
            # the one defined by the auto keys and the `model_config_dict` arguments in
            # `self.rl_module()`.
            if module_spec.model_config_dict is None:
                module_spec.model_config_dict = self.model_config
            # Otherwise we combine the two dictionaries where settings from the
            # `RLModuleSpec` have higher priority.
            else:
                module_spec.model_config_dict = (
                    self.model_config | module_spec.model_config_dict
                )
            # Set the `inference_only` flag for the module spec.
            module_spec.model_config_dict[INFERENCE_ONLY] = inference_only

        return marl_module_spec


    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> PPORMSingleAgentRLModuleSpec:

        if self.framework_str == "torch":

            return PPORMSingleAgentRLModuleSpec(
                module_class=PPORMTorchRLModule, catalog_class=self.get_catalog()
            )
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Only 'torch' is supported."
            )

    def network(
        self,
        *,
        shared_hidden_layers: Optional[tuple] = NotProvided,
    ) -> AlgorithmConfig:

        if shared_hidden_layers is not NotProvided:
            self.shared_hidden_layers = shared_hidden_layers

        return self
    
    @property
    @override(AlgorithmConfig)
    def _model_config_auto_includes(self) -> Dict[str, Any]:
        return super()._model_config_auto_includes | {"vf_share_layers": True}

class PPORM(PPO):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMConfig()

    @PublicAPI
    def get_rm(self) -> RewardMachine:
        def _get_rm(w):
            return w.env.get_rm()

        rms = self.workers.foreach_worker(_get_rm)
        from collections import Counter
        assert len(set(rms)) == 1, f"More than 1 RM: {Counter(rms)}"
        return list(set(rms))[0]

    @PublicAPI
    def get_action_space(self) -> gym.Space:
        def _get_action_space(w):
            env = w.env
            # `env` is a gymnasium.vector.Env.
            if hasattr(env, "single_action_space") and isinstance(
                env.single_action_space, gym.Space
            ):
                return space_to_dict(env.single_action_space)
            # `env` is a gymnasium.Env.
            elif hasattr(env, "action_space") and isinstance(
                env.action_space, gym.Space
            ):
                return space_to_dict(env.action_space)

            return None

        action_spaces = self.workers.foreach_worker(_get_action_space)
        return space_from_dict(action_spaces[0])

    @PublicAPI
    def get_obs_space(self) -> gym.Space:
        def _get_obs_space(w):
            env = w.env
            if hasattr(env, "single_observation_space") and isinstance(
                env.single_observation_space, gym.Space
            ):
                return space_to_dict(env.single_observation_space)
            # `env` is a gymnasium.Env.
            elif hasattr(env, "observation_space") and isinstance(
                env.observation_space, gym.Space
            ):
                return space_to_dict(env.observation_space)
            
            return None

        obs_spaces = self.workers.foreach_worker(_get_obs_space)
        return space_from_dict(obs_spaces[0])

    @PublicAPI
    def set_rm(self, rm: RewardMachine) -> None:
        def _set_rm(w):
            w.env.set_rm(rm)

        self.workers.foreach_worker(_set_rm)

        if self.evaluation_workers is not None:
            self.evaluation_workers.foreach_worker(_set_rm)

    def reset_policies(self) -> None:

        obs_spaces = self.get_obs_space()
        act_spaces = self.get_action_space()

        def _update_config(w):
            w.config._is_frozen = False
            w.config._rl_module_spec = None

            rl_module_spec = w.config.get_marl_module_spec(
                spaces={
                    pid: (
                        obs_spaces.get(pid, next(iter(obs_spaces.values()))), 
                        act_spaces.get(pid, next(iter(act_spaces.values()))),
                    )
                    for pid in w.config.policies
                }
            )
            w.config.rl_module(
                rl_module_spec=rl_module_spec
            )
            w.config._is_frozen = True

        def _reset_worker(w):
            _update_config(w)

            w._env_to_module = w.config.build_env_to_module_connector(w.env)
            w._cached_to_module = None
            w.module = w._make_module()
            w._module_to_env = w.config.build_module_to_env_connector(w.env)
            w._needs_initial_reset = True

        self.workers.foreach_worker(_reset_worker)

        if self.evaluation_workers is not None:
            self.evaluation_workers.foreach_worker(_reset_worker)

        _update_config(self)

        def _reset_learner(l):
            _update_config(l)

            l._module_spec = l.config.rl_module_spec
            l._module = l._make_module()
            l.configure_optimizers()

        self.learner_group.foreach_learner(_reset_learner)

tune.register_trainable("PPORM", PPORM)

class PPORMLearningConfig(PPORMConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class or PPORMLearning)

        self.rm_learning_freq = 1
        self.traces_buffer = SharedTracesBuffer.SHARED_TRACES_BUFFER_REF
        self.rm_learner_class = None
        self.rm_learner_kws = None

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> PPORMSingleAgentRLModuleSpec:

        if self.framework_str == "torch":

            return PPORMSingleAgentRLModuleSpec(
                module_class=PPORMTorchRLModule, 
                model_config_dict=self._model_config_dict,
                catalog_class=self.get_catalog()
            )
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Only 'torch' is supported."
            )

    def rm(
        self,
        *,
        rm_learning_freq: Optional[int] = NotProvided,
        traces_buffer: Optional[str] = NotProvided,
        rm_learner_class: Optional[type] = NotProvided,
        rm_learner_kws: Optional[dict] = NotProvided,
    ) -> AlgorithmConfig:

        if rm_learning_freq is not NotProvided:
            self.rm_learning_freq = rm_learning_freq

        if traces_buffer is not NotProvided:
            self.traces_buffer = traces_buffer

        if rm_learner_class is not NotProvided:
            self.rm_learner_class = rm_learner_class

        if rm_learner_kws is not NotProvided:
            self.rm_learner_kws = rm_learner_kws

        return self

    def validate(self) -> None:
        super().validate()

        if self.traces_buffer is None:
            raise ValueError("A `traces_buffer` must be provided.")

        if self.rm_learner_class is None:
            raise ValueError("A `rm_learner_class` must be provided.")

class PPORMLearning(PPORM):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMLearningConfig()

    @override(PPO)
    def setup(self, config: AlgorithmConfig) -> None:
        self.traces_buffer = self.create_shared_traces_buffer(config.traces_buffer)
        self.rm_learning_freq = self.config.rm_learning_freq
        
        if config.rm_learner_class:
            self.rm_learner = config.rm_learner_class(**config.rm_learner_kws)
            self.rm_learner.set_log_folder(os.path.join(self._logdir, "rm"))
            self.current_rm = None
        else:
            self.rm_learner = None

        super().setup(config)

    @staticmethod
    def create_shared_traces_buffer(name):
        return SharedTracesBuffer.options(name=name, lifetime="detached").remote()

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMLearningConfig()

    def _rm_to_img(self, rm: RewardMachine) -> None:
        import io

        from PIL import Image

        data = io.BytesIO()
        data.write(rm.to_digraph().pipe(format="png"))
        data.seek(0)

        return Image.open(data)

    # @override(PPORM)
    # def get_rm(self) -> RewardMachine:
    #     return self.current_rm or super().get_rm()

    @override(PPORM)
    def set_rm(self, rm: RewardMachine) -> None:
        self.current_rm = rm
        super().set_rm(rm)

    @override(Algorithm)
    def __getstate__(self) -> Dict:
        """ """
        state = super().__getstate__()
        if hasattr(self, "current_rm"):
            state["current_rm"] = self.current_rm
        return state

    @override(Algorithm)
    def __setstate__(self, state) -> None:
        """ """
        self.current_rm = state.pop("current_rm", None)
        super().__setstate__(state)

    def _learn_rm(self):
        if (
            self._timers[TRAINING_ITERATION_TIMER].count and
            self._timers[TRAINING_ITERATION_TIMER].count % self.rm_learning_freq
            == 0
        ):
            pos, dend, inc = ray.get(self.traces_buffer.get_all_examples.remote())
            rm = self.get_rm()

            candidate_rm = self.rm_learner.learn(rm, pos, dend, inc)
            if candidate_rm:
                self.patience = 0
                self.set_rm(candidate_rm)
                self.reset_policies()

                # DEBUG
                assert (
                    candidate_rm == self.get_rm()
                ), "Something went wrong when setting the new RM"

                current_rm_plot = os.path.join(self._logdir, "rm", "current")
                candidate_rm.plot(current_rm_plot)
            # else:
            #     self.patience += 1
            #     if self.patience > 5:
            #         self.rm_learning_freq = 2 * self.rm_learning_freq
            #         self.patience = 0
                
    @override(Algorithm)
    def training_step(self) -> ResultDict:
        # Collect batches from sample workers until we have a full batch.
        with self.metrics.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
            # Sample in parallel from the workers.
            if self.config.count_steps_by == "agent_steps":
                episodes, env_runner_results = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config.total_train_batch_size,
                    sample_timeout_s=self.config.sample_timeout_s,
                    _uses_new_env_runners=(
                        self.config.enable_env_runner_and_connector_v2
                    ),
                    _return_metrics=True,
                )
            else:
                episodes, env_runner_results = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_env_steps=self.config.total_train_batch_size,
                    sample_timeout_s=self.config.sample_timeout_s,
                    _uses_new_env_runners=(
                        self.config.enable_env_runner_and_connector_v2
                    ),
                    _return_metrics=True,
                )
            # Return early if all our workers failed.
            if not episodes:
                return {}

            # Reduce EnvRunner metrics over the n EnvRunners.
            self.metrics.merge_and_log_n_dicts(
                env_runner_results, key=ENV_RUNNER_RESULTS
            )
            # Log lifetime counts for env- and agent steps.
            self.metrics.log_dict(
                {
                    NUM_AGENT_STEPS_SAMPLED_LIFETIME: self.metrics.peek(
                        ENV_RUNNER_RESULTS, NUM_AGENT_STEPS_SAMPLED
                    ),
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: self.metrics.peek(
                        ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED
                    ),
                    NUM_EPISODES_LIFETIME: self.metrics.peek(
                        ENV_RUNNER_RESULTS, NUM_EPISODES
                    ),
                },
                reduce="sum",
            )

        # Perform a learner update step on the collected episodes.
        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            learner_results = self.learner_group.update_from_episodes(
                episodes=episodes,
                timesteps={
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: (
                        self.metrics.peek(NUM_ENV_STEPS_SAMPLED_LIFETIME)
                    ),
                },
                minibatch_size=(
                    self.config.mini_batch_size_per_learner
                    or self.config.sgd_minibatch_size
                ),
                num_iters=self.config.num_sgd_iter,
            )
            self.metrics.merge_and_log_n_dicts(learner_results, key=LEARNER_RESULTS)
            self.metrics.log_dict(
                {
                    NUM_ENV_STEPS_TRAINED_LIFETIME: self.metrics.peek(
                        LEARNER_RESULTS, ALL_MODULES, NUM_ENV_STEPS_TRAINED
                    ),
                    # NUM_MODULE_STEPS_TRAINED_LIFETIME: self.metrics.peek(
                    #    LEARNER_RESULTS, NUM_MODULE_STEPS_TRAINED
                    # ),
                },
                reduce="sum",
            )

        with self.metrics.log_time((TIMERS, LEARNER_RM_TIMER)):
            self._learn_rm()

        # Update weights - after learning on the local worker - on all remote
        # workers.
        with self.metrics.log_time((TIMERS, SYNCH_WORKER_WEIGHTS_TIMER)):
            # The train results's loss keys are ModuleIDs to their loss values.
            # But we also return a total_loss key at the same level as the ModuleID
            # keys. So we need to subtract that to get the correct set of ModuleIDs to
            # update.
            # TODO (sven): We should also not be using `learner_results` as a messenger
            #  to infer which modules to update. `policies_to_train` might also NOT work
            #  as it might be a very large set (100s of Modules) vs a smaller Modules
            #  set that's present in the current train batch.
            modules_to_update = set(learner_results[0].keys()) - {ALL_MODULES}
            if self.workers.num_remote_workers() > 0:
                self.workers.sync_weights(
                    # Sync weights from learner_group to all rollout workers.
                    from_worker_or_learner_group=self.learner_group,
                    policies=modules_to_update,
                    inference_only=True,
                )
            else:
                weights = self.learner_group.get_weights(inference_only=True)
                self.workers.local_worker().set_weights(weights)

        with self.metrics.log_time((TIMERS, LEARNER_ADDITIONAL_UPDATE_TIMER)):
            kl_dict = {}
            if self.config.use_kl_loss:
                for mid in modules_to_update:
                    kl = convert_to_numpy(
                        self.metrics.peek(LEARNER_RESULTS, mid, LEARNER_RESULTS_KL_KEY)
                    )
                    if np.isnan(kl):
                        logger.warning(
                            f"KL divergence for Module {mid} is non-finite, this "
                            "will likely destabilize your model and the training "
                            "process. Action(s) in a specific state have near-zero "
                            "probability. This can happen naturally in deterministic "
                            "environments where the optimal policy has zero mass for a "
                            "specific action. To fix this issue, consider setting "
                            "`kl_coeff` to 0.0 or increasing `entropy_coeff` in your "
                            "config."
                        )
                    kl_dict[mid] = kl

            # TODO (sven): Move to Learner._after_gradient_based_update().
            # Triggers a special update method on RLOptimizer to update the KL values.
            additional_results = self.learner_group.additional_update(
                module_ids_to_update=modules_to_update,
                sampled_kl_values=kl_dict,
                timestep=self.metrics.peek(NUM_ENV_STEPS_SAMPLED_LIFETIME),
            )
            self.metrics.merge_and_log_n_dicts(additional_results, key=LEARNER_RESULTS)

        return self.metrics.reduce()

tune.register_trainable("PPORMLearning", PPORMLearning)