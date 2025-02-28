import os

import gymnasium as gym
import numpy as np
import ray
from common import make_env
from ray import tune
from ray.rllib.env import EnvContext, MultiAgentEnv
from form import (
    ClingoChecker,
    CRMWrapper,
    ObsExtensionAutomataWrapper,
    PositiveTraceWrapper,
    RewardMachine,
    RewardMachineWrapper,
    Rule,
    retrieve_types,
)


class RMEnv(MultiAgentEnv):

    _RM_STATE_AGENT_FORMAT = "rm_state_{u}"

    def __init__(self, env_config):
        self._skip_env_checking = True

        env = make_env(env_config["env_id"])(env_config)

        self._agent_ids = {self._rm_state_to_agent_id(f"u{i}") for i in range(env_config["max_rm_num_states"])}

        if env_config.get("reward_shaping", True):

            def _reward_shaping(rm_, u_, u_next_, reward_):
                if u_ != u_next_:
                    graph = {k: set(v.values()) for k, v in rm_.transitions.items()}
                    distance_u = (len(graph) - distance(graph, u_, rm_.uacc)) / len(
                        graph
                    )
                    distance_u_next = (
                        len(graph) - distance(graph, u_next_, rm_.uacc)
                    ) / len(graph)
                    return 0.99 * distance_u_next - distance_u
                return reward_

        else:

            def _reward_shaping(rm_, u_, u_next_, reward_):
                return reward_

        self.traces_buffer = None
        if env_config.get("rm_learning", False):
            env = RewardMachineWrapper(
                PositiveTraceWrapper(env),
                self._init_rm(env, env_config),
                reward_function=_reward_shaping,
                termination_mode=RewardMachineWrapper.TerminationMode.ENV,
                truncation_mode=RewardMachineWrapper.TruncationMode.MISMATCH,
            )

            if "shared_traces_buffer" in env_config:
                try:
                    self.traces_buffer = ray.get_actor(env_config["shared_traces_buffer"])
                except ValueError:
                    self.traces_buffer = None
        else:
            types = retrieve_types(env.get_all_labels(), [], [], [])
            true_rm = RewardMachine.load_from_file(
                env_config["rm_path"],
                rule_cls=lambda r: Rule(r, ClingoChecker(r, types)),
            )
            env = RewardMachineWrapper(
                env,
                true_rm,
                reward_function=_reward_shaping,
                termination_mode=RewardMachineWrapper.TerminationMode.ENV,
                truncation_mode=RewardMachineWrapper.TruncationMode.MISMATCH,
            )

        flat_env = ObsExtensionAutomataWrapper(
            env, 
            extend=env_config.get("extend_obs_space", True), 
            shared_policy=env_config.get("shared_policy", False)
        )
        if env_config.get("shared_policy", False):
            flat_env = CRMWrapper(flat_env, max_rm_num_states=env_config.get("max_rm_num_states", 10))
        self.flat_env = flat_env

        self.underlying_observation_space = self.flat_env.observation_space
        self.observation_space = self._build_observation_space()
        self._obs_space_in_preferred_format = True
        
        self.action_space = gym.spaces.Dict(
            {i: self.flat_env.action_space for i in self._agent_ids}
        )
        self._action_space_in_preferred_format = True

        self._seed = env_config["seed"]
        if isinstance(env_config, EnvContext):
            self._seed += env_config.worker_index

        self.action_space.seed(self._seed)
        self.observation_space.seed(self._seed)

        super().__init__()

    def _agent_id_to_rm_state(self, agent_id):
        return agent_id.replace(self._RM_STATE_AGENT_FORMAT.format(u=""), "")

    def _rm_state_to_agent_id(self, rm_state):
        return self._RM_STATE_AGENT_FORMAT.format(u=rm_state)

    def _build_observation_space(self):
        return gym.spaces.Dict(
            {
                i: self.flat_env.extend_observation_space(
                    self._agent_id_to_rm_state(i), 
                    self.underlying_observation_space
                )
                for i in self._agent_ids
            }
        )

    @property
    def max_steps(self):
        return self.flat_env.max_steps

    def _init_rm(self, env, env_config):
        if not env_config.get("restore", False):
            return self._default_rm()
        else:
            # TODO_LEO: change to store the RM learned in the valid format and switch to checkpoint data
            # env_config["restore_path"]
            types = retrieve_types(env.get_all_labels(), [], [], [])
            return RewardMachine.load_from_file(
                env_config["rm_path"],
                rule_cls=lambda r: Rule(r, ClingoChecker(r, types)),
            )

    @staticmethod
    def _default_rm():
        rm = RewardMachine()
        rm.add_states(["u0"])
        rm.set_u0("u0")
        return rm

    def reset(self, *, seed=None, options=None):
        obs, infos = self.flat_env.reset(
            seed=seed or self._seed, 
            options=options
        )

        return {f"rm_state_{self.flat_env.u}": obs}, {
            f"rm_state_{self.flat_env.u}": infos
        }

    def step(self, action_dict):
        action = action_dict[f"rm_state_{self.flat_env.u}"]
        predecessors = set(self.flat_env.rm.traverse(self.flat_env.trace))

        obs, rew, terminated, truncated, info = self.flat_env.step(action)
        
        original_obs = info.get("original_observation", obs)

        if info.get("wrong_rm", False):
            self.add_trace(self.flat_env.trace, terminated, info["is_positive_trace"])

        obs = (
            {f"rm_state_{self.flat_env.u}": obs}
            if not self.flat_env.rm.is_state_terminal(self.flat_env.u)
            else {}
        ) | { # when env is truncated we need to send last obs
            f"rm_state_{s}": self.flat_env.extend_observation(s, original_obs)
            for s in self.flat_env.rm.states
            if truncated and not self.flat_env.rm.is_state_terminal(s) 
            and s != self.flat_env.u
        }

        rew = {f"rm_state_{s}": rew / len(predecessors) for s in predecessors}
        terminated = {
            f"rm_state_{s}": terminated
            for s in self.flat_env.rm.states
            if not self.flat_env.rm.is_state_terminal(s)
        } | {"__all__": terminated}
        truncated = {
            f"rm_state_{s}": truncated
            for s in self.flat_env.rm.states
            if not self.flat_env.rm.is_state_terminal(s)
        } | {"__all__": truncated}
        info = (
            {f"rm_state_{self.flat_env.u}": info}
            if not self.flat_env.rm.is_state_terminal(self.flat_env.u)
            else {}
        )

        return obs, rew, terminated, truncated, info

    def get_rm(self):
        return self.flat_env.get_rm()

    def set_rm(self, rm):
        self.flat_env.set_rm(rm)
        self.underlying_observation_space = self.flat_env.observation_space
        self.observation_space = self._build_observation_space()

    def add_trace(self, trace, terminated, positive):
        if terminated:
            if positive:
                self.add_positive_trace(trace)
            else:
                self.add_dend_trace(trace)
        else:
            self.add_incomplete_trace(trace)

    def add_positive_trace(self, trace):
        if self.traces_buffer:
            ray.get(self.traces_buffer.add_positive.remote(trace))

    def add_incomplete_trace(self, trace):
        if self.traces_buffer:
            ray.get(self.traces_buffer.add_incomplete.remote(trace))

    def add_dend_trace(self, trace):
        if self.traces_buffer:
            ray.get(self.traces_buffer.add_dend.remote(trace))


tune.register_env("form/RMEnv", RMEnv)

def distance(graph, start, goal):
    visited = []  # List to keep track of visited nodes
    queue = [[start]]  # Initialize a queue

    if start == goal:
        return 0

    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in visited:
            neighbours = graph.get(node, [])

            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                if neighbour == goal:
                    return len(new_path) - 1

            visited.append(node)

    return len(graph) * 100
