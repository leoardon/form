import abc
import copy
import itertools
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

from ..reward_machine import RewardMachine
from ..utils.trace import TraceTracker


class LabelingFunctionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.prev_obs = None

    def filter_labels(self, labels, u):
        return labels

    @abc.abstractmethod
    def get_labels(self, obs: dict, prev_obs: dict):
        raise NotImplementedError("get_labels")

    @abc.abstractmethod
    def get_all_labels(self):
        raise NotImplementedError("get_all_labels")

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["labels"] = self.get_labels(observation, self.prev_obs)
        self.prev_obs = copy.deepcopy(observation)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with kwargs."""
        obs, info = super().reset(**kwargs)
        info["labels"] = self.get_labels(obs, None)
        self.prev_obs = copy.deepcopy(obs)
        return obs, info

class TraceWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, compressed=True):
        super().__init__(env)

        self.compressed_trace = compressed
        self._trace = TraceTracker(compressed=compressed)

    @property
    def trace(self):
        return self._trace.labels_sequence

    @property
    def flat_trace(self):
        return self._trace.flatten_labels_sequence

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        if "labels" in info:
            prev_labels = info["labels"]
            info["original_labels"] = prev_labels
            new_labels = self._trace.update(prev_labels, observation)
            info["labels"] = new_labels
            info["compressed_labels"] = prev_labels != new_labels
            # info["flat_trace"] = self._trace.flatten_labels_sequence

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._trace.reset()
        return obs, info

class AutomataWrapper(gym.Wrapper):
    class LabelMode(Enum):
        ALL = 0
        RM = 1
        STATE = 2

    class TerminationMode(Enum):
        RM = 0
        ENV = 1

    class TruncationMode(Enum):
        ENV = 0
        MISMATCH = 1

    def __init__(
        self,
        env: gym.Env,
        rm: "RewardMachine",
        label_mode: LabelMode = LabelMode.ALL,
        termination_mode: TerminationMode = TerminationMode.ENV,
        truncation_mode: TruncationMode = TruncationMode.MISMATCH,
    ):
        """
        label_mode:
            - ALL: returns all the labels returned by the labeling function
            - RM: returns only the labels present in the RM
            - STATE: returns only the lables that can be observed from the current state
        termination_mode:
            - RM: ends the episode when the RM reaches accepting/rejecting state
            - ENV: ends the episode when the underlying env is returning the end signal
        """
        assert hasattr(env, "get_labels"), "The LabelingFunctionWrapper is required"
        assert hasattr(env, "compressed_trace"), "The TraceWrapper is required"

        self.label_mode = label_mode
        self.termination_mode = termination_mode
        self.truncation_mode = truncation_mode

        self.state_trace_tracker = TraceTracker(compressed=env.compressed_trace)

        self.rm = rm
        self.u = self.rm.u0
        super().__init__(env)

    def filter_labels(self, labels, u):
        return [e for e in labels if self._is_valid_event(e, u)]

    def _is_valid_event(self, event, u):
        if self.label_mode == self.LabelMode.ALL:
            return True
        elif self.label_mode == self.LabelMode.RM:
            return event in self.rm.get_valid_events()
        elif self.label_mode == self.LabelMode.STATE:
            return event in self.rm.get_valid_events(u)

    def set_rm(self, rm):
        self.rm = rm

    def get_rm(self):
        return self.rm

    def step(self, action):
        observation, reward, env_terminated, env_truncated, info = super().step(action)

        info["env_terminated"] = env_terminated
        info["env_truncated"] = env_truncated
        info["labels"] = self.filter_labels(info["labels"], self.u)
        simulated_updates = info.pop("env_simulated_updates", {})

        info["labels"] = self._apply_simulated_updates(
            info["labels"], simulated_updates
        )

        self.state_trace_tracker.update(info["labels"], obs=observation, state=self.u)

        u_next = self.u
        # if the labels have been compressed we know it will remain in the same state.
        if not info.get("compressed_labels", False):
            u_next = self.rm.get_next_state(
                self.u, info["labels"], self.state_trace_tracker.state_labels_sequence
            )

        reward = self._get_reward(reward, u_next)
        self.u = u_next

        terminated = self._get_terminated(env_terminated)
        info["rm_state"] = self.u

        # Assume every trace is positive unless otherwise defined
        if "is_positive_trace" not in info:
            info["is_positive_trace"] = True

        truncated, info_ = self._get_truncated(env_truncated, env_terminated)
        info.update(info_)

        return observation, reward, terminated, truncated, info

    def _get_reward(self, reward, u_next):
        return reward

    def _get_terminated(self, terminated):
        if self.termination_mode == self.TerminationMode.ENV:
            return terminated
        else:  # should be TerminationMode.RM
            return self.rm.is_state_terminal(self.u)

    def _get_truncated(self, truncated, terminated):
        info = {}
        ret = truncated
        if self.truncation_mode == self.TruncationMode.MISMATCH:
            if terminated != self.rm.is_state_terminal(self.u):
                ret = True
                info["wrong_rm"] = True

        return ret, info

    def _apply_simulated_updates(self, original_labels, simulated_updates):
        labels = copy.deepcopy(original_labels)
        for e in original_labels:
            # apply simulated updates to the environment
            if e in simulated_updates:
                labels_update = simulated_updates[e](self.unwrapped)
                if labels_update:
                    labels = labels_update(labels)
        return labels

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.u = self.rm.u0
        self.state_trace_tracker.reset()

        info["labels"] = self.filter_labels(info.get("labels", {}), self.u)
        simulated_updates = info.pop("env_simulated_updates", {})
        info["labels"] = self._apply_simulated_updates(
            info["labels"], simulated_updates
        )

        self.state_trace_tracker.update(info["labels"], obs=obs, state=self.u)
        u_next = self.rm.get_next_state(
            self.u, info["labels"], self.state_trace_tracker.state_labels_sequence
        )
        self.u = u_next

        info["rm_state"] = self.u
        return obs, info

class CRMWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, max_rm_num_states: int = 10):
        assert hasattr(env, "set_rm"), "The AutomataWrapper is required"

        super().__init__(env)

        self._rm_states = sorted([f"u{i}" for i in range(max_rm_num_states)] + ["u_acc", "u_rej"])
        self.observation_space = gym.spaces.Dict(
            {
                **(
                    self.observation_space 
                    if isinstance(self.observation_space, gym.spaces.Dict) 
                    else {"image": self.observation_space}
                ),
                "rm_state": gym.spaces.Box(
                    low=0., 
                    high=max_rm_num_states,
                    dtype=np.float32
                )
            }
        )

    def _get_state_index(self, state):
        return self._rm_states.index(state)

    def observation(self, obs):
        obs = {
            **(obs if isinstance(obs, dict) else {"image": obs}),
            "rm_state": np.array([float(self._get_state_index(self.env.u))], dtype=np.float32)
        }
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "original_observation" in info:
            info["original_observation"] = self.observation(info["original_observation"])
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if "original_observation" in info:
            info["original_observation"] = self.observation(info["original_observation"])
        return self.observation(obs), info

class LabelObservationWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = self._extend_observation_space(
            self.observation_space, self.get_all_labels()
        )
    
    def get_all_labels(self):
        return sorted(self.env.get_all_labels())

    def _extend_observation_space(self, original_obs_space, all_labels):
        if isinstance(original_obs_space, gym.spaces.Box):
            obs_space = {
                "image": original_obs_space
            }
        elif isinstance(original_obs_space, gym.spaces.Dict):
            obs_space = original_obs_space
        else:
            raise ValueError(f"Unsupported space type {type(original_obs_space)}")

        labels_obs_space = gym.spaces.Box(low=0., high=1., shape=(len(all_labels),), dtype=np.float32)

        return gym.spaces.Dict(
            {
                **obs_space,
                "labels": labels_obs_space,
            }
        )

    @staticmethod
    def _process_obs(obs):
        if isinstance(obs, np.ndarray):
            return {
                "image": obs
            }
        elif isinstance(obs, dict):
            return obs
        else:
            raise ValueError(f"Unsupported space type {type(obs)}")

    def _extend_observation(self, original_obs, labels, all_labels):
        obs = self._process_obs(original_obs)

        return {
            **obs,
            "labels": np.array(
                [
                    # TODO - this logic assumes we cannot revert an observation,
                    # if we have seen the proposition that's it we consider it remains true
                    float(prop in labels)
                    for prop in all_labels
                ],
                dtype=np.float32
            )
        }

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        labels = info.get("labels", [])
        obs = self._extend_observation(obs, labels, self.get_all_labels())

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        labels = info.get("labels", [])
        obs = self._extend_observation(obs, labels, self.get_all_labels())

        return obs, reward, terminated, truncated, info

class ObsExtensionAutomataWrapper(LabelObservationWrapper):

    def __init__(self, env: gym.Env, extend: bool = True, shared_policy: bool = False):

        assert hasattr(env, "set_rm"), "The AutomataWrapper is required"

        self._extend = extend
        self._shared_policy = shared_policy

        gym.Wrapper.__init__(self, env)

        for u in self.rm.states:
            _ = self.extend_observation_space(u, self.env.observation_space)

    def extend_observation_space(self, u, observation_space):
        if self._shared_policy:
            extension_labels = set()
            for u_ in self.rm.states:
                extension_labels = extension_labels.union(set(self.rm.extensions_labels(u_, self.get_all_labels())))
            extension_labels = sorted(extension_labels)
        else:
            extension_labels = self.rm.extensions_labels(u, self.get_all_labels())
        if not self._extend or not extension_labels:
            return observation_space

        return super()._extend_observation_space(observation_space, extension_labels)

    def extend_observation(self, u, observation):
        if self._shared_policy:
            extension_labels = set()
            for u_ in self.rm.states:
                extension_labels = extension_labels.union(set(self.rm.extensions_labels(u_, self.get_all_labels())))
            extension_labels = sorted(extension_labels)
        else:
            extension_labels = self.rm.extensions_labels(u, self.get_all_labels())
        if not self._extend or not extension_labels:
            return observation

        state_trace = (
            self.state_trace_tracker.state_labels_sequence
            if self.state_trace_tracker._current_state == u
            else tuple()
        )

        labels = itertools.chain.from_iterable(state_trace)
        return super()._extend_observation(observation, labels, extension_labels)

    def observation(self, observation):
        return self.extend_observation(self.u, observation)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_observation"] = copy.deepcopy(obs)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["original_observation"] = copy.deepcopy(obs)
        return self.observation(obs), info

class RewardMachineWrapper(AutomataWrapper):

    def __init__(
        self,
        env: gym.Env,
        rm: RewardMachine,
        label_mode: AutomataWrapper.LabelMode = AutomataWrapper.LabelMode.ALL,
        termination_mode: AutomataWrapper.TerminationMode = AutomataWrapper.TerminationMode.RM,
        truncation_mode: AutomataWrapper.TruncationMode = AutomataWrapper.TruncationMode.MISMATCH,
        reward_function: callable = None,
    ):
        super().__init__(env, rm, label_mode, termination_mode, truncation_mode)

        self.reward_function = reward_function or self._simple_reward_func

    @staticmethod
    def _simple_reward_func(rm, u, u_next, reward):
        return rm.get_reward(u, u_next)

    def _get_reward(self, reward, u_next):
        return self.reward_function(self.rm, self.u, u_next, reward)
