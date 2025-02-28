from dotenv import load_dotenv

load_dotenv()

import os
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import gymnasium as gym
import numpy as np
from ray import air, train, tune
from ray.air.constants import TRAINING_ITERATION
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MAX,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.typing import ResultDict
from ray.tune import CLIReporter, register_env
from form import (
    ClingoChecker,
    LabelObservationWrapper,
    RewardMachine,
    RewardMachineWrapper,
    Rule,
    TraceWrapper,
    retrieve_types,
)

torch, nn = try_import_torch()

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig

ENV_NAME = "FOLRoom-AllYellow-2"

SHARED_HIDDEN_LAYERS_REF = "actor::shared_layers"

@dataclass
class Args:
    experiment: str = ""
    path: str = ""

    hyperparam: bool = False
    resume: bool = False

    rm: bool = False
    rm_learning: bool = False
    prop: bool = False
    max_rm_num_states: int = 6
    rm_learning_freq: int = 1
    rm_path: str = None
    restore: bool = False
    restore_path: str = ""
    policies_to_train: tuple[str, ...] = tuple()
    
    shared_policy: bool = False
    reward_shaping: bool = True
    use_lstm: bool = False
    use_labels: bool = False

    num_iterations: int = 5000
    num_timesteps: int = int(1e10)
    reward_threshold: float = 1.
    eval: bool = False
    render: bool = False

    env: str = ENV_NAME
    seed: int = 123
    debug: bool = False
    local: bool = False
    num_workers: int = 10
    max_concurrent_trials: int | None = None
    num_samples: int = 50

    ###### fixed

    shared_hidden_layers: tuple = tuple() # (1,) # deprecated for now
    extend_obs_space: bool = True
    use_cnn: bool = True

    hidden_dim: int = 64
    hidden_count: int = 2
    activation: str = "tanh"
    vf_share_layers: bool = True

    def __post_init__(self):

        if self.shared_hidden_layers:
            raise NotImplementedError("shared hidden layers are not supported at the moment")

        assert not self.shared_hidden_layers or (
            self.shared_hidden_layers and not self.shared_policy
        ), "cannot have shared layers and shared policy simultaneously"

        if self.rm_path is None:
            self.rm_path = os.path.join(
                os.path.dirname(__file__),
                f"data/fol_room/rm/{self.env}.txt",
            )

        register_env(f"form/{self.env}", make_env(f"form/{self.env}"))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_env(env_id):
    def thunk(_env_ctx):

        from form.env.fol_room import (
            FullyObsWrapper,
            IdentifierObsWrapper,
            ImgObsWrapper,
        )

        env = gym.make(env_id)
        env.reset()
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if _env_ctx["use_cnn"]:
            env = FullyObsWrapper(env)
            env = ImgObsWrapper(env)
        else:
            env = IdentifierObsWrapper(env)
            env = gym.wrappers.FlattenObservation(env)

        env = gym.experimental.wrappers.DtypeObservationV0(env, **{"dtype": np.float32})
        env = gym.wrappers.TransformObservation(
            env, **{"f": lambda obs: obs.astype("float32")}
        )

        if _env_ctx["use_labels"]:
            env = LabelObservationWrapper(env)

        env = TraceWrapper(env, compressed=_env_ctx.get("compressed_trace", True))

        types = retrieve_types(env.get_all_labels(), [], [], [])
        true_rm = RewardMachine.load_from_file(
            os.path.join(
                os.path.dirname(__file__),
                f"data/fol_room/rm/{_env_ctx['env_id'].rsplit('/')[-1]}.txt",
            ),
            rule_cls=lambda r: Rule(r, ClingoChecker(r, types)),
        )
        env = RewardMachineWrapper(
            env,
            true_rm,
            termination_mode=RewardMachineWrapper.TerminationMode.RM,
            truncation_mode=RewardMachineWrapper.TruncationMode.ENV,
            reward_function=lambda rm_, u_, u_next_, reward_: rm_.get_reward(
                u_, u_next_
            )
            * reward_,
        )

        return env

    return thunk

def hyperparams_opt(
    num_iterations=400,
    seed=123,
    points_to_evaluate=None,
    num_samples=50,
    max_concurrent_trials=None,
):

    from ray.tune.schedulers import (
        AsyncHyperBandScheduler,
        PopulationBasedTraining,
        create_scheduler,
    )
    from ray.tune.search import BasicVariantGenerator
    from ray.tune.search.optuna import OptunaSearch

    hyperparam_bounds = {
        "lambda": (0.9, 1.0), # 0.97
        "clip_param": (0.01, 0.5), # 0.15
        "vf_clip_param": (0.01, 0.5), # 0.1
        "entropy_coeff": (0.0, 0.1), # 0.05
        "kl_target": (0.0, 0.2), # 0.15
    }

    hyperparam_mutations = {k: tune.uniform(*v) for k, v in hyperparam_bounds.items()}

    scheduler_name = "asynchyperband"
    # scheduler_name = "medianstopping"
    # scheduler_name = "hyperband"

    scheduler = create_scheduler(
        scheduler_name,
        time_attr="training_iteration",
        perturbation_interval=50,
        resample_probability=0.1,
        # Specifies the mutations of these hyperparams
        hyperparam_bounds=hyperparam_bounds,
        hyperparam_mutations=hyperparam_mutations,
        max_t=num_iterations,
        grace_period=int(num_iterations / 10 * 2),
        # hard_stop=False
    )

    if points_to_evaluate:
        points_to_evaluate = [
            {k: v for k, v in d.items() if k in hyperparam_mutations}
            for d in points_to_evaluate
        ]

    return dict(
        tune_config=tune.TuneConfig(
            metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            reuse_actors=False,
            search_alg=OptunaSearch(
                space=hyperparam_mutations,
                metric=[
                    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MIN}",
                    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MAX}",
                ],
                mode=["max", "max", "max"],
                seed=seed,
                points_to_evaluate=points_to_evaluate,
            ),
        )
    )

def run_rllib_experiment(
    base_config: "AlgorithmConfig",
    args: Args,
    *,
    tune_callbacks: Optional[List] = None,
) -> Union[ResultDict, tune.result_grid.ResultGrid]:

    stop = {
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.reward_threshold,
        # f"{NUM_ENV_STEPS_SAMPLED_LIFETIME}": args.num_timesteps,
        TRAINING_ITERATION: args.num_iterations,
    }

    config = base_config

    # Log results using WandB.
    tune_callbacks = tune_callbacks or []
    if hasattr(args, "wandb_key") and args.wandb_key is not None:
        project = args.wandb_project or (
            args.algo.lower() + "-" + re.sub("\\W+", "-", str(config.env).lower())
        )
        tune_callbacks.append(
            WandbLoggerCallback(
                api_key=args.wandb_key,
                project=project,
                upload_checkpoints=True,
                **({"name": args.wandb_run_name} if args.wandb_run_name else {}),
            )
        )

    # Auto-configure a CLIReporter (to log the results to the console).
    # Use better ProgressReporter for multi-agent cases: List individual policy rewards.
    progress_reporter = None
    # if args.num_agents > 0:
    progress_reporter = CLIReporter(
        metric_columns={
            **{
                TRAINING_ITERATION: "iter",
                "time_total_s": "total time (s)",
                NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "return_mean",
            },
        },
    )

    # Force Tuner to use old progress output as the new one silently ignores our custom
    # `CLIReporter`.
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

    # Run the actual experiment (using Tune).
    results = tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=air.RunConfig(
            storage_path="~/ray_results/debug" if (args.debug or args.local) else None,
            stop=stop,
            verbose=1,
            callbacks=tune_callbacks,
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=1,
                num_to_keep=5,
                checkpoint_score_order="max",
                checkpoint_score_attribute=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                checkpoint_at_end=True,
            ),
            progress_reporter=progress_reporter,
            failure_config=train.FailureConfig(fail_fast=True),
            sync_config=train.SyncConfig(sync_artifacts=True),
        ),
        **({} if not args.hyperparam else hyperparams_opt(
            num_iterations=args.num_iterations,
            num_samples=args.num_samples,
            seed=args.seed,
            max_concurrent_trials=args.max_concurrent_trials,
        )),
    ).fit()

    return results