import os
import sys
import time

import ray
import tyro
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

sys.path.append(os.path.dirname(__file__))
from algo import (
    PPORMConfig,
    PPORMLearningConfig,
    PPORMMultiAgentRLModuleSpec,
    PPORMSingleAgentRLModuleSpec,
    PPORMTorchRLModule,
    SharedTracesBuffer,
)
from callbacks import EnvRenderCallback, make_multi_callbacks
from common import Args, make_env, run_rllib_experiment, set_seed
from env import RMEnv
from model import PPORMLearningCatalog, PPORMLearningSharedLayersCatalog
from form import ILASPLearner


def build_model_config(args):
    hidden_dims = [args.hidden_dim] * args.hidden_count

    model_config_dict = {
        "fcnet_hiddens": hidden_dims,
        "fcnet_activation": args.activation,
        "vf_share_layers": args.vf_share_layers,
        "use_lstm": args.use_lstm,
    }

    if args.shared_hidden_layers:
        shared_hidden_layers_ref = PPORMLearningSharedLayersCatalog.build_shared_layers_ref(model_config_dict)
        model_config_dict.update({
            "shared_hidden_layers_indices": args.shared_hidden_layers,
            "shared_hidden_layers_ref": shared_hidden_layers_ref
        })

    if args.use_cnn:
        model_config_dict.update({
            "conv_filters": [
                [16, [2, 2], 1, "valid"],
                [32, [2, 2], 1, "valid"],
                [32, [2, 2], 1, "valid"],
            ],
            "conv_max_pooling": [
                (2, 2),
                tuple(),
                tuple()
            ]
        })

    return model_config_dict


if __name__ == "__main__":
    # Parse the command line arguments.
    args = tyro.cli(Args)

    set_seed(args.seed)

    ray.init(
        num_cpus=None, 
        local_mode=args.local, 
        runtime_env={
            "env_vars": ({"RAY_DEBUG": f"{str(int(args.debug))}"} if args.debug else {}), 
        }
    )

    env_config = {
        "env_id": f"form/{args.env}",
        "seed": args.seed,
        "reward_threshold": args.reward_threshold,
        "rm": args.rm,
        "use_cnn": args.use_cnn,
        "use_labels": args.use_labels
    }

    model_config_dict = build_model_config(args)

    if args.rm:
        env_config.update(
            {
                "reward_shaping": args.reward_shaping,
                "shared_policy": args.shared_policy,
                "shared_hidden_layers": args.shared_hidden_layers,
                "max_rm_num_states": args.max_rm_num_states,
                "extend_obs_space": args.extend_obs_space,
                "rm_path": args.rm_path,
                "restore": args.restore,
                "restore_path": args.restore_path
            }
        )

        traces_buffer_id = SharedTracesBuffer.SHARED_TRACES_BUFFER_REF + "_" + str(time.time())

        if args.rm_learning:
            env_config.update({
                "rm_learning": True,
            })

            config = (
                PPORMLearningConfig()
                .rm(
                    rm_learning_freq=args.rm_learning_freq,
                    traces_buffer=traces_buffer_id,
                    rm_learner_class=ILASPLearner,
                    rm_learner_kws={
                        "agent_id": "A0",
                        "init_rm_num_states": 3,
                        "max_rm_num_states": args.max_rm_num_states,
                        "wait_for_pos_only": True,
                        "learn_first_order": not args.prop,
                        "learn_acyclic": True
                    },
                )
                .rl_module(
                    model_config_dict=model_config_dict
                )
            )
        else:
            config = (
                PPORMConfig()
                .rl_module(
                    model_config_dict=model_config_dict
                )
            )

        config = config.network(
            shared_hidden_layers=args.shared_hidden_layers
        )

        env = RMEnv(env_config)

        if args.rm_learning:
            env_config.update({
                "shared_traces_buffer": traces_buffer_id,
            })

        if not args.shared_policy:
            def policy_mapping_fn_(aid, *args, **kwargs):
                return aid
            def policies_to_train_fn_(module_id, multi_agent_batch):
                return (
                    module_id in multi_agent_batch.policy_batches and
                    ((not args.restore) or (not args.policies_to_train) or (module_id in args.policies_to_train))
                )

            policies = {
                f"rm_state_u{i}"
                for i in range(args.max_rm_num_states)
            }
            config.multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn_,
                policies_to_train=policies_to_train_fn_,
            )
        else:
            def policy_mapping_fn_(aid, *args, **kwargs):
                return "shared_policy"
            def policies_to_train_fn_(module_id, multi_agent_batch):
                return True
            policies = {"shared_policy",}
            config.multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn_,
                policies_to_train=policies_to_train_fn_,
            )

        module_specs = {}
        for pid in policies:
            module_specs[pid] = PPORMSingleAgentRLModuleSpec(
                module_class=PPORMTorchRLModule, 
                observation_space=env.observation_space.get(pid, next(iter(env.observation_space.values()))),
                action_space=env.action_space.get(pid, next(iter(env.action_space.values()))),
                catalog_class=config.get_catalog()
            )

        kwargs = {}
        if args.restore:
            kwargs["load_state_path"] = args.restore_path

        rl_module_spec = PPORMMultiAgentRLModuleSpec(
            module_specs=module_specs, 
            **kwargs
        )

        config = (
            config
            .environment(
                env="form/RMEnv",
            )
        )

    else:
        config = (
            PPOConfig()
            .environment(
                env=f"form/{args.env}",
            )
            .rl_module(
                model_config_dict=model_config_dict
            )
            .env_runners(
                sample_timeout_s=600,
            )
        )

        env = make_env(f"form/{args.env}")(env_config)

        rl_module_spec = PPORMSingleAgentRLModuleSpec(
            module_class=PPOTorchRLModule,
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_config_dict=model_config_dict,
            catalog_class=PPORMLearningCatalog
        )

    if args.num_workers:
        if not args.rm or args.shared_policy:
            config.training(
                sgd_minibatch_size=2 ** 8,
                train_batch_size=args.num_workers * (2 ** 12),
            )
        else:
            config.training(
                sgd_minibatch_size=2 ** 12, # 2**8 2**5
                train_batch_size=args.num_workers * (2 ** 14), # 2**11 2**8
            )

    callbacks = []
    if args.render:
        callbacks.append(EnvRenderCallback)
    callback_cls = make_multi_callbacks(callbacks)

    config = (
        config
        .environment(
            env_config=env_config, 
            is_atari=False
        )
        .framework("torch")
        .rl_module(
            rl_module_spec=rl_module_spec,
        )
        .training(
            gamma=0.999,
            vf_loss_coeff=0.5,
            lambda_=0.95,
            clip_param=0.15,
            vf_clip_param=0.1,
            entropy_coeff=0.01,
            kl_target=0.15,
            grad_clip=0.7,
            num_sgd_iter=20,
            lr=7e-4,
            grad_clip_by="value",
            use_critic=True,
            use_gae=True,
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .env_runners(
            batch_mode="complete_episodes" if args.rm else "truncate_episodes",
            use_worker_filter_stats=False,
            observation_filter="MeanStdFilter",
            num_env_runners=args.num_workers,
            sample_timeout_s=600.0,
        )
        .evaluation(
            evaluation_num_workers=1,
            evaluation_interval=10 if args.eval else 0,
            evaluation_duration=5,
            evaluation_duration_unit="episodes",
            evaluation_sample_timeout_s=600,
            evaluation_parallel_to_training=False,
            evaluation_config=config.overrides(
                entropy_coeff=0.0,
                env_config=(
                    env_config | {
                        "rm_learning": False, 
                        "reward_shaping": False,
                        "rm_path": args.rm_path,
                        "shared_hidden_layers": tuple(),
                    }
                ),
                _model_config_dict=(
                    model_config_dict | {
                        "shared_hidden_layers_indices": tuple(),
                        "shared_hidden_layers_ref": None
                    }
                )
            ),
        )
        .debugging(seed=args.seed)
    )

    results = run_rllib_experiment(config, args)

    if env_config.get("shared_traces_buffer"):
        traces_buffer_actor = ray.get_actor(env_config["shared_traces_buffer"])
        ray.kill(traces_buffer_actor)
    
    ray.shutdown()
