import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import ray
import tree
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT, Encoder, Model
from ray.rllib.core.models.configs import (
    CNNEncoderConfig,
    MLPEncoderConfig,
    MLPHeadConfig,
    ModelConfig,
    RecurrentEncoderConfig,
    _framework_implemented,
)
from ray.rllib.core.models.specs.specs_base import Spec, TensorSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.encoder import (
    TorchActorCriticEncoder,
    TorchCNNEncoder,
    TorchLSTMEncoder,
    TorchMLPEncoder,
    TorchStatefulActorCriticEncoder,
)
from ray.rllib.core.models.torch.primitives import TorchMLP
from ray.rllib.models import MODEL_DEFAULTS as _MODEL_DEFAULTS
from ray.rllib.models.torch.misc import same_padding, valid_padding
from ray.rllib.models.utils import (
    get_activation_fn,
    get_filter_config,
    get_initializer_fn,
)
from ray.rllib.policy.rnn_sequencing import get_fold_unfold_fns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

SHARED_HIDDEN_LAYERS_REF = "actor::shared_layers"

MODEL_DEFAULTS = (_MODEL_DEFAULTS | {"shared_hidden_layers_indices": tuple(), "shared_hidden_layers_ref": SHARED_HIDDEN_LAYERS_REF})

class PPORMLearningCatalog(PPOCatalog):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, model_config_dict: Dict):
        
        self.observation_space = observation_space
        self.action_space = action_space

        # TODO (Artur): Make model defaults a dataclass
        self._model_config_dict = {**MODEL_DEFAULTS, **model_config_dict}
        self._view_requirements = None
        self._latent_dims = None

        self._determine_components_hook()

        self.actor_critic_encoder_config = RMLearningActorCriticEncoderConfig(
            base_encoder_config=self._encoder_config,
            shared=self._model_config_dict["vf_share_layers"],
        )

        self.pi_and_vf_head_hiddens = self._model_config_dict["post_fcnet_hiddens"]
        self.pi_and_vf_head_activation = self._model_config_dict[
            "post_fcnet_activation"
        ]

        # We don't have the exact (framework specific) action dist class yet and thus
        # cannot determine the exact number of output nodes (action space) required.
        # -> Build pi config only in the `self.build_pi_head` method.
        self.pi_head_config = None

        self.vf_head_config = MLPHeadConfig(
            input_dims=self.latent_dims,
            hidden_layer_dims=self.pi_and_vf_head_hiddens,
            hidden_layer_activation=self.pi_and_vf_head_activation,
            output_layer_activation="linear",
            output_layer_dim=1,
        )

    @classmethod
    def _get_mlp_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
        view_requirements=None,
    ) -> ModelConfig:
        fcnet_hiddens = model_config_dict["fcnet_hiddens"]
        # TODO (sven): Move to a new ModelConfig object (dataclass) asap, instead of
        #  "linking" into the old ModelConfig (dict)! This just causes confusion as to
        #  which old keys now mean what for the new RLModules-based default models.
        encoder_latent_dim = (
            model_config_dict["encoder_latent_dim"] or fcnet_hiddens[-1]
        )

        # In order to guarantee backward compatability with old configs,
        # we need to check if no latent dim was set and simply reuse the last
        # fcnet hidden dim for that purpose.
        if model_config_dict["encoder_latent_dim"]:
            hidden_layer_dims = model_config_dict["fcnet_hiddens"]
        else:
            hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
        
        encoder_config = RMLearningMLPEncoderConfig(
            input_dims=observation_space.shape,
            hidden_layer_dims=hidden_layer_dims,
            hidden_layer_activation=model_config_dict["fcnet_activation"],
            hidden_layer_weights_initializer=model_config_dict[
                "fcnet_weights_initializer"
            ],
            hidden_layer_weights_initializer_config=model_config_dict[
                "fcnet_weights_initializer_config"
            ],
            hidden_layer_bias_initializer=model_config_dict[
                "fcnet_bias_initializer"
            ],
            hidden_layer_bias_initializer_config=model_config_dict[
                "fcnet_bias_initializer_config"
            ],
            output_layer_dim=encoder_latent_dim,
            output_layer_activation=model_config_dict["fcnet_activation"],
            output_layer_weights_initializer=model_config_dict[
                "post_fcnet_weights_initializer"
            ],
            output_layer_weights_initializer_config=model_config_dict[
                "post_fcnet_weights_initializer_config"
            ],
            output_layer_bias_initializer=model_config_dict[
                "post_fcnet_bias_initializer"
            ],
            output_layer_bias_initializer_config=model_config_dict[
                "post_fcnet_bias_initializer_config"
            ],
        )
        return encoder_config

    @classmethod
    def _get_cnn_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
        view_requirements=None,
    ) -> ModelConfig:
        if not model_config_dict.get("conv_filters"):
            model_config_dict["conv_filters"] = get_filter_config(
                observation_space.shape
            )

        encoder_config = RMLearningCNNEncoderConfig(
            input_dims=observation_space.shape,
            cnn_filter_specifiers=model_config_dict["conv_filters"],
            cnn_max_pooling_specifiers=model_config_dict["conv_max_pooling"],
            cnn_activation=model_config_dict["conv_activation"],
            cnn_use_layernorm=model_config_dict.get(
                "conv_use_layernorm", False
            ),
            cnn_kernel_initializer=model_config_dict["conv_kernel_initializer"],
            cnn_kernel_initializer_config=model_config_dict[
                "conv_kernel_initializer_config"
            ],
            cnn_bias_initializer=model_config_dict["conv_bias_initializer"],
            cnn_bias_initializer_config=model_config_dict[
                "conv_bias_initializer_config"
            ],
            output_dim=model_config_dict["encoder_latent_dim"]
        )
        return encoder_config

    @classmethod
    def _get_lstm_encoder_config(
        cls,
        input_dims: Union[List[int], Tuple[int]],
        model_config_dict: dict,
        action_space: gym.Space = None,
        view_requirements=None,
        tokenizer_config=None
    ):
        return RMLearningMixedRecurrentEncoderConfig(
                input_dims=input_dims,
                recurrent_layer_type="lstm",
                hidden_dim=model_config_dict["lstm_cell_size"],
                hidden_weights_initializer=model_config_dict[
                    "lstm_weights_initializer"
                ],
                hidden_weights_initializer_config=model_config_dict[
                    "lstm_weights_initializer_config"
                ],
                hidden_bias_initializer=model_config_dict["lstm_bias_initializer"],
                hidden_bias_initializer_config=model_config_dict[
                    "lstm_bias_initializer_config"
                ],
                batch_major=not model_config_dict["_time_major"],
                num_layers=1,
                tokenizer_config=tokenizer_config
            )

    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
        view_requirements=None,
    ) -> ModelConfig:
        """Returns an EncoderConfig for the given input_space and model_config_dict.

        Encoders are usually used in RLModules to transform the input space into a
        latent space that is then fed to the heads. The returned EncoderConfig
        objects correspond to the built-in Encoder classes in RLlib.
        For example, for a simple 1D-Box input_space, RLlib offers an
        MLPEncoder, hence this method returns the MLPEncoderConfig. You can overwrite
        this method to produce specific EncoderConfigs for your custom Models.

        The following input spaces lead to the following configs:
        - 1D-Box: MLPEncoderConfig
        - 3D-Box: CNNEncoderConfig
        # TODO (Artur): Support more spaces here
        # ...

        Args:
            observation_space: The observation space to use.
            model_config_dict: The model config to use.
            action_space: The action space to use if actions are to be encoded. This
                is commonly the case for LSTM models.
            view_requirements: The view requirements to use if anything else than
                observation_space or action_space is to be encoded. This signifies an
                advanced use case.

        Returns:
            The encoder config.
        """
        # TODO (Artur): Make it so that we don't work with complete MODEL_DEFAULTS
        model_config_dict = {**MODEL_DEFAULTS, **model_config_dict}

        use_lstm = model_config_dict["use_lstm"]
        use_attention = model_config_dict["use_attention"]

        if use_attention:
            raise NotImplementedError
        else:
            cnn_obs_space = None
            if isinstance(observation_space, spaces.Dict) and "image" in list(observation_space.keys()):
                cnn_obs_space = observation_space["image"]
            elif isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
                cnn_obs_space = observation_space
            
            cnn_encoder_config = cls._get_cnn_encoder_config(
                cnn_obs_space, model_config_dict, action_space, view_requirements
            ) if cnn_obs_space else None

            labels_obs_space = None
            if isinstance(observation_space, spaces.Dict) and "labels" in list(observation_space.keys()):
                labels_obs_space = observation_space["labels"]
            # elif isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 1:
            #     labels_obs_space = observation_space

            labels_encoder_config = cls._get_mlp_encoder_config(
                labels_obs_space, model_config_dict, action_space, view_requirements
            ) if labels_obs_space else None

            rm_state_obs_space = None
            if isinstance(observation_space, spaces.Dict) and "rm_state" in list(observation_space.keys()):
                rm_state_obs_space = observation_space["rm_state"]
            # elif isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 1:
            #     labels_obs_space = observation_space

            rm_state_encoder_config = cls._get_mlp_encoder_config(
                rm_state_obs_space, model_config_dict, action_space, view_requirements
            ) if rm_state_obs_space else None
            
            mixed_encoder = RMLearningMixedEncoderConfig(
                cnn_encoder_config=cnn_encoder_config,
                labels_encoder_config=labels_encoder_config,
                rm_state_encoder_config=rm_state_encoder_config
            )

            lstm_encoder_config = None
            if use_lstm:
                lstm_encoder_config = cls._get_lstm_encoder_config(
                    mixed_encoder.output_dims, model_config_dict, action_space, view_requirements, 
                    mixed_encoder
                )

            if cnn_encoder_config is None:
                raise RuntimeError
            elif use_lstm:
                encoder_config = lstm_encoder_config
            else:
                encoder_config = mixed_encoder

        return encoder_config

class PPORMLearningSharedLayersCatalog(PPORMLearningCatalog):
    
    @classmethod
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
        view_requirements=None,
    ) -> ModelConfig:
        """Returns an EncoderConfig for the given input_space and model_config_dict.

        Encoders are usually used in RLModules to transform the input space into a
        latent space that is then fed to the heads. The returned EncoderConfig
        objects correspond to the built-in Encoder classes in RLlib.
        For example, for a simple 1D-Box input_space, RLlib offers an
        MLPEncoder, hence this method returns the MLPEncoderConfig. You can overwrite
        this method to produce specific EncoderConfigs for your custom Models.

        The following input spaces lead to the following configs:
        - 1D-Box: MLPEncoderConfig
        - 3D-Box: CNNEncoderConfig
        # TODO (Artur): Support more spaces here
        # ...

        Args:
            observation_space: The observation space to use.
            model_config_dict: The model config to use.
            action_space: The action space to use if actions are to be encoded. This
                is commonly the case for LSTM models.
            view_requirements: The view requirements to use if anything else than
                observation_space or action_space is to be encoded. This signifies an
                advanced use case.

        Returns:
            The encoder config.
        """
        # TODO (Artur): Make it so that we don't work with complete MODEL_DEFAULTS
        model_config_dict = {**MODEL_DEFAULTS, **model_config_dict}

        fcnet_hiddens = model_config_dict["fcnet_hiddens"]
        # TODO (sven): Move to a new ModelConfig object (dataclass) asap, instead of
        #  "linking" into the old ModelConfig (dict)! This just causes confusion as to
        #  which old keys now mean what for the new RLModules-based default models.
        encoder_latent_dim = (
            model_config_dict["encoder_latent_dim"] or fcnet_hiddens[-1]
        )
        use_lstm = model_config_dict["use_lstm"]
        use_attention = model_config_dict["use_attention"]

        if use_lstm:
            encoder_config = RecurrentEncoderConfig(
                input_dims=observation_space.shape,
                recurrent_layer_type="lstm",
                hidden_dim=model_config_dict["lstm_cell_size"],
                hidden_weights_initializer=model_config_dict[
                    "lstm_weights_initializer"
                ],
                hidden_weights_initializer_config=model_config_dict[
                    "lstm_weights_initializer_config"
                ],
                hidden_bias_initializer=model_config_dict["lstm_bias_initializer"],
                hidden_bias_initializer_config=model_config_dict[
                    "lstm_bias_initializer_config"
                ],
                batch_major=not model_config_dict["_time_major"],
                num_layers=1,
                tokenizer_config=cls.get_tokenizer_config(
                    observation_space,
                    model_config_dict,
                    view_requirements,
                ),
            )
        elif use_attention:
            raise NotImplementedError
        else:
            # TODO (Artur): Maybe check for original spaces here
            # input_space is a 1D Box
            if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 1:
                # In order to guarantee backward compatability with old configs,
                # we need to check if no latent dim was set and simply reuse the last
                # fcnet hidden dim for that purpose.
                if model_config_dict["encoder_latent_dim"]:
                    hidden_layer_dims = model_config_dict["fcnet_hiddens"]
                else:
                    hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
                encoder_config = RMLearningSharedMLPEncoderConfig(
                    input_dims=observation_space.shape,
                    hidden_layer_dims=hidden_layer_dims,
                    hidden_layer_activation=model_config_dict["fcnet_activation"],
                    hidden_layer_weights_initializer=model_config_dict[
                        "fcnet_weights_initializer"
                    ],
                    hidden_layer_weights_initializer_config=model_config_dict[
                        "fcnet_weights_initializer_config"
                    ],
                    hidden_layer_bias_initializer=model_config_dict[
                        "fcnet_bias_initializer"
                    ],
                    hidden_layer_bias_initializer_config=model_config_dict[
                        "fcnet_bias_initializer_config"
                    ],
                    output_layer_dim=encoder_latent_dim,
                    output_layer_activation=model_config_dict["fcnet_activation"],
                    output_layer_weights_initializer=model_config_dict[
                        "post_fcnet_weights_initializer"
                    ],
                    output_layer_weights_initializer_config=model_config_dict[
                        "post_fcnet_weights_initializer_config"
                    ],
                    output_layer_bias_initializer=model_config_dict[
                        "post_fcnet_bias_initializer"
                    ],
                    output_layer_bias_initializer_config=model_config_dict[
                        "post_fcnet_bias_initializer_config"
                    ],
                    shared_hidden_layers_indices=model_config_dict[
                        "shared_hidden_layers_indices"
                    ],
                    shared_hidden_layers_ref=model_config_dict[
                        "shared_hidden_layers_ref"
                    ],
                )

            # input_space is a 3D Box
            elif (
                isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3
            ):
                if not model_config_dict.get("conv_filters"):
                    model_config_dict["conv_filters"] = get_filter_config(
                        observation_space.shape
                    )

                encoder_config = CNNEncoderConfig(
                    input_dims=observation_space.shape,
                    cnn_filter_specifiers=model_config_dict["conv_filters"],
                    cnn_activation=model_config_dict["conv_activation"],
                    cnn_use_layernorm=model_config_dict.get(
                        "conv_use_layernorm", False
                    ),
                    cnn_kernel_initializer=model_config_dict["conv_kernel_initializer"],
                    cnn_kernel_initializer_config=model_config_dict[
                        "conv_kernel_initializer_config"
                    ],
                    cnn_bias_initializer=model_config_dict["conv_bias_initializer"],
                    cnn_bias_initializer_config=model_config_dict[
                        "conv_bias_initializer_config"
                    ],
                )
            # input_space is a 2D Box
            elif (
                isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 2
            ):
                # RLlib used to support 2D Box spaces by silently flattening them
                raise ValueError(
                    f"No default encoder config for obs space={observation_space},"
                    f" lstm={use_lstm} and attention={use_attention} found. 2D Box "
                    f"spaces are not supported. They should be either flattened to a "
                    f"1D Box space or enhanced to be a 3D box space."
                )
            # input_space is a possibly nested structure of spaces.
            else:
                # NestedModelConfig
                raise ValueError(
                    f"No default encoder config for obs space={observation_space},"
                    f" lstm={use_lstm} and attention={use_attention} found."
                )

        return encoder_config

    @staticmethod
    def build_shared_layers_ref(model_config_dict) -> ray.ObjectRef:
        model_config_dict = MODEL_DEFAULTS | model_config_dict

        name = model_config_dict["shared_hidden_layers_ref"]

        _actor = SharedLayers.options(name=name, lifetime="detached").remote(
            hidden_layer_dims=model_config_dict["fcnet_hiddens"],
            hidden_layer_activation=model_config_dict["fcnet_activation"],
            hidden_layer_weights_initializer=model_config_dict[
                "fcnet_weights_initializer"
            ],
            hidden_layer_weights_initializer_config=model_config_dict[
                "fcnet_weights_initializer_config"
            ],
            hidden_layer_bias_initializer=model_config_dict[
                "fcnet_bias_initializer"
            ],
            hidden_layer_bias_initializer_config=model_config_dict[
                "fcnet_bias_initializer_config"
            ]
        )

        return model_config_dict["shared_hidden_layers_ref"]

@dataclass
class RMLearningActorCriticEncoderConfig(ModelConfig):
    """Configuration for an ActorCriticEncoder.

    The base encoder functions like other encoders in RLlib. It is wrapped by the
    ActorCriticEncoder to provides a shared encoder Model to use in RLModules that
    provides twofold outputs: one for the actor and one for the critic. See
    ModelConfig for usage details.

    Attributes:
        base_encoder_config: The configuration for the wrapped encoder(s).
        shared: Whether the base encoder is shared between the actor and critic.
    """

    base_encoder_config: ModelConfig = None
    shared: bool = True

    @_framework_implemented(tf2=False)
    def build(self, framework: str = "torch") -> "Encoder":
        if framework == "torch":
            if isinstance(self.base_encoder_config, RMLearningMixedRecurrentEncoderConfig):
                return TorchStatefulActorCriticEncoder(self)
            else:
                return TorchActorCriticEncoder(self)
        else:
            raise NotImplementedError(framework)

@dataclass
class RMLearningMLPEncoderConfig(MLPEncoderConfig):
    """Configuration for an MLP that acts as an encoder.
    """

    @_framework_implemented(torch=True, tf2=False)
    def build(self, framework: str = "torch") -> "Encoder":
        self._validate(framework)

        if framework == "torch":
            return TorchRMLearningMLPEncoder(self)
        else:
            raise NotImplementedError(framework)

@dataclass
class RMLearningSharedMLPEncoderConfig(MLPEncoderConfig):
    """Configuration for an MLP that acts as an encoder.
    """
    shared_hidden_layers_indices: tuple = tuple()
    shared_hidden_layers_ref: object = None

    @_framework_implemented(torch=True, tf2=False)
    def build(self, framework: str = "torch") -> "Encoder":
        self._validate(framework)

        if framework == "torch":
            return TorchRMLearningSharedMLPEncoder(self)
        else:
            raise NotImplementedError(framework)

    def _validate(self, framework: str):
        super()._validate(framework)

        if not all(
            (0 < i and i < len(self.hidden_layer_dims)+1) # we add the output layer (+1)
            for i in self.shared_hidden_layers_indices
        ):
            raise ValueError(
                "Trying to share a layer that does not exist"
            )

@dataclass
class RMLearningCNNEncoderConfig(CNNEncoderConfig):
    cnn_max_pooling_specifiers: List[Tuple[int]] = None
    output_dim: Optional[int] = None

    def __post_init__(self):
        if self.cnn_max_pooling_specifiers is None:
            self.cnn_max_pooling_specifiers = [tuple()] * len(self.cnn_filter_specifiers)

    def compute_cnn_output_dims(self):
        # Infer output dims, layer by layer.
        dims = self.input_dims  # Creates a copy (works for tuple/list).
        for filter_spec, max_pooling_spec in zip(self.cnn_filter_specifiers, self.cnn_max_pooling_specifiers):
            # Padding not provided, "same" by default.
            if len(filter_spec) == 3:
                num_filters, kernel, stride = filter_spec
                padding = "same"
            # Padding option provided, use given value.
            else:
                num_filters, kernel, stride, padding = filter_spec

            # Same padding.
            if padding == "same":
                _, dims = same_padding(dims[:2], kernel, stride)
            # Valid padding.
            else:
                dims = valid_padding(dims[:2], kernel, stride)

            # Add depth (num_filters) to the end (our utility functions for same/valid
            # only return the image width/height).
            dims = [dims[0], dims[1], num_filters]

            if max_pooling_spec:
                dims = valid_padding(dims[:2], max_pooling_spec[0], max_pooling_spec[1])

        return tuple(dims)

    @property
    def output_dims(self):
        if not self.input_dims:
            return None

        # Infer output dims, layer by layer.
        dims = self.compute_cnn_output_dims()

        # Flatten everything.
        if self.flatten_at_end:
            return (int(np.prod(dims)),)
        
        if self.output_dim:
            return (self.output_dim,)

        return tuple(dims)

    @_framework_implemented(torch=True, tf2=False)
    def build(self, framework: str = "torch") -> "Model":
        self._validate(framework)

        if framework == "torch":
            return TorchRMLearningCNNEncoder(self)

        elif framework == "tf2":
            raise NotImplementedError(framework)

@dataclass
class RMLearningMixedEncoderConfig(ModelConfig):
    cnn_encoder_config: Optional[CNNEncoderConfig] = None
    labels_encoder_config: Optional[MLPEncoderConfig] = None
    rm_state_encoder_config: Optional[MLPEncoderConfig] = None
        
    def _validate(self, framework: str = "torch"):
        """Makes sure that settings are valid."""
        if not len(self.cnn_encoder_config.output_dims) == 1:
            raise ValueError("CNN Encoder must have an output of shape (1,) to be concatenated with labels encoding")

    @property
    def output_dims(self):
        return (
            self.cnn_encoder_config.output_dims[0] + 
            (0 if self.rm_state_encoder_config is None else self.rm_state_encoder_config.output_dims[0]) + 
            (0 if self.labels_encoder_config is None else self.labels_encoder_config.output_dims[0]),
        )

    @_framework_implemented(torch=True, tf2=False)
    def build(self, framework: str = "torch") -> "Model":
        self._validate(framework)

        if framework == "torch":
            return TorchRMLearningMixedEncoder(self)

        elif framework == "tf2":
            raise NotImplementedError(framework)


@dataclass
class RMLearningMixedRecurrentEncoderConfig(RecurrentEncoderConfig):

    @property
    def output_dims(self):
        return (self.hidden_dim,)

    @_framework_implemented(torch=True, tf2=False)
    def build(self, framework: str = "torch") -> "Model":
        self._validate(framework)

        if framework == "torch":
            if self.recurrent_layer_type == "lstm":
                return TorchRMLearningLSTM(self)
            else:
                raise NotImplementedError(self.recurrent_layer_type)

        elif framework == "tf2":
            raise NotImplementedError(framework)

class TorchRMLearningMLPEncoder(TorchMLPEncoder, TorchModel):
    def __init__(self, config: RMLearningMLPEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        # Create the neural network.
        self.net = TorchMLP(
            input_dim=config.input_dims[0],
            hidden_layer_dims=config.hidden_layer_dims,
            hidden_layer_activation=config.hidden_layer_activation,
            hidden_layer_use_layernorm=config.hidden_layer_use_layernorm,
            hidden_layer_use_bias=config.hidden_layer_use_bias,
            hidden_layer_weights_initializer=config.hidden_layer_weights_initializer,
            hidden_layer_weights_initializer_config=(
                config.hidden_layer_weights_initializer_config
            ),
            hidden_layer_bias_initializer=config.hidden_layer_bias_initializer,
            hidden_layer_bias_initializer_config=(
                config.hidden_layer_bias_initializer_config
            ),
            output_dim=config.output_layer_dim,
            output_activation=config.output_layer_activation,
            output_use_bias=config.output_layer_use_bias,
            output_weights_initializer=config.output_layer_weights_initializer,
            output_weights_initializer_config=(
                config.output_layer_weights_initializer_config
            ),
            output_bias_initializer=config.output_layer_bias_initializer,
            output_bias_initializer_config=config.output_layer_bias_initializer_config,
        )

class TorchRMLearningSharedMLPEncoder(TorchMLPEncoder, TorchModel):
    def __init__(self, config: RMLearningSharedMLPEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        # Create the neural network.
        self.net = TorchRMLearningSharedMLP(
            input_dim=config.input_dims[0],
            hidden_layer_dims=config.hidden_layer_dims,
            shared_layers_indices=config.shared_hidden_layers_indices,
            shared_layers_ref=config.shared_hidden_layers_ref,
            hidden_layer_activation=config.hidden_layer_activation,
            hidden_layer_use_layernorm=config.hidden_layer_use_layernorm,
            hidden_layer_use_bias=config.hidden_layer_use_bias,
            hidden_layer_weights_initializer=config.hidden_layer_weights_initializer,
            hidden_layer_weights_initializer_config=(
                config.hidden_layer_weights_initializer_config
            ),
            hidden_layer_bias_initializer=config.hidden_layer_bias_initializer,
            hidden_layer_bias_initializer_config=(
                config.hidden_layer_bias_initializer_config
            ),
            output_dim=config.output_layer_dim,
            output_activation=config.output_layer_activation,
            output_use_bias=config.output_layer_use_bias,
            output_weights_initializer=config.output_layer_weights_initializer,
            output_weights_initializer_config=(
                config.output_layer_weights_initializer_config
            ),
            output_bias_initializer=config.output_layer_bias_initializer,
            output_bias_initializer_config=config.output_layer_bias_initializer_config,
        )

class TorchRMLearningCNNEncoder(TorchCNNEncoder, TorchModel):

    def __init__(self, config: RMLearningCNNEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        layers = []
        # The bare-bones CNN (no flatten, no succeeding dense).
        cnn = TorchMaxPoolingCNN(
            input_dims=config.input_dims,
            cnn_filter_specifiers=config.cnn_filter_specifiers,
            cnn_max_pooling_specifiers=config.cnn_max_pooling_specifiers,
            cnn_activation=config.cnn_activation,
            cnn_use_layernorm=config.cnn_use_layernorm,
            cnn_use_bias=config.cnn_use_bias,
            cnn_kernel_initializer=config.cnn_kernel_initializer,
            cnn_kernel_initializer_config=config.cnn_kernel_initializer_config,
            cnn_bias_initializer=config.cnn_bias_initializer,
            cnn_bias_initializer_config=config.cnn_bias_initializer_config,
        )
        layers.append(cnn)

        # Add a flatten operation to move from 2/3D into 1D space.
        if config.flatten_at_end:
            layers.append(nn.Flatten())

            if config.output_dim:
                layers.append(
                    nn.Linear(np.prod(config.compute_cnn_output_dims()), config.output_dim)
                )

        self.net = nn.Sequential(*layers)

class TorchRMLearningMixedEncoder(Encoder, TorchModel):
    def __init__(self, config: RMLearningMixedEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.rm_state_encoder = None
        self.labels_encoder = None

        self.cnn_encoder = TorchRMLearningCNNEncoder(config.cnn_encoder_config)

        if config.rm_state_encoder_config:
            self.rm_state_encoder = TorchRMLearningMLPEncoder(config.rm_state_encoder_config)

        if config.labels_encoder_config:
            self.labels_encoder = TorchRMLearningMLPEncoder(config.labels_encoder_config)

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        specs = {}
        if self.config.rm_state_encoder_config:
            specs["rm_state"] = TensorSpec(
                "b, d", d=self.config.rm_state_encoder_config.input_dims[0], framework="torch"
            )

        if self.config.labels_encoder_config:
            specs["labels"] = TensorSpec(
                "b, d", d=self.config.labels_encoder_config.input_dims[0], framework="torch"
            )

        image_specs = TensorSpec(
            "b, w, h, c",
            w=self.config.cnn_encoder_config.input_dims[0],
            h=self.config.cnn_encoder_config.input_dims[1],
            c=self.config.cnn_encoder_config.input_dims[2],
            framework="torch",
        )

        if specs:
            specs["image"] = image_specs
        else:
            specs = image_specs

        return SpecDict({
            Columns.OBS: specs,
        })

    @override(Model)
    def get_output_specs(self) -> Optional[Spec]:
        return SpecDict(
            {
                ENCODER_OUT: (
                    TensorSpec("b, d", d=self.config.output_dims[0], framework="torch")
                )
            }
        )

    @override(Model)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        image = inputs[Columns.OBS]
        if isinstance(image, (dict, ray.rllib.utils.nested_dict.NestedDict)):
            image = image["image"]
        cnn_output = self.cnn_encoder._forward({Columns.OBS: image})[ENCODER_OUT]

        outputs = [cnn_output]

        if self.config.rm_state_encoder_config:
            rm_state_output = self.rm_state_encoder._forward({Columns.OBS: inputs[Columns.OBS]["rm_state"]})[ENCODER_OUT]
            outputs.append(rm_state_output)

        if self.config.labels_encoder_config:
            labels_output = self.labels_encoder._forward({Columns.OBS: inputs[Columns.OBS]["labels"]})[ENCODER_OUT]
            outputs.append(labels_output)

        encoder_output = torch.concat(outputs, dim=-1)

        return {ENCODER_OUT: encoder_output}

class TorchMaxPoolingCNN(nn.Module):

    def __init__(
        self,
        *,
        input_dims: Union[List[int], Tuple[int]],
        cnn_filter_specifiers: List[List[Union[int, List]]],
        cnn_max_pooling_specifiers: List[Tuple[int]] = None,
        cnn_use_bias: bool = True,
        cnn_use_layernorm: bool = False,
        cnn_activation: str = "relu",
        cnn_kernel_initializer: Optional[Union[str, Callable]] = None,
        cnn_kernel_initializer_config: Optional[Dict] = None,
        cnn_bias_initializer: Optional[Union[str, Callable]] = None,
        cnn_bias_initializer_config: Optional[Dict] = None,
    ):
        """Initializes a TorchCNN instance.

        Args:
            input_dims: The 3D input dimensions of the network (incoming image).
            cnn_filter_specifiers: A list in which each element is another (inner) list
                of either the following forms:
                `[number of channels/filters, kernel, stride]`
                OR:
                `[number of channels/filters, kernel, stride, padding]`, where `padding`
                can either be "same" or "valid".
                When using the first format w/o the `padding` specifier, `padding` is
                "same" by default. Also, `kernel` and `stride` may be provided either as
                single ints (square) or as a tuple/list of two ints (width- and height
                dimensions) for non-squared kernel/stride shapes.
                A good rule of thumb for constructing CNN stacks is:
                When using padding="same", the input "image" will be reduced in size by
                the factor `stride`, e.g. input=(84, 84, 3) stride=2 kernel=x
                padding="same" filters=16 -> output=(42, 42, 16).
                For example, if you would like to reduce an Atari image from its
                original (84, 84, 3) dimensions down to (6, 6, F), you can construct the
                following stack and reduce the w x h dimension of the image by 2 in each
                layer:
                [[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]] -> output=(6, 6, 128)
            cnn_max_pooling_specifiers: [[2, 2], [], []]
            cnn_use_bias: Whether to use bias on all Conv2D layers.
            cnn_activation: The activation function to use after each Conv2D layer.
            cnn_use_layernorm: Whether to insert a LayerNormalization functionality
                in between each Conv2D layer's outputs and its activation.
            cnn_kernel_initializer: The initializer function or class to use for kernel
                initialization in the CNN layers. If `None` the default initializer of
                the respective CNN layer is used. Note, only the in-place
                initializers, i.e. ending with an underscore "_" are allowed.
            cnn_kernel_initializer_config: Configuration to pass into the initializer
                defined in `cnn_kernel_initializer`.
            cnn_bias_initializer: The initializer function or class to use for bias
                initializationcin the CNN layers. If `None` the default initializer of
                the respective CNN layer is used. Note, only the in-place initializers,
                i.e. ending with an underscore "_" are allowed.
            cnn_bias_initializer_config: Configuration to pass into the initializer
                defined in `cnn_bias_initializer`.
        """
        super().__init__()

        assert len(input_dims) == 3

        cnn_activation = get_activation_fn(cnn_activation, framework="torch")
        cnn_kernel_initializer = get_initializer_fn(
            cnn_kernel_initializer, framework="torch"
        )
        cnn_bias_initializer = get_initializer_fn(
            cnn_bias_initializer, framework="torch"
        )
        layers = []

        # Add user-specified hidden convolutional layers first
        width, height, in_depth = input_dims
        in_size = [width, height]
        for i, filter_specs in enumerate(cnn_filter_specifiers):
            # Padding information not provided -> Use "same" as default.
            if len(filter_specs) == 3:
                out_depth, kernel_size, strides = filter_specs
                padding = "same"
            # Padding information provided.
            else:
                out_depth, kernel_size, strides, padding = filter_specs

            # Pad like in tensorflow's SAME/VALID mode.
            if padding == "same":
                padding_size, out_size = same_padding(in_size, kernel_size, strides)
                layers.append(nn.ZeroPad2d(padding_size))
            # No actual padding is performed for "valid" mode, but we will still
            # compute the output size (input for the next layer).
            else:
                out_size = valid_padding(in_size, kernel_size, strides)

            layer = nn.Conv2d(
                in_depth, out_depth, kernel_size, strides, bias=cnn_use_bias
            )

            # Initialize CNN layer kernel if necessary.
            if cnn_kernel_initializer:
                cnn_kernel_initializer(
                    layer.weight, **cnn_kernel_initializer_config or {}
                )
            # Initialize CNN layer bias if necessary.
            if cnn_bias_initializer:
                cnn_bias_initializer(layer.bias, **cnn_bias_initializer_config or {})

            layers.append(layer)

            # Layernorm.
            if cnn_use_layernorm:
                # We use an epsilon of 0.001 here to mimick the Tf default behavior.
                layers.append(nn.LayerNorm1D(out_depth, eps=0.001))
            # Activation.
            if cnn_activation is not None:
                layers.append(cnn_activation())

            if cnn_max_pooling_specifiers[i]:
                layers.append(nn.MaxPool2d(cnn_max_pooling_specifiers[i]))

            in_size = out_size
            in_depth = out_depth

        # Create the CNN.
        self.cnn = nn.Sequential(*layers)

        self.expected_input_dtype = torch.float32

    def forward(self, inputs):
        # Permute b/c data comes in as channels_last ([B, dim, dim, channels]) ->
        # Convert to `channels_first` for torch:
        try:
            inputs = inputs.permute(0, 3, 1, 2)
        except:
            raise ValueError(inputs)
        out = self.cnn(inputs.type(self.expected_input_dtype))
        # Permute back to `channels_last`.
        return out.permute(0, 2, 3, 1)

# @DeveloperAPI
def _tokenize(tokenizer: Encoder, inputs: dict, framework: str) -> dict:
    """Tokenizes the observations from the input dict.

    Args:
        tokenizer: The tokenizer to use.
        inputs: The input dict.

    Returns:
        The output dict.
    """
    # Tokenizer may depend solely on observations.
    obs = inputs[Columns.OBS]
    tokenizer_inputs = {Columns.OBS: obs}
    
    if isinstance(obs, dict):
        size = list(list(obs.values())[0].size())
    elif isinstance(obs, np.ndarray):
        size = list(obs.size() if framework == "torch" else obs.shape)
    else:
        raise ValueError(type(obs))

    b_dim, t_dim = size[:2]
    fold, unfold = get_fold_unfold_fns(b_dim, t_dim, framework=framework)
    # Push through the tokenizer encoder.
    out = tokenizer(fold(tokenizer_inputs))
    out = out[ENCODER_OUT]
    # Then unfold batch- and time-dimensions again.
    return unfold(out)

class TorchRMLearningLSTM(TorchLSTMEncoder):
    def __init__(self, config: RecurrentEncoderConfig) -> None:
        TorchModel.__init__(self, config)

        # Maybe create a tokenizer
        if config.tokenizer_config is not None:
            self.tokenizer = config.tokenizer_config.build(framework="torch")
            lstm_input_dims = config.tokenizer_config.output_dims
        else:
            self.tokenizer = None
            lstm_input_dims = config.input_dims

        # We only support 1D spaces right now.
        assert len(lstm_input_dims) == 1
        lstm_input_dim = lstm_input_dims[0]

        lstm_weights_initializer = get_initializer_fn(
            config.hidden_weights_initializer, framework="torch"
        )
        lstm_bias_initializer = get_initializer_fn(
            config.hidden_bias_initializer, framework="torch"
        )

        # Create the torch LSTM layer.
        self.lstm = nn.LSTM(
            lstm_input_dim,
            config.hidden_dim,
            config.num_layers,
            batch_first=config.batch_major,
            bias=config.use_bias,
        )

        # Initialize LSTM layer weigths and biases, if necessary.
        for layer in self.lstm.all_weights:
            if lstm_weights_initializer:
                lstm_weights_initializer(
                    layer[0], **config.hidden_weights_initializer_config or {}
                )
                lstm_weights_initializer(
                    layer[1], **config.hidden_weights_initializer_config or {}
                )
            if lstm_bias_initializer:
                lstm_bias_initializer(
                    layer[2], **config.hidden_bias_initializer_config or {}
                )
                lstm_bias_initializer(
                    layer[3], **config.hidden_bias_initializer_config or {}
                )

        self._state_in_out_spec = {
            "h": TensorSpec(
                "b, l, d",
                d=self.config.hidden_dim,
                l=self.config.num_layers,
                framework="torch",
            ),
            "c": TensorSpec(
                "b, l, d",
                d=self.config.hidden_dim,
                l=self.config.num_layers,
                framework="torch",
            ),
        }

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        return SpecDict(
            {
                Columns.OBS: {
                    "image": TensorSpec(
                        "b, t, w, h, c",
                        w=self.config.tokenizer_config.cnn_encoder_config.input_dims[0],
                        h=self.config.tokenizer_config.cnn_encoder_config.input_dims[1],
                        c=self.config.tokenizer_config.cnn_encoder_config.input_dims[2],
                        framework="torch",
                    ),
                    "labels": TensorSpec(
                        "b, t, d", d=self.config.tokenizer_config.labels_encoder_config.input_dims[0], framework="torch"
                    ),
                },
                Columns.STATE_IN: self._state_in_out_spec,
            }
        )

    @override(Model)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        outputs = {}

        out = _tokenize(self.tokenizer, inputs, framework="torch")

        # States are batch-first when coming in. Make them layers-first.
        states_in = tree.map_structure(
            lambda s: s.transpose(0, 1), inputs[Columns.STATE_IN]
        )

        out, states_out = self.lstm(out, (states_in["h"], states_in["c"]))
        states_out = {"h": states_out[0], "c": states_out[1]}

        # Insert them into the output dict.
        outputs[ENCODER_OUT] = out
        outputs[Columns.STATE_OUT] = tree.map_structure(
            lambda s: s.transpose(0, 1), states_out
        )
        return outputs

class TorchRMLearningSharedMLP(nn.Module):
    """A multi-layer perceptron with N dense layers.

    All layers (except for an optional additional extra output layer) share the same
    activation function, bias setup (use bias or not), and LayerNorm setup
    (use layer normalization or not).

    If `output_dim` (int) is not None, an additional, extra output dense layer is added,
    which might have its own activation function (e.g. "linear"). However, the output
    layer does NOT use layer normalization.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_layer_dims: List[int],
        shared_layers_indices: tuple = tuple(),
        shared_layers_ref: object = None,
        hidden_layer_activation: Union[str, Callable] = "relu",
        hidden_layer_use_bias: bool = True,
        hidden_layer_use_layernorm: bool = False,
        hidden_layer_weights_initializer: Optional[Union[str, Callable]] = None,
        hidden_layer_weights_initializer_config: Optional[Union[str, Callable]] = None,
        hidden_layer_bias_initializer: Optional[Union[str, Callable]] = None,
        hidden_layer_bias_initializer_config: Optional[Dict] = None,
        output_dim: Optional[int] = None,
        output_use_bias: bool = True,
        output_activation: Union[str, Callable] = "linear",
        output_weights_initializer: Optional[Union[str, Callable]] = None,
        output_weights_initializer_config: Optional[Dict] = None,
        output_bias_initializer: Optional[Union[str, Callable]] = None,
        output_bias_initializer_config: Optional[Dict] = None,
    ):
        """Initialize a TorchMLP object.

        Args:
            input_dim: The input dimension of the network. Must not be None.
            hidden_layer_dims: The sizes of the hidden layers. If an empty list, only a
                single layer will be built of size `output_dim`.
            hidden_layer_use_layernorm: Whether to insert a LayerNormalization
                functionality in between each hidden layer's output and its activation.
            hidden_layer_use_bias: Whether to use bias on all dense layers (excluding
                the possible separate output layer).
            hidden_layer_activation: The activation function to use after each layer
                (except for the output). Either a torch.nn.[activation fn] callable or
                the name thereof, or an RLlib recognized activation name,
                e.g. "ReLU", "relu", "tanh", "SiLU", or "linear".
            hidden_layer_weights_initializer: The initializer function or class to use
                forweights initialization in the hidden layers. If `None` the default
                initializer of the respective dense layer is used. Note, only the
                in-place initializers, i.e. ending with an underscore "_" are allowed.
            hidden_layer_weights_initializer_config: Configuration to pass into the
                initializer defined in `hidden_layer_weights_initializer`.
            hidden_layer_bias_initializer: The initializer function or class to use for
                bias initialization in the hidden layers. If `None` the default
                initializer of the respective dense layer is used. Note, only the
                in-place initializers, i.e. ending with an underscore "_" are allowed.
            hidden_layer_bias_initializer_config: Configuration to pass into the
                initializer defined in `hidden_layer_bias_initializer`.
            output_dim: The output dimension of the network. If None, no specific output
                layer will be added and the last layer in the stack will have
                size=`hidden_layer_dims[-1]`.
            output_use_bias: Whether to use bias on the separate output layer,
                if any.
            output_activation: The activation function to use for the output layer
                (if any). Either a torch.nn.[activation fn] callable or
                the name thereof, or an RLlib recognized activation name,
                e.g. "ReLU", "relu", "tanh", "SiLU", or "linear".
            output_layer_weights_initializer: The initializer function or class to use
                for weights initialization in the output layers. If `None` the default
                initializer of the respective dense layer is used. Note, only the
                in-place initializers, i.e. ending with an underscore "_" are allowed.
            output_layer_weights_initializer_config: Configuration to pass into the
                initializer defined in `output_layer_weights_initializer`.
            output_layer_bias_initializer: The initializer function or class to use for
                bias initialization in the output layers. If `None` the default
                initializer of the respective dense layer is used. Note, only the
                in-place initializers, i.e. ending with an underscore "_" are allowed.
            output_layer_bias_initializer_config: Configuration to pass into the
                initializer defined in `output_layer_bias_initializer`.
        """
        super().__init__()
        assert input_dim > 0

        self.input_dim = input_dim

        hidden_activation = get_activation_fn(
            hidden_layer_activation, framework="torch"
        )
        hidden_weights_initializer = get_initializer_fn(
            hidden_layer_weights_initializer, framework="torch"
        )
        hidden_bias_initializer = get_initializer_fn(
            hidden_layer_bias_initializer, framework="torch"
        )
        output_weights_initializer = get_initializer_fn(
            output_weights_initializer, framework="torch"
        )
        output_bias_initializer = get_initializer_fn(
            output_bias_initializer, framework="torch"
        )

        if shared_layers_ref:
            try:
                shared_layers_actor = ray.get_actor(shared_layers_ref)

                def get_shared_layer(i):
                    if i in shared_layers_indices:
                        return ray.get(
                            shared_layers_actor.get_shared_layer.remote(i)
                        )
                    return None

                get_shared_layer = lambda i: ray.get(
                    shared_layers_actor.get_shared_layer.remote(i)
                )
            except ValueError:
                print("Default to normal behaviour, without shared layers")
                get_shared_layer = lambda i: None
        else:
            get_shared_layer = lambda i: None

        layers = []
        dims = (
            [self.input_dim]
            + list(hidden_layer_dims)
            + ([output_dim] if output_dim else [])
        )
        for i in range(0, len(dims) - 1):
            # Whether we are already processing the last (special) output layer.
            is_output_layer = output_dim is not None and i == len(dims) - 2

            layer = get_shared_layer(i)
            if layer is None:
                layer = nn.Linear(
                    dims[i],
                    dims[i + 1],
                    bias=output_use_bias if is_output_layer else hidden_layer_use_bias,
                )
                # Initialize layers, if necessary.
                if is_output_layer:
                    # Initialize output layer weigths if necessary.
                    if output_weights_initializer:
                        output_weights_initializer(
                            layer.weight, **output_weights_initializer_config or {}
                        )
                    # Initialize output layer bias if necessary.
                    if output_bias_initializer:
                        output_bias_initializer(
                            layer.bias, **output_bias_initializer_config or {}
                        )
                # Must be hidden.
                else:
                    # Initialize hidden layer weights if necessary.
                    if hidden_layer_weights_initializer:
                        hidden_weights_initializer(
                            layer.weight, **hidden_layer_weights_initializer_config or {}
                        )
                    # Initialize hidden layer bias if necessary.
                    if hidden_layer_bias_initializer:
                        hidden_bias_initializer(
                            layer.bias, **hidden_layer_bias_initializer_config or {}
                        )

            layers.append(layer)

            # We are still in the hidden layer section: Possibly add layernorm and
            # hidden activation.
            if not is_output_layer:
                # Insert a layer normalization in between layer's output and
                # the activation.
                if hidden_layer_use_layernorm:
                    # We use an epsilon of 0.001 here to mimick the Tf default behavior.
                    layers.append(nn.LayerNorm(dims[i + 1], eps=0.001))
                # Add the activation function.
                if hidden_activation is not None:
                    layers.append(hidden_activation())

        # Add output layer's (if any) activation.
        output_activation = get_activation_fn(output_activation, framework="torch")
        if output_dim is not None and output_activation is not None:
            layers.append(output_activation())

        self.mlp = nn.Sequential(*layers)

        self.expected_input_dtype = torch.float32

    def forward(self, x):
        return self.mlp(x.type(self.expected_input_dtype))
    
@ray.remote
class SharedLayers:
    def __init__(
            self,
            hidden_layer_dims: List[int],
            hidden_layer_activation: Union[str, Callable] = "relu",
            hidden_layer_use_bias: bool = True,
            hidden_layer_use_layernorm: bool = False,
            hidden_layer_weights_initializer: Optional[Union[str, Callable]] = None,
            hidden_layer_weights_initializer_config: Optional[Union[str, Callable]] = None,
            hidden_layer_bias_initializer: Optional[Union[str, Callable]] = None,
            hidden_layer_bias_initializer_config: Optional[Dict] = None,
        ):

        hidden_activation = get_activation_fn(
            hidden_layer_activation, framework="torch"
        )
        hidden_weights_initializer = get_initializer_fn(
            hidden_layer_weights_initializer, framework="torch"
        )
        hidden_bias_initializer = get_initializer_fn(
            hidden_layer_bias_initializer, framework="torch"
        )

        layers = []
        for i in range(0, len(hidden_layer_dims)-1):

            layer = nn.Linear(
                hidden_layer_dims[i],
                hidden_layer_dims[i + 1],
                bias=hidden_layer_use_bias,
            )


            if hidden_layer_weights_initializer:
                hidden_weights_initializer(
                    layer.weight, **hidden_layer_weights_initializer_config or {}
                )
            if hidden_layer_bias_initializer:
                hidden_bias_initializer(
                    layer.bias, **hidden_layer_bias_initializer_config or {}
                )
            
            layers.append(layer)

            if hidden_layer_use_layernorm:
                # We use an epsilon of 0.001 here to mimick the Tf default behavior.
                layers.append(nn.LayerNorm(hidden_layer_dims[i], eps=0.001))
            if hidden_activation is not None:
                layers.append(hidden_activation())
            
        self._layers = layers

    def get_shared_layer(self, i):
        return self._layers[i-1]