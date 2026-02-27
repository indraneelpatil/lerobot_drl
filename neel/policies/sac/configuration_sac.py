"""
Created by Indraneel on 02/26/2026

Configuration for the SAC Policy
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.optim.optimizers import MultiAdamConfig


def is_image_feature(key: str) -> bool:
    """Check if feature key represents image feature"""
    return key.startswith(OBS_IMAGE)

@dataclass
class CriticNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    final_activation: str | None = None

@dataclass
class ConcurrencyConfig:
    """
    Docstring for ConcurrencyConfig

    Can be either threads or processes
    """
    actor: str = "threads"
    learner: str = "threads"

@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2

@dataclass
class ActorNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final:bool = True

@dataclass
class PolicyConfig:
    use_tanh_squash: bool = True
    std_min: float = 1e-5
    std_max: float = 10.0
    init_final: float = 0.05

@PreTrainedConfig.register_subclass("sac")
@dataclass
class SACConfig(PreTrainedConfig):
    """ SAC configuration
    
    
    """

    # Mapping of feature types to normalization modes
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        }
    )

    # Statistics for normalizing different types of inputs
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            OBS_IMAGE: {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            OBS_STATE: {
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
            ACTION: {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            }
        }
    )

    # Architecture specifics
    # Device to run the model on 
    device: str = "cpu"
    # Device to store the model on
    storage_device: str = "cpu"
    # Name of the vision encoder model
    vision_encoder_name: str | None = None
    # Whether to freeze the vision encoder during training
    freeze_vision_encoder: bool = True
    # Hidden dimension size for the image encoder
    image_encoder_hidden_dim: int = 32
    # Whether to use a shared encoder for actor and critic
    shared_encoder: bool = True
    # Number of discrete actions
    num_discrete_actions: int | None = None
    # Dimension of image embedding pooling
    image_embedding_pooling_dim: int = 8

    # Training parameter
    # Number of steps for online training
    online_steps: int = 1000000
    # Capacity of the online replay buffer
    online_buffer_capacity: int = 100000
    # Capacity of the offline replay buffer
    offline_buffer_capacity: int = 100000
    # Use Asynchronoys prefetching for the buffers
    async_prefetch: bool = False
    # Number of steps before learning starts
    online_step_before_learning: int = 100
    # Frequency of policy updates
    policy_update_freq: int = 1

    # SAC Algorithm parameters
    # Discount factor for the SAC algorithm
    discount: float = 0.99
    # Initial temperature value
    temperature_init: float = 1.0
    # Number of critics in the ensemble
    num_critics: int = 2
    # Number of subsampled critics for training
    num_subsample_critics: int | None = None
    # Learning rate for the critic network
    critic_lr: float = 3e-4
    # Learning rate for the actor network
    actor_lr: float = 3e-4
    # Learning rate for the temeperature parameter
    temperature_lr: float = 3e-4
    # Weight for the critic target update 
    critic_target_update_weight: float = 0.005
    # Update to data ratio
    utd_ratio: int = 1
    # Hidden dimension size for the state encoder
    state_encoder_hidden_dim: int = 256
    # Dimension of the latent space
    latent_dim: int = 256
    # Target entropy for the SAC algorithm
    target_entropy: float | None = None
    # Backup entropy for the SAC algorithm
    use_backup_entropy: bool = True
    # Gradient clipping norm for the SAC algorithm
    grad_clip_norm: float = 40.0

    # Network configuration
    # Configuration for the critic network architecture
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Config for actor network architecture
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    # Config for the policy parameters
    policy_kwargs: PolicyConfig= field(default_factory=PolicyConfig)
    # Configuration for the discrete critic network
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for actor-learner architecture
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # Configuration for concurrency settings
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Optimizations
    use_torch_compile: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Any validation specific to SAC configuration
    
    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
                "temperature": {"lr": self.temperature_lr}
            }
        )
    
    def get_scheduler_preset(self) -> None:
        return None


    def validate_features(self) -> None:
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either `observation.state` or an image observation"
            )

        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action` in the output features")
    
    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]
    
    @property
    def observation_delta_indices(self) -> list:
        return None
    

    @property
    def action_delta_indices(self) -> list:
        return None # SAC typically predicts one action at a time
    
    @property
    def reward_delta_indices(self) -> None:
        return None