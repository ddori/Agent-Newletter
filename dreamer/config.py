"""DreamerV3 hyperparameters and configuration."""

from dataclasses import dataclass


@dataclass
class DreamerConfig:
    # Environment
    env_name: str = "walker_walk"
    action_repeat: int = 2
    image_size: int = 64
    time_limit: int = 1000

    # RSSM
    deter_size: int = 4096
    stoch_size: int = 32
    num_classes: int = 32
    gru_layers: int = 1
    unimix: float = 0.01

    # CNN Encoder/Decoder
    cnn_depth: int = 48
    cnn_layers: int = 4
    cnn_kernels: tuple = (4, 4, 4, 4)

    # MLP
    mlp_units: int = 640
    mlp_layers: int = 3

    # World Model Training
    batch_size: int = 16
    seq_length: int = 64
    lr_world: float = 1e-4
    lr_actor: float = 3e-5
    lr_critic: float = 3e-5
    max_grad_norm: float = 1000.0
    weight_decay: float = 0.0

    # KL
    kl_free: float = 1.0
    kl_balance: float = 0.8

    # Imagination
    imagine_horizon: int = 15
    gamma: float = 0.997
    lambda_gae: float = 0.95

    # Actor
    actor_entropy: float = 3e-4
    actor_grad: str = "dynamics"  # "dynamics" or "reinforce"

    # Critic
    critic_ema_decay: float = 0.98
    num_bins: int = 255

    # Return normalization
    return_norm_low: float = 5.0
    return_norm_high: float = 95.0

    # Replay buffer
    buffer_capacity: int = 1_000_000
    prefill_steps: int = 5000

    # Training schedule
    total_steps: int = 1_000_000
    train_every: int = 5
    train_steps: int = 1
    eval_every: int = 10_000
    eval_episodes: int = 5
    log_every: int = 1000
    save_every: int = 50_000

    # General
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "logs"
