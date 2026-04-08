"""DreamerV3 training script.

Usage:
    python -m dreamer.train --env_name walker_walk --total_steps 1000000
"""

import argparse
import logging
import os
import time

import numpy as np
import torch

from dreamer.config import DreamerConfig
from dreamer.agent import DreamerV3Agent
from dreamer.utils.dmc_env import DMCEnv
from dreamer.utils.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


def make_env(config: DreamerConfig) -> DMCEnv:
    """Create a DMC environment from config."""
    domain, task = DMCEnv.parse_env_name(config.env_name)
    return DMCEnv(
        domain=domain,
        task=task,
        image_size=config.image_size,
        action_repeat=config.action_repeat,
        seed=config.seed,
    )


def evaluate(agent: DreamerV3Agent, config: DreamerConfig,
             num_episodes: int = 5) -> float:
    """Run evaluation episodes and return mean reward."""
    env = make_env(config)
    total_rewards = []

    for _ in range(num_episodes):
        obs = env.reset()
        agent.init_state(1)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.policy(obs["image"], is_first=obs["is_first"], training=False)
            obs = env.step(action)
            episode_reward += obs["reward"]
            done = obs["is_terminal"]

        total_rewards.append(episode_reward)

    return float(np.mean(total_rewards))


def train(config: DreamerConfig) -> None:
    """Main training loop."""
    # Setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)

    # Environment
    env = make_env(config)
    logger.info(f"Environment: {config.env_name}, action_size: {env.action_size}")

    # Agent
    agent = DreamerV3Agent(env.action_size, config).to(device)
    total_params = sum(p.numel() for p in agent.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Replay buffer
    buffer = ReplayBuffer(capacity=config.buffer_capacity)

    # Logging
    os.makedirs(config.log_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.log_dir)
    except ImportError:
        writer = None
        logger.warning("TensorBoard not available, skipping logging")

    # ---- Prefill with random actions ----
    logger.info(f"Prefilling buffer with {config.prefill_steps} random steps...")
    obs = env.reset()
    buffer.add_step(obs, np.zeros(env.action_size, dtype=np.float32))

    for _ in range(config.prefill_steps):
        action = np.random.uniform(-1, 1, size=env.action_size).astype(np.float32)
        obs = env.step(action)
        buffer.add_step(obs, action)

        if obs["is_terminal"]:
            buffer.end_episode()
            obs = env.reset()
            buffer.add_step(obs, np.zeros(env.action_size, dtype=np.float32))

    buffer.end_episode()
    logger.info(f"Prefill done. Buffer: {buffer.num_episodes} episodes, {buffer.total_steps} steps")

    # ---- Main training loop ----
    obs = env.reset()
    agent.init_state(1)
    buffer.add_step(obs, np.zeros(env.action_size, dtype=np.float32))

    episode_reward = 0.0
    episode_count = 0
    start_time = time.time()

    for step in range(1, config.total_steps + 1):
        # Act in environment
        action = agent.policy(obs["image"], is_first=obs["is_first"], training=True)
        obs = env.step(action)
        buffer.add_step(obs, action)
        episode_reward += obs["reward"]

        if obs["is_terminal"]:
            buffer.end_episode()
            episode_count += 1

            if writer:
                writer.add_scalar("train/episode_reward", episode_reward, step)
                writer.add_scalar("train/episodes", episode_count, step)

            logger.info(f"Step {step:>8d} | Episode {episode_count:>4d} | "
                       f"Reward: {episode_reward:>8.1f}")

            episode_reward = 0.0
            obs = env.reset()
            agent.init_state(1)
            buffer.add_step(obs, np.zeros(env.action_size, dtype=np.float32))

        # Train
        if step % config.train_every == 0 and buffer.num_episodes >= 1:
            for _ in range(config.train_steps):
                data = buffer.sample(config.batch_size, config.seq_length, device)
                metrics = agent.train_step(data)

            if step % config.log_every == 0:
                elapsed = time.time() - start_time
                fps = step / elapsed
                logger.info(f"Step {step:>8d} | FPS: {fps:.0f} | " +
                           " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

                if writer:
                    for k, v in metrics.items():
                        writer.add_scalar(f"train/{k}", v, step)

        # Evaluate
        if step % config.eval_every == 0:
            eval_reward = evaluate(agent, config, config.eval_episodes)
            logger.info(f"Step {step:>8d} | Eval reward: {eval_reward:.1f}")
            if writer:
                writer.add_scalar("eval/reward", eval_reward, step)

        # Save checkpoint
        if step % config.save_every == 0:
            ckpt_path = os.path.join(config.log_dir, f"checkpoint_{step}.pt")
            agent.save(ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final save
    agent.save(os.path.join(config.log_dir, "checkpoint_final.pt"))
    if writer:
        writer.close()
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="DreamerV3 Training")
    parser.add_argument("--env_name", type=str, default="walker_walk")
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--prefill_steps", type=int, default=5000)
    parser.add_argument("--imagine_horizon", type=int, default=15)
    args = parser.parse_args()

    config = DreamerConfig(
        env_name=args.env_name,
        total_steps=args.total_steps,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        action_repeat=args.action_repeat,
        prefill_steps=args.prefill_steps,
        imagine_horizon=args.imagine_horizon,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    train(config)


if __name__ == "__main__":
    main()
