"""DreamerV3 Agent: integrates World Model, Actor, and Critic."""

import torch
import torch.nn as nn
import numpy as np

from dreamer.config import DreamerConfig
from dreamer.world_model import WorldModel
from dreamer.models.actor import Actor
from dreamer.models.critic import Critic
from dreamer.utils.math_utils import symlog, ReturnNormalizer


class DreamerV3Agent(nn.Module):
    """Full DreamerV3 agent with world model learning and behavior learning."""

    def __init__(self, action_size: int, config: DreamerConfig):
        super().__init__()
        self.config = config
        self.action_size = action_size

        # World Model
        self.world_model = WorldModel(action_size, config)

        state_dim = self.world_model.rssm.state_dim

        # Actor-Critic
        self.actor = Actor(
            state_dim=state_dim,
            action_size=action_size,
            mlp_units=config.mlp_units,
            mlp_layers=config.mlp_layers,
        )
        self.critic = Critic(
            state_dim=state_dim,
            mlp_units=config.mlp_units,
            mlp_layers=config.mlp_layers,
            num_bins=config.num_bins,
            ema_decay=config.critic_ema_decay,
        )

        # Return normalization
        self.return_normalizer = ReturnNormalizer(
            percentile_low=config.return_norm_low,
            percentile_high=config.return_norm_high,
        )

        # Optimizers
        self.world_opt = torch.optim.Adam(
            self.world_model.parameters(),
            lr=config.lr_world,
            weight_decay=config.weight_decay,
        )
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.lr_actor,
            weight_decay=config.weight_decay,
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.critic.parameters(),
            lr=config.lr_critic,
            weight_decay=config.weight_decay,
        )

        # Running RSSM state for online interaction
        self._rssm_state = None

    def init_state(self, batch_size: int = 1) -> None:
        """Initialize the RSSM state for environment interaction."""
        device = next(self.parameters()).device
        self._rssm_state = self.world_model.rssm.initial_state(batch_size, device)

    @torch.no_grad()
    def policy(self, obs: np.ndarray, is_first: bool = False,
               training: bool = True) -> np.ndarray:
        """Select action for a single environment step.

        Args:
            obs: (H, W, 3) uint8 image observation.
            is_first: Whether this is the first step of an episode.
            training: Whether to add exploration noise.

        Returns:
            (action_size,) numpy action array.
        """
        device = next(self.parameters()).device

        if self._rssm_state is None or is_first:
            self.init_state(1)

        # Preprocess image
        image = torch.from_numpy(obs).float().to(device) / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # Encode
        embed = self.world_model.encoder(image)

        # Create is_first tensor
        is_first_t = torch.tensor([[float(is_first)]], device=device)

        # Get zero action for the first step
        action_in = torch.zeros(1, 1, self.action_size, device=device)

        # Run single-step RSSM observe
        rssm_out = self.world_model.rssm.observe(
            embed.unsqueeze(1), action_in, is_first_t, self._rssm_state
        )

        # Update running state
        self._rssm_state = {
            "h": rssm_out["final_h"],
            "z": rssm_out["final_z"],
        }

        # Get action from actor
        state_vec = self.world_model.rssm.get_state_vector(self._rssm_state)
        action = self.actor(state_vec)

        if training:
            # Add exploration noise
            noise = torch.randn_like(action) * 0.3
            action = (action + noise).clamp(-1.0, 1.0)

        return action.squeeze(0).cpu().numpy()

    def train_step(self, data: dict) -> dict:
        """Perform one training step: world model + actor-critic.

        Args:
            data: Batch from replay buffer with shape (B, T, ...).

        Returns:
            Dict of loss metrics.
        """
        metrics = {}

        # ---- World Model Training ----
        rssm_out, wm_losses = self.world_model.observe(data)
        self.world_opt.zero_grad()
        wm_losses["total"].backward()
        nn.utils.clip_grad_norm_(
            self.world_model.parameters(), self.config.max_grad_norm
        )
        self.world_opt.step()

        metrics["wm_loss"] = wm_losses["total"].item()
        metrics["recon_loss"] = wm_losses["recon"].item()
        metrics["reward_loss"] = wm_losses["reward"].item()
        metrics["continue_loss"] = wm_losses["continue"].item()
        metrics["kl_loss"] = wm_losses["kl"].item()

        # ---- Behavior Learning (Actor-Critic in Imagination) ----
        with torch.no_grad():
            # Get posterior states as starting points for imagination
            B, T = data["image"].shape[:2]
            # Flatten and pick random starting states
            h_flat = rssm_out["h"].detach().reshape(B * T, -1)
            z_flat = rssm_out["z"].detach().reshape(B * T, -1)

            # Sample a subset of states as imagination starting points
            num_starts = B * T
            indices = torch.randperm(num_starts, device=h_flat.device)[:B]
            init_state = {
                "h": h_flat[indices],
                "z": z_flat[indices],
            }

        # Imagine trajectories
        self.actor.train()
        imagined = self.world_model.imagine(
            init_state, self.actor, self.config.imagine_horizon
        )

        # Compute lambda returns
        with torch.no_grad():
            values = self.critic.target_value(imagined["states"])
            returns = self._compute_lambda_returns(
                imagined["rewards"], values, imagined["continues"]
            )
            self.return_normalizer.update(returns)
            normalized_returns = self.return_normalizer.normalize(returns)

        # ---- Critic Update ----
        critic_loss = self.critic.loss(
            imagined["states"][:, :-1].detach(),
            returns[:, :-1].detach(),
        )
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.critic.parameters(), self.config.max_grad_norm
        )
        self.critic_opt.step()
        self.critic.update_target()
        metrics["critic_loss"] = critic_loss.item()

        # ---- Actor Update ----
        # Re-imagine to get gradients through actor
        imagined = self.world_model.imagine(
            init_state, self.actor, self.config.imagine_horizon
        )

        with torch.no_grad():
            values = self.critic.target_value(imagined["states"])
            returns = self._compute_lambda_returns(
                imagined["rewards"], values, imagined["continues"]
            )
            normalized_returns = self.return_normalizer.normalize(returns)

        # Actor loss: maximize normalized returns
        # DreamerV3 uses dynamics backprop (gradients flow through world model imagination)
        actor_loss = -normalized_returns[:, :-1].mean()

        # Entropy regularization
        state_for_entropy = imagined["states"][:, :-1].detach()
        entropy = self.actor.entropy(state_for_entropy).mean()
        actor_loss -= self.config.actor_entropy * entropy

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.max_grad_norm
        )
        self.actor_opt.step()

        metrics["actor_loss"] = actor_loss.item()
        metrics["entropy"] = entropy.item()

        return metrics

    def _compute_lambda_returns(
        self, rewards: torch.Tensor, values: torch.Tensor,
        continues: torch.Tensor
    ) -> torch.Tensor:
        """Compute GAE-style lambda returns.

        Args:
            rewards: (B, H) imagined rewards.
            values: (B, H) critic value estimates.
            continues: (B, H) continuation probabilities.

        Returns:
            (B, H) lambda returns.
        """
        gamma = self.config.gamma
        lam = self.config.lambda_gae
        H = rewards.shape[1]

        returns = torch.zeros_like(rewards)
        last_value = values[:, -1]

        for t in reversed(range(H)):
            if t == H - 1:
                next_value = last_value
            else:
                next_value = (1.0 - lam) * values[:, t + 1] + lam * returns[:, t + 1]

            returns[:, t] = rewards[:, t] + gamma * continues[:, t] * next_value

        return returns

    def save(self, path: str) -> None:
        """Save agent state to disk."""
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "world_opt": self.world_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load agent state from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.world_opt.load_state_dict(checkpoint["world_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])
