"""DreamerV3 World Model: integrates Encoder, RSSM, Decoder, and prediction heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.models.encoder import Encoder
from dreamer.models.decoder import Decoder
from dreamer.models.rssm import RSSM
from dreamer.models.heads import SymlogTwoHotHead, ContinueHead
from dreamer.utils.math_utils import symlog


class WorldModel(nn.Module):
    """DreamerV3 World Model.

    Components:
        - Encoder: image -> embedding
        - RSSM: sequence model with prior/posterior
        - Decoder: state -> image reconstruction
        - Reward head: state -> reward (symlog two-hot)
        - Continue head: state -> continuation probability
    """

    def __init__(self, action_size: int, config):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = Encoder(
            in_channels=3,
            depth=config.cnn_depth,
            layers=config.cnn_layers,
            kernels=config.cnn_kernels,
        )

        # RSSM
        self.rssm = RSSM(
            embed_dim=self.encoder.embed_dim,
            action_size=action_size,
            deter_size=config.deter_size,
            stoch_size=config.stoch_size,
            num_classes=config.num_classes,
            mlp_units=config.mlp_units,
            mlp_layers=config.mlp_layers,
            unimix=config.unimix,
        )

        state_dim = self.rssm.state_dim

        # Decoder
        self.decoder = Decoder(
            state_dim=state_dim,
            depth=config.cnn_depth,
            layers=config.cnn_layers,
            kernels=config.cnn_kernels,
        )

        # Prediction heads
        self.reward_head = SymlogTwoHotHead(
            in_dim=state_dim,
            mlp_units=config.mlp_units,
            mlp_layers=config.mlp_layers,
            num_bins=config.num_bins,
        )
        self.continue_head = ContinueHead(
            in_dim=state_dim,
            mlp_units=config.mlp_units,
            mlp_layers=config.mlp_layers,
        )

    def observe(self, data: dict, state: dict = None) -> tuple[dict, dict]:
        """Process a batch of sequences through the world model.

        Args:
            data: Dict with keys 'image', 'action', 'reward', 'is_first', 'is_terminal'.
                  Each has shape (B, T, ...).
            state: Optional initial RSSM state.

        Returns:
            (rssm_output, losses) tuple.
        """
        B, T = data["image"].shape[:2]
        device = data["image"].device

        if state is None:
            state = self.rssm.initial_state(B, device)

        # Encode all observations
        images_flat = data["image"].reshape(B * T, *data["image"].shape[2:])
        embed_flat = self.encoder(images_flat)
        embed = embed_flat.reshape(B, T, -1)

        # Run RSSM
        rssm_out = self.rssm.observe(embed, data["action"], data["is_first"], state)

        # Get state vectors for predictions
        states = torch.cat([rssm_out["h"], rssm_out["z"]], dim=-1)  # (B, T, state_dim)
        states_flat = states.reshape(B * T, -1)

        # Compute losses
        losses = {}

        # Image reconstruction loss (MSE in symlog space)
        recon = self.decoder(states_flat).reshape(B, T, *data["image"].shape[2:])
        recon_loss = F.mse_loss(recon, symlog(data["image"]))
        losses["recon"] = recon_loss

        # Reward prediction loss
        reward_loss = self.reward_head.loss(states_flat, data["reward"].reshape(B * T))
        losses["reward"] = reward_loss

        # Continue prediction loss
        # Target: 1.0 for non-terminal, 0.0 for terminal
        continue_target = 1.0 - data["is_terminal"]
        continue_loss = self.continue_head.loss(states_flat, continue_target.reshape(B * T))
        losses["continue"] = continue_loss

        # KL loss
        kl_loss = RSSM.kl_loss(
            rssm_out["prior_logits"],
            rssm_out["post_logits"],
            free_nats=self.config.kl_free,
            balance=self.config.kl_balance,
        )
        losses["kl"] = kl_loss

        # Total world model loss
        losses["total"] = recon_loss + reward_loss + continue_loss + kl_loss

        return rssm_out, losses

    def imagine(self, initial_state: dict, actor: nn.Module,
                horizon: int) -> dict:
        """Imagine trajectories using the actor policy.

        Args:
            initial_state: Starting RSSM state dict.
            actor: Actor network that maps state_vector -> action.
            horizon: Number of imagination steps.

        Returns:
            Dict with imagined trajectory:
                states: (B, H, state_dim)
                actions: (B, H, action_size)
                rewards: (B, H)
                continues: (B, H)
        """
        state = {k: v for k, v in initial_state.items()}
        states, actions, rewards, continues = [], [], [], []

        for _ in range(horizon):
            state_vec = self.rssm.get_state_vector(state)

            # Actor selects action
            action = actor(state_vec).detach() if not actor.training else actor(state_vec)

            # Imagine next state
            state = self.rssm.imagine(action, state)
            next_state_vec = self.rssm.get_state_vector(state)

            # Predict reward and continuation
            reward = self.reward_head.predict(next_state_vec)
            cont = self.continue_head.predict(next_state_vec)

            states.append(next_state_vec)
            actions.append(action)
            rewards.append(reward)
            continues.append(cont)

        return {
            "states": torch.stack(states, dim=1),
            "actions": torch.stack(actions, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "continues": torch.stack(continues, dim=1),
        }
