"""RSSM (Recurrent State-Space Model) for DreamerV3.

State = (h: deterministic GRU hidden, z: stochastic discrete categorical)
- Sequence model: h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1}))
- Prior (dynamics): p(z_t | h_t)
- Posterior (representation): q(z_t | h_t, embed_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class RSSM(nn.Module):

    def __init__(self, embed_dim: int, action_size: int, deter_size: int = 4096,
                 stoch_size: int = 32, num_classes: int = 32, mlp_units: int = 640,
                 mlp_layers: int = 3, unimix: float = 0.01):
        super().__init__()
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.num_classes = num_classes
        self.unimix = unimix

        stoch_dim = stoch_size * num_classes

        # Input projection for GRU: concat(z_{t-1}, a_{t-1}) -> deter_size
        self.input_proj = nn.Sequential(
            nn.Linear(stoch_dim + action_size, mlp_units),
            nn.SiLU(),
        )

        # GRU sequence model
        self.gru = nn.GRUCell(mlp_units, deter_size)

        # Prior: h_t -> logits for z_t
        prior_layers = []
        in_dim = deter_size
        for _ in range(mlp_layers):
            prior_layers.append(nn.Linear(in_dim, mlp_units))
            prior_layers.append(nn.SiLU())
            in_dim = mlp_units
        prior_layers.append(nn.Linear(mlp_units, stoch_dim))
        self.prior_net = nn.Sequential(*prior_layers)

        # Posterior: concat(h_t, embed_t) -> logits for z_t
        post_layers = []
        in_dim = deter_size + embed_dim
        for _ in range(mlp_layers):
            post_layers.append(nn.Linear(in_dim, mlp_units))
            post_layers.append(nn.SiLU())
            in_dim = mlp_units
        post_layers.append(nn.Linear(mlp_units, stoch_dim))
        self.posterior_net = nn.Sequential(*post_layers)

    @property
    def state_dim(self) -> int:
        """Total model state dimension (deterministic + stochastic)."""
        return self.deter_size + self.stoch_size * self.num_classes

    def initial_state(self, batch_size: int, device: torch.device) -> dict:
        """Return zero-initialized state."""
        return {
            "h": torch.zeros(batch_size, self.deter_size, device=device),
            "z": torch.zeros(batch_size, self.stoch_size * self.num_classes, device=device),
        }

    def observe(self, embed: torch.Tensor, action: torch.Tensor,
                is_first: torch.Tensor, state: dict) -> dict:
        """Process a sequence of observations through the RSSM.

        Args:
            embed: (B, T, embed_dim) encoded observations.
            action: (B, T, action_size) actions taken.
            is_first: (B, T) flags for episode starts.
            state: Initial state dict with 'h' and 'z'.

        Returns:
            Dict with keys:
                h: (B, T, deter_size) deterministic states
                z: (B, T, stoch_dim) posterior stochastic states
                prior_logits: (B, T, stoch_size, num_classes)
                post_logits: (B, T, stoch_size, num_classes)
        """
        B, T = embed.shape[:2]
        h_list, z_list = [], []
        prior_logits_list, post_logits_list = [], []

        h, z = state["h"], state["z"]

        for t in range(T):
            # Reset state on episode boundaries
            mask = (1.0 - is_first[:, t]).unsqueeze(-1)
            h = h * mask
            z = z * mask

            # Shift actions: use previous action (action at t is taken after observing t)
            a = action[:, t]

            # Sequence model: update deterministic state
            inp = self.input_proj(torch.cat([z, a], dim=-1))
            h = self.gru(inp, h)

            # Prior
            prior_logit = self.prior_net(h).reshape(B, self.stoch_size, self.num_classes)

            # Posterior
            post_logit = self.posterior_net(
                torch.cat([h, embed[:, t]], dim=-1)
            ).reshape(B, self.stoch_size, self.num_classes)

            # Sample from posterior with unimix and straight-through
            z = self._sample(post_logit)

            h_list.append(h)
            z_list.append(z)
            prior_logits_list.append(prior_logit)
            post_logits_list.append(post_logit)

        return {
            "h": torch.stack(h_list, dim=1),
            "z": torch.stack(z_list, dim=1),
            "prior_logits": torch.stack(prior_logits_list, dim=1),
            "post_logits": torch.stack(post_logits_list, dim=1),
            # Final state for continuation
            "final_h": h,
            "final_z": z,
        }

    def imagine(self, action: torch.Tensor, state: dict) -> dict:
        """Imagine forward one step using the prior (no observation).

        Args:
            action: (B, action_size) action to take.
            state: Current state dict.

        Returns:
            Next state dict with 'h', 'z', 'prior_logits'.
        """
        h, z = state["h"], state["z"]

        inp = self.input_proj(torch.cat([z, action], dim=-1))
        h = self.gru(inp, h)

        prior_logit = self.prior_net(h).reshape(
            h.shape[0], self.stoch_size, self.num_classes
        )
        z = self._sample(prior_logit)

        return {"h": h, "z": z, "prior_logits": prior_logit}

    def get_state_vector(self, state: dict) -> torch.Tensor:
        """Concatenate h and z into a single state vector."""
        return torch.cat([state["h"], state["z"]], dim=-1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution with unimix and straight-through.

        Args:
            logits: (B, stoch_size, num_classes) unnormalized log-probs.

        Returns:
            (B, stoch_size * num_classes) flattened one-hot samples.
        """
        # Apply unimix regularization
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / self.num_classes
        probs = (1.0 - self.unimix) * probs + self.unimix * uniform

        # Sample with straight-through gradient
        dist = D.OneHotCategorical(probs=probs)
        sample = dist.sample()
        # Straight-through: gradient flows through probs
        sample = sample + probs - probs.detach()

        return sample.reshape(sample.shape[0], -1)

    @staticmethod
    def kl_loss(prior_logits: torch.Tensor, post_logits: torch.Tensor,
                free_nats: float = 1.0, balance: float = 0.8) -> torch.Tensor:
        """Compute KL loss with free nats and KL balancing.

        Args:
            prior_logits: (B, T, stoch_size, num_classes)
            post_logits: (B, T, stoch_size, num_classes)
            free_nats: Minimum KL threshold.
            balance: Weight for forward vs reverse KL (0.8 = 80% prior learning).

        Returns:
            Scalar KL loss.
        """
        prior_dist = D.Independent(
            D.OneHotCategorical(logits=prior_logits), 1
        )
        post_dist = D.Independent(
            D.OneHotCategorical(logits=post_logits), 1
        )

        # Forward KL: trains the prior to match posterior (sg on posterior)
        post_dist_sg = D.Independent(
            D.OneHotCategorical(logits=post_logits.detach()), 1
        )
        kl_forward = D.kl_divergence(post_dist_sg, prior_dist)

        # Reverse KL: trains the posterior to match prior (sg on prior)
        prior_dist_sg = D.Independent(
            D.OneHotCategorical(logits=prior_logits.detach()), 1
        )
        kl_reverse = D.kl_divergence(post_dist, prior_dist_sg)

        # Apply free nats
        kl_forward = torch.clamp(kl_forward, min=free_nats)
        kl_reverse = torch.clamp(kl_reverse, min=free_nats)

        # Balanced KL
        kl = balance * kl_forward + (1.0 - balance) * kl_reverse
        return kl.mean()
