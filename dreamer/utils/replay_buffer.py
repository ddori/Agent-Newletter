"""Episode-based replay buffer for DreamerV3."""

import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    """Stores complete episodes and samples fixed-length sequences for training."""

    def __init__(self, capacity: int = 1_000_000):
        self._capacity = capacity
        self._episodes: deque[dict[str, np.ndarray]] = deque()
        self._total_steps = 0
        self._current_episode: dict[str, list] = {}

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def add_step(self, obs: dict, action: np.ndarray) -> None:
        """Add a single transition to the current episode."""
        if not self._current_episode:
            self._current_episode = {
                "image": [],
                "action": [],
                "reward": [],
                "is_first": [],
                "is_terminal": [],
            }

        self._current_episode["image"].append(obs["image"])
        self._current_episode["action"].append(action)
        self._current_episode["reward"].append(obs["reward"])
        self._current_episode["is_first"].append(obs["is_first"])
        self._current_episode["is_terminal"].append(obs["is_terminal"])
        self._total_steps += 1

    def end_episode(self) -> None:
        """Finalize and store the current episode."""
        if not self._current_episode or len(self._current_episode["image"]) == 0:
            return

        episode = {
            "image": np.stack(self._current_episode["image"]),         # (T, H, W, 3)
            "action": np.stack(self._current_episode["action"]),       # (T, A)
            "reward": np.array(self._current_episode["reward"], dtype=np.float32),
            "is_first": np.array(self._current_episode["is_first"], dtype=np.bool_),
            "is_terminal": np.array(self._current_episode["is_terminal"], dtype=np.bool_),
        }
        self._episodes.append(episode)
        self._current_episode = {}

        # Evict old episodes if over capacity
        while self._total_steps > self._capacity and len(self._episodes) > 1:
            removed = self._episodes.popleft()
            self._total_steps -= len(removed["reward"])

    def sample(self, batch_size: int, seq_length: int,
               device: torch.device) -> dict[str, torch.Tensor]:
        """Sample random sequences from stored episodes.

        Returns dict of tensors with shape (batch_size, seq_length, ...).
        """
        batch = {
            "image": [],
            "action": [],
            "reward": [],
            "is_first": [],
            "is_terminal": [],
        }

        for _ in range(batch_size):
            # Pick a random episode long enough
            valid_episodes = [e for e in self._episodes if len(e["reward"]) >= seq_length]
            if not valid_episodes:
                # Fallback: use longest episode with wraparound
                ep = max(self._episodes, key=lambda e: len(e["reward"]))
            else:
                idx = np.random.randint(len(valid_episodes))
                ep = valid_episodes[idx]

            ep_len = len(ep["reward"])
            if ep_len >= seq_length:
                start = np.random.randint(0, ep_len - seq_length + 1)
                end = start + seq_length
            else:
                start = 0
                end = ep_len

            for key in batch:
                segment = ep[key][start:end]
                # Pad if needed
                if len(segment) < seq_length:
                    pad_shape = (seq_length - len(segment),) + segment.shape[1:]
                    segment = np.concatenate([segment, np.zeros(pad_shape, dtype=segment.dtype)])
                batch[key].append(segment)

        result = {}
        for key in batch:
            arr = np.stack(batch[key])
            if key == "image":
                # (B, T, H, W, C) -> (B, T, C, H, W) normalized to [0, 1]
                tensor = torch.from_numpy(arr).float().to(device) / 255.0
                tensor = tensor.permute(0, 1, 4, 2, 3)
            elif key in ("is_first", "is_terminal"):
                tensor = torch.from_numpy(arr).float().to(device)
            else:
                tensor = torch.from_numpy(arr).float().to(device)
            result[key] = tensor

        return result
