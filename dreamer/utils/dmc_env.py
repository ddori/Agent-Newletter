"""DeepMind Control Suite environment wrapper for DreamerV3."""

import numpy as np


class DMCEnv:
    """Wraps a dm_control environment into a simple step/reset interface.

    - Renders pixel observations (image_size x image_size RGB).
    - Applies action repeat.
    - Normalizes actions to [-1, 1].
    """

    def __init__(self, domain: str, task: str, image_size: int = 64,
                 action_repeat: int = 2, seed: int = 0):
        from dm_control import suite
        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        self._image_size = image_size
        self._action_repeat = action_repeat

        # Action spec
        action_spec = self._env.action_spec()
        self.action_size = int(action_spec.shape[0])
        self.action_low = action_spec.minimum.astype(np.float32)
        self.action_high = action_spec.maximum.astype(np.float32)

    def reset(self) -> dict:
        """Reset environment and return initial observation dict."""
        time_step = self._env.reset()
        obs = self._render()
        return {
            "image": obs,
            "reward": 0.0,
            "is_first": True,
            "is_terminal": False,
        }

    def step(self, action: np.ndarray) -> dict:
        """Take action with action repeat, return observation dict."""
        action = np.clip(action, self.action_low, self.action_high)
        total_reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            total_reward += time_step.reward or 0.0
            if time_step.last():
                break

        obs = self._render()
        return {
            "image": obs,
            "reward": total_reward,
            "is_first": False,
            "is_terminal": time_step.last(),
        }

    def _render(self) -> np.ndarray:
        """Render camera observation as uint8 numpy array (H, W, 3)."""
        return self._env.physics.render(
            height=self._image_size,
            width=self._image_size,
            camera_id=0,
        ).copy()

    @staticmethod
    def parse_env_name(env_name: str) -> tuple[str, str]:
        """Parse 'domain_task' string into (domain, task) tuple."""
        parts = env_name.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Expected 'domain_task' format, got '{env_name}'")
        return parts[0], parts[1]
