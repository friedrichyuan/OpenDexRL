"""Frozen DexJoCo pi0.5 policy with externally supplied diffusion noise.

Loads the SFT checkpoint exactly as DexJoCo's policy server does
(``openpi.policies.policy_config.create_trained_policy`` on the DexJoCo
``TrainConfig`` of the task), so the LoRA weights, transforms and norm stats are
the original ones -- no conversion. On top of ``Policy.infer`` (single
observation) this adds batched inference: inputs are transformed per sample
like ``Policy.infer`` does, stacked, and sampled in one jitted call.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
from typing import Any

import numpy as np


@contextlib.contextmanager
def _chdir(path: pathlib.Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def import_dexjoco_openpi(dexjoco_root: str | os.PathLike):
    """Import DexJoCo's ``openpi.training.config`` and ``policy_config``.

    ``openpi/training/dexjoco_configs.py`` reads ``config.yaml`` from the current
    directory at import time, so the import runs inside ``dexjoco/openpi``.
    """
    with _chdir(pathlib.Path(dexjoco_root) / "openpi"):
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config
    return openpi_config, policy_config


def openpi_config_name(task: str, suite: str) -> str:
    """DexJoCo TrainConfig name for a task/suite (see dexjoco_configs.py)."""
    if suite == "multi_task":
        return "multi_task"
    if suite == "rand_full":
        return f"{task}_rand_full"
    if suite == "rand_obj":
        return task
    raise ValueError(f"Unknown suite {suite!r}; expected rand_obj, rand_full or multi_task.")


def default_checkpoint_dir(models_root: str | os.PathLike, task: str, suite: str) -> pathlib.Path:
    """Checkpoint layout of the DexJoCo-Pi05 release."""
    root = pathlib.Path(models_root)
    if suite == "multi_task":
        return root / "pi05_dexjoco_multi_task"
    sub = "pi05_dexjoco_ckpt" if suite == "rand_obj" else "pi05_dexjoco_rand_full_ckpt"
    return root / sub / task


class FrozenPi0:
    """Batched, noise-conditioned inference with a DexJoCo pi0.5 SFT checkpoint."""

    def __init__(
        self,
        *,
        dexjoco_root: str | os.PathLike,
        checkpoint_dir: str | os.PathLike,
        config_name: str,
        num_steps: int = 10,
        seed: int = 0,
    ) -> None:
        import jax

        openpi_config, policy_config = import_dexjoco_openpi(dexjoco_root)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.train_config = openpi_config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(
            self.train_config,
            self.checkpoint_dir,
            sample_kwargs={"num_steps": int(num_steps)},
        )
        model_cfg = self.train_config.model
        self.action_horizon = int(model_cfg.action_horizon)
        self.model_action_dim = int(model_cfg.action_dim)
        self._rng = jax.random.key(seed)

    def norm_stats(self) -> dict[str, dict[str, np.ndarray]]:
        """The checkpoint's norm stats (``assets/local_repo/norm_stats.json``)."""
        path = self.checkpoint_dir / "assets" / "local_repo" / "norm_stats.json"
        stats = json.loads(path.read_text())["norm_stats"]
        return {k: {kk: np.asarray(vv, dtype=np.float32) for kk, vv in v.items()} for k, v in stats.items()}

    def infer(self, obs_list: list[dict[str, Any]], noise: np.ndarray) -> np.ndarray:
        """Sample action chunks for a batch of observations.

        Args:
            obs_list: per-env DexJoCo policy inputs (camera keys, ``state``,
                ``prompt``), as produced by the DexJoCo OpenPI client wrapper.
            noise: ``[B, action_horizon, model_action_dim]`` initial noise.

        Returns:
            ``[B, action_horizon, env_action_dim]`` rotvec actions (unnormalized).
        """
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        noise = np.asarray(noise, dtype=np.float32)
        expected = (len(obs_list), self.action_horizon, self.model_action_dim)
        assert noise.shape == expected, f"Expected noise shape {expected}, got {noise.shape}."
        policy = self.policy
        inputs = [policy._input_transform(jax.tree.map(lambda x: x, obs)) for obs in obs_list]
        batched = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = policy._sample_actions(
            sample_rng,
            _model.Observation.from_dict(batched),
            noise=jnp.asarray(noise),
            **policy._sample_kwargs,
        )
        actions = np.asarray(actions)
        states = np.asarray(batched["state"])
        decoded = [
            policy._output_transform({"state": states[i], "actions": actions[i]})["actions"]
            for i in range(len(obs_list))
        ]
        return np.stack(decoded).astype(np.float32)
