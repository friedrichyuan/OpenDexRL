"""Observation conversion for DSRL on DexJoCo.

``policy_input`` builds the pi0.5 input dict exactly as DexJoCo's OpenPI client
does (224x224 padded uint8 cameras keyed ``base``/``wrist``/``wrist_left``/
``wrist_right``, the proprio ``state`` and the ``prompt``); the task
subprocesses already produce that layout.

``SACObsBuilder`` produces DSRL's SAC observation: the chosen cameras resized
to ``resize`` and stacked along channels as ``pixels`` ``[H, W, 3*C, 1]`` uint8
(``jaxrl2`` layout), plus the proprio ``state`` ``[S, 1]`` mapped to roughly
[-1, 1] with the checkpoint's q01/q99 state quantiles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def policy_input(task_obs: dict[str, np.ndarray], prompt: str) -> dict[str, Any]:
    """pi0.5 input for one env (copies; transforms may modify inputs in place)."""
    out = {k: np.array(v) for k, v in task_obs.items()}
    out["prompt"] = prompt
    return out


class SACObsBuilder:
    def __init__(
        self,
        *,
        cameras: list[str],
        resize: int,
        state_stats: dict[str, np.ndarray] | None,
        add_states: bool = True,
    ) -> None:
        self.cameras = list(cameras)
        self.resize = int(resize)
        self.add_states = add_states
        self._q01 = self._q99 = None
        self.state_dim = None
        if add_states:
            assert state_stats is not None, "add_states requires the checkpoint state norm stats."
            self._q01 = np.asarray(state_stats["q01"], dtype=np.float32)
            self._q99 = np.asarray(state_stats["q99"], dtype=np.float32)
            self.state_dim = int(self._q01.shape[0])

    def observation_spaces(self, box_cls) -> dict:
        size = self.resize
        spaces = {
            "pixels": box_cls(low=0, high=255, shape=(size, size, 3 * len(self.cameras), 1), dtype=np.uint8)
        }
        if self.add_states:
            spaces["state"] = box_cls(low=-np.inf, high=np.inf, shape=(self.state_dim, 1), dtype=np.float32)
        return spaces

    def _image(self, img: np.ndarray) -> np.ndarray:
        if self.resize > 0 and img.shape[0] != self.resize:
            img = np.asarray(Image.fromarray(img).resize((self.resize, self.resize), Image.BILINEAR))
        return img

    def single(self, task_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = [c for c in self.cameras if c not in task_obs]
        if missing:
            raise KeyError(
                f"SAC cameras {missing} not in the task observation (keys {sorted(task_obs)})."
            )
        pixels = np.concatenate([self._image(task_obs[c]) for c in self.cameras], axis=-1)
        out = {"pixels": pixels[..., None].astype(np.uint8)}
        if self.add_states:
            state = np.asarray(task_obs["state"], dtype=np.float32)[: self.state_dim]
            scaled = 2.0 * (state - self._q01) / np.maximum(self._q99 - self._q01, 1e-6) - 1.0
            out["state"] = np.clip(scaled, -5.0, 5.0)[..., None].astype(np.float32)
        return out

    def batch(self, task_obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        singles = [self.single(o) for o in task_obs_list]
        return {k: np.stack([s[k] for s in singles]) for k in singles[0]}
