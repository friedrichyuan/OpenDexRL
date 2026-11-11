"""One DexJoCo task environment, hosted inside a simulator subprocess.

This module mirrors the DexJoCo OpenPI evaluation client
(``dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py``) so that RL rollouts
see exactly the observations and action conversion the pretrained policies were
evaluated with:

* images are resized with padding to 224x224 uint8 (``image_tools.resize_with_pad``),
* the policy state is the first 23 (single-arm) / 46 (dual-arm) entries of the
  flattened DexJoCo state, optionally zero-padded to 46 for the multi-task model,
* policy actions use rotation vectors and are converted to the quaternion
  layout the simulator executes,
* ``click_mouse`` replays the 30-step alignment prefix the evaluator uses.

The differences to the evaluator are deliberate and RL-specific: actions are
executed synchronously in chunks, the 30 Hz real-time sleep inside
``env.step`` is disabled, cameras are only rendered on steps whose observation
is consumed, and resets are seeded explicitly (equal seeds give equal
scenes). It imports no ML framework so the subprocess starts quickly.
"""

from __future__ import annotations

import random
import traceback
from typing import Any

import numpy as np

SINGLE_ARM_STATE_DIM = 23
DUAL_ARM_STATE_DIM = 46
SINGLE_ARM_ACTION_DIM = 22
DUAL_ARM_ACTION_DIM = 44

# Arm/hand pose the DexJoCo evaluator holds for 30 steps after a click_mouse
# reset so the first observation matches the demonstrations.
_CLICK_MOUSE_ALIGN_ACTION = np.array(
    [
        -4.4294e-01, 1.3729e-06, 1.5170e00,
        -3.14156462e00, -6.91584035e-05, -1.40317984e-03,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0.263, 0, 0, 0,
    ],
    dtype=np.float64,
)  # fmt: skip
_CLICK_MOUSE_ALIGN_STEPS = 30


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """PIL bilinear resize-with-pad, identical to ``openpi_client.image_tools``."""
    from PIL import Image

    if image.shape[-3:-1] == (height, width):
        return image
    pil = Image.fromarray(image)
    cur_width, cur_height = pil.size
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = pil.resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    canvas.paste(resized, (pad_width, pad_height))
    return np.asarray(canvas)


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """Match ``openpi_client.image_tools.convert_to_uint8``."""
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    return image


def rotvec_action_to_env_action(action: np.ndarray, dual_arm: bool) -> np.ndarray:
    """Convert a rotvec policy action to the quaternion action DexJoCo executes.

    Single-arm ``[xyz, rotvec, hand16]`` -> ``[xyz, quat_wxyz, hand16]``; dual-arm
    ``[r_xyz, r_rotvec, r_hand, l_xyz, l_rotvec, l_hand]`` ->
    ``[r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand]``.
    """
    from scipy.spatial.transform import Rotation

    if dual_arm:
        r_quat = Rotation.from_rotvec(action[3:6]).as_quat(scalar_first=True)
        l_quat = Rotation.from_rotvec(action[25:28]).as_quat(scalar_first=True)
        return np.concatenate(
            [action[:3], r_quat, action[22:25], l_quat, action[6:22], action[28:44]]
        )
    quat = Rotation.from_rotvec(action[3:6]).as_quat(scalar_first=True)
    return np.concatenate([action[:3], quat, action[6:22]])


def hold_pose_actions(states: np.ndarray, dual_arm: bool) -> np.ndarray:
    """Rotvec policy actions that hold the current pose (``DexJoCoOpenPIEnv.stay``).

    ``states`` is ``[N, 23]`` (single-arm) or ``[N, 46]`` (dual-arm) proprio.
    """
    from scipy.spatial.transform import Rotation

    actions = []
    for s in states:
        if dual_arm:
            r_rot = Rotation.from_quat(s[3:7], scalar_first=True).as_rotvec()
            l_rot = Rotation.from_quat(s[10:14], scalar_first=True).as_rotvec()
            actions.append(np.concatenate([s[:3], r_rot, s[14:30], s[7:10], l_rot, s[30:46]]))
        else:
            rot = Rotation.from_quat(s[3:7], scalar_first=True).as_rotvec()
            actions.append(np.concatenate([s[:3], rot, s[7:23]]))
    return np.stack(actions).astype(np.float32)


class DexJoCoTaskEnv:
    """A single DexJoCo task with the observation/action layout of the evaluator.

    Episode bookkeeping lives here so a whole action chunk runs without
    round-trips to the parent process. Once an episode ends (simulator
    termination or ``max_episode_steps``) the physics is frozen and later chunk
    steps only repeat the terminal flags; the parent masks those steps.
    """

    def __init__(
        self,
        task_cfg: dict[str, Any],
        *,
        rand_full: bool,
        randomize_dynamics: bool,
        seed: int,
        image_size: int,
        pad_state_dim46: bool,
        max_episode_steps: int,
    ) -> None:
        from dexjoco.tasks import CONFIG_MAPPING

        self.env_name = task_cfg["env_name"]
        self.camera_mapping: dict[str, str] = dict(task_cfg["camera_mapping"])
        self.dual_arm = task_cfg["robot_type"] == "dual_arm"
        self.image_size = image_size
        self.pad_state_dim46 = pad_state_dim46
        self.max_episode_steps = max_episode_steps

        env_kwargs = {}
        if self.env_name == "bimanual_unlock_ipad" and task_cfg.get("password"):
            env_kwargs["password"] = task_cfg["password"]
        self.env = CONFIG_MAPPING[self.env_name]().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=rand_full,
            seed=seed,
            randomize_dynamics=randomize_dynamics,
            **env_kwargs,
        )
        raw = self.env.unwrapped
        # ``step`` sleeps until 1/hz has passed; RL rollouts run faster than real time.
        raw.hz = float("inf")
        # Every task env renders through ``self.render()`` inside
        # ``_compute_observation``. Shadowing it on the instance lets us skip
        # rendering on steps whose images are never read.
        self._raw_render = raw.render
        self._render_enabled = True
        self._cached_frames = None
        raw.render = self._render

        self._obs: dict[str, np.ndarray] | None = None
        self._done = False
        self._terminated = False
        self._truncated = False
        self._success = False
        self._elapsed = 0

    def _render(self):
        if self._render_enabled or self._cached_frames is None:
            self._cached_frames = self._raw_render()
        return self._cached_frames

    def _process_obs(self, env_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        obs = {}
        for policy_key, env_key in self.camera_mapping.items():
            obs[policy_key] = convert_to_uint8(
                resize_with_pad(env_obs[env_key], self.image_size, self.image_size)
            )
        if self.dual_arm:
            state = env_obs["state"][:DUAL_ARM_STATE_DIM]
        else:
            state = env_obs["state"][:SINGLE_ARM_STATE_DIM]
            if self.pad_state_dim46:
                state = np.concatenate(
                    [state, np.zeros(DUAL_ARM_STATE_DIM - len(state))]
                )
        obs["state"] = np.asarray(state, dtype=np.float32)
        return obs

    def _env_step(self, policy_action: np.ndarray, render: bool):
        self._render_enabled = render
        env_action = rotvec_action_to_env_action(policy_action, self.dual_arm)
        return self.env.step(env_action)

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        """Reset with an explicit seed; equal seeds give equal initial states."""
        # DexJoCo samples object poses, table height and rand_full scene
        # parameters from the global ``random`` / ``np.random`` streams.
        random.seed(seed)
        np.random.seed(seed % (2**32))
        self._render_enabled = True
        env_obs, _ = self.env.reset()
        if self.env_name == "click_mouse":
            for i in range(_CLICK_MOUSE_ALIGN_STEPS):
                env_obs, *_ = self._env_step(
                    _CLICK_MOUSE_ALIGN_ACTION,
                    render=i == _CLICK_MOUSE_ALIGN_STEPS - 1,
                )
        self._obs = self._process_obs(env_obs)
        self._done = False
        self._terminated = False
        self._truncated = False
        self._success = False
        self._elapsed = 0
        return self._obs

    def chunk_step(self, actions: np.ndarray, return_frames: bool = False) -> dict[str, Any]:
        """Execute up to ``len(actions)`` steps; render only the chunk's last step.

        Returns per-step ``success`` (first success step only), ``terminated``,
        ``truncated`` and ``executed`` arrays plus the observation after the chunk.
        If the episode ends inside the chunk, the returned images are the last
        rendered frames (up to one chunk old); those steps are past the episode
        end and are masked by the trainer. With ``return_frames`` every executed
        step is rendered and its processed base-camera image is returned in
        ``frames`` (for videos).
        """
        assert self._obs is not None, "reset() must be called before chunk_step()"
        actions = np.asarray(actions, dtype=np.float64)
        action_dim = DUAL_ARM_ACTION_DIM if self.dual_arm else SINGLE_ARM_ACTION_DIM
        if actions.shape[-1] < action_dim:
            raise ValueError(
                f"{self.env_name}: expected actions with at least {action_dim} dims, "
                f"got shape {actions.shape}."
            )
        # The multi-task model emits 44 dims; single-arm tasks use the first 22.
        actions = actions[:, :action_dim]
        num_steps = actions.shape[0]
        success = np.zeros(num_steps, dtype=bool)
        terminated = np.zeros(num_steps, dtype=bool)
        truncated = np.zeros(num_steps, dtype=bool)
        executed = np.zeros(num_steps, dtype=bool)
        frames = []
        env_obs = None
        for i in range(num_steps):
            if not self._done:
                env_obs, _, term, _, info = self._env_step(
                    actions[i], render=return_frames or i == num_steps - 1
                )
                if return_frames:
                    base_key = self.camera_mapping["base"]
                    frames.append(
                        convert_to_uint8(
                            resize_with_pad(env_obs[base_key], self.image_size, self.image_size)
                        )
                    )
                self._elapsed += 1
                executed[i] = True
                if bool(info.get("succeed", False)) and not self._success:
                    self._success = True
                    success[i] = True
                self._terminated = bool(term)
                self._truncated = (
                    not self._terminated and self._elapsed >= self.max_episode_steps
                )
                self._done = self._terminated or self._truncated
            terminated[i] = self._terminated
            truncated[i] = self._truncated
        if env_obs is not None:
            self._obs = self._process_obs(env_obs)
        return {
            "obs": self._obs,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "executed": executed,
            "success_once": self._success,
            "elapsed": self._elapsed,
            "frames": frames,
        }

    def close(self) -> None:
        self.env.close()


def worker_main(remote, parent_remote, env_kwargs: dict[str, Any]) -> None:
    """Subprocess loop serving ``reset`` / ``chunk_step`` / ``close`` commands."""
    parent_remote.close()
    env = None
    try:
        env = DexJoCoTaskEnv(**env_kwargs)
        remote.send(("ok", None))
    except Exception:
        remote.send(("error", traceback.format_exc()))
        remote.close()
        return
    try:
        while True:
            cmd, payload = remote.recv()
            if cmd == "close":
                remote.send(("ok", None))
                break
            try:
                if cmd == "reset":
                    result = env.reset(payload)
                elif cmd == "chunk_step":
                    if isinstance(payload, dict):
                        result = env.chunk_step(**payload)
                    else:
                        result = env.chunk_step(payload)
                else:
                    raise ValueError(f"Unknown command {cmd!r}")
                remote.send(("ok", result))
            except Exception:
                remote.send(("error", traceback.format_exc()))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if env is not None:
            env.close()
        remote.close()
