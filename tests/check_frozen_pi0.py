"""GPU check of the frozen pi0.5 wrapper against DexJoCo's ``Policy.infer``.

Loads a DexJoCo SFT checkpoint as-is, then for the same observations and noise
compares ``FrozenPi0.infer`` (batched) with ``Policy.infer`` (one at a time, the
DexJoCo policy-server path), checks that different noise gives different
actions, and times both.

    CUDA_VISIBLE_DEVICES=0 python -m tests.check_frozen_pi0 --task water_plant
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="water_plant")
    parser.add_argument("--suite", default="rand_obj")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--models_root", default=os.environ.get(
        "DEXJOCO_MODELS_ROOT", str(_ROOT / "checkpoints" / "DexJoCo-Pi05")))
    args = parser.parse_args()

    from opendexrl.obs import policy_input
    from opendexrl.pi0_policy import FrozenPi0, default_checkpoint_dir, openpi_config_name
    from opendexrl.envs.subproc import SubprocEnvs, load_task_cfg

    dexjoco_root = pathlib.Path(os.environ.get("DEXJOCO_ROOT", _ROOT / "third_party" / "dexjoco"))
    task_cfg = load_task_cfg(dexjoco_root=str(dexjoco_root), task_suite=args.suite, task_name=args.task)
    envs = SubprocEnvs([
        dict(task_cfg=task_cfg, rand_full=args.suite == "rand_full", randomize_dynamics=False,
             image_size=224, pad_state_dim46=False, max_episode_steps=1000, seed=i)
        for i in range(args.batch)
    ])
    try:
        obs = envs.call("reset", range(args.batch), list(range(args.batch)))
    finally:
        envs.close()
    inputs = [policy_input(o, task_cfg["prompt"]) for o in obs]

    t = time.time()
    pi0 = FrozenPi0(
        dexjoco_root=dexjoco_root,
        checkpoint_dir=default_checkpoint_dir(args.models_root, args.task, args.suite),
        config_name=openpi_config_name(args.task, args.suite),
    )
    print(f"[check] loaded in {time.time() - t:.0f}s: horizon {pi0.action_horizon}, "
          f"model action dim {pi0.model_action_dim}")

    rng = np.random.default_rng(0)
    noise = rng.standard_normal((args.batch, pi0.action_horizon, pi0.model_action_dim)).astype(np.float32)

    t = time.time()
    batched = pi0.infer(inputs, noise)
    print(f"[check] batched infer (incl. compile) {time.time() - t:.1f}s -> {batched.shape}")
    t = time.time()
    for _ in range(3):
        batched = pi0.infer(inputs, noise)
    print(f"[check] batched infer B={args.batch}: {(time.time() - t) / 3 * 1000:.0f} ms/call")

    t = time.time()
    single = np.stack([
        pi0.policy.infer(dict(inp), noise=noise[i])["actions"] for i, inp in enumerate(inputs)
    ])
    print(f"[check] Policy.infer x{args.batch} (incl. compile) {time.time() - t:.1f}s")
    t = time.time()
    single = np.stack([
        pi0.policy.infer(dict(inp), noise=noise[i])["actions"] for i, inp in enumerate(inputs)
    ])
    print(f"[check] Policy.infer: {(time.time() - t) / args.batch * 1000:.0f} ms/obs")

    err = np.abs(batched - single).max()
    scale = np.abs(single).max()
    print(f"[check] batched vs Policy.infer: max |diff| {err:.2e} (max |action| {scale:.2f})")
    other = pi0.infer(inputs, -noise)
    spread = np.abs(other - batched).mean()
    print(f"[check] mean |a(noise) - a(-noise)| = {spread:.4f}  (noise steers the actions)")
    pos_err = np.abs(batched[:, 0, :3] - np.stack([o["state"][:3] for o in obs])).max()
    print(f"[check] first action xyz vs current tcp xyz: max |diff| {pos_err:.3f} m")
    assert batched.shape == (args.batch, pi0.action_horizon, 22 if task_cfg["robot_type"] != "dual_arm" else 44)
    assert err < 5e-2 * max(scale, 1.0), "batched inference disagrees with Policy.infer"
    assert spread > 1e-3, "actions do not depend on the injected noise"
    print("[check] OK")


if __name__ == "__main__":
    main()
