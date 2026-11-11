"""Check the parallel DexJoCo simulators (no GPU or policy needed).

Resets three envs (two with the same seed), steps hold-pose chunks, and checks
that equal seeds give equal scenes, different seeds different ones, the arm
holds still, and per-step frames come back for videos.

    python -m tests.check_subproc_envs --tasks water_plant bimanual_hanoi
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np

from opendexrl.envs.dexjoco_task import hold_pose_actions
from opendexrl.envs.subproc import SubprocEnvs, load_task_cfg

_DEXJOCO_ROOT = pathlib.Path(
    os.environ.get("DEXJOCO_ROOT", pathlib.Path(__file__).resolve().parents[1] / "third_party" / "dexjoco")
)


def check_task(task: str, suite: str, chunk: int = 10, num_chunks: int = 5) -> None:
    cfg = load_task_cfg(dexjoco_root=str(_DEXJOCO_ROOT), task_suite=suite, task_name=task)
    kwargs = dict(
        task_cfg=cfg, rand_full=suite == "rand_full", randomize_dynamics=False,
        image_size=224, pad_state_dim46=False, max_episode_steps=1000,
    )
    t = time.time()
    envs = SubprocEnvs([{**kwargs, "seed": i} for i in range(3)])
    print(f"{task}/{suite}: started 3 envs in {time.time() - t:.1f}s")
    try:
        obs = envs.call("reset", range(3), [7, 7, 8])
        print("  obs:", {k: (v.shape, str(v.dtype)) for k, v in obs[0].items()})
        same = np.array_equal(obs[0]["base"], obs[1]["base"])
        diff = not np.array_equal(obs[0]["base"], obs[2]["base"])
        print(f"  same seed -> same scene: {same}; different seed -> different scene: {diff}")
        assert same and diff, "reset seeding is not deterministic per seed"
        actions = hold_pose_actions(np.stack([o["state"] for o in obs]), cfg["robot_type"] == "dual_arm")
        chunk_actions = np.repeat(actions[:, None], chunk, axis=1)
        t = time.time()
        for _ in range(num_chunks):
            results = envs.call("chunk_step", range(3), list(chunk_actions))
        dt = (time.time() - t) / num_chunks
        drift = max(np.abs(r["obs"]["state"][:3] - o["state"][:3]).max() for r, o in zip(results, obs))
        print(f"  {chunk}-step chunk x3 envs: {dt * 1000:.0f} ms; hold-pose xyz drift {drift:.4f} m; "
              f"elapsed {results[0]['elapsed']}; terminated {bool(results[0]['terminated'][-1])}")
        assert drift < 0.05, f"arm drifted {drift:.3f} m under hold-pose actions"
        res = envs.call("chunk_step", [0], [{"actions": chunk_actions[0], "return_frames": True}])[0]
        print(f"  return_frames: {len(res['frames'])} x {res['frames'][0].shape}")
        assert len(res["frames"]) == chunk
    finally:
        envs.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["water_plant", "bimanual_hanoi"])
    parser.add_argument("--suite", default="rand_obj")
    args = parser.parse_args()
    for task in args.tasks:
        check_task(task, args.suite)
    print("[check_subproc_envs] OK")


if __name__ == "__main__":
    main()
