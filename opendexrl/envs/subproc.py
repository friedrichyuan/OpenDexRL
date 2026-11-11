"""Parallel DexJoCo simulators: one subprocess per env.

Used by the trainer (:mod:`opendexrl.train`); each subprocess hosts a
:class:`opendexrl.envs.dexjoco_task.DexJoCoTaskEnv`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import yaml

from opendexrl.envs.dexjoco_task import worker_main


def load_task_cfg(
    *,
    dexjoco_root: str,
    task_suite: str,
    task_name: str,
    task_config: str | None = None,
) -> dict[str, Any]:
    """Load the DexJoCo evaluation YAML that defines cameras, prompt and arm type.

    ``task_config`` may be an explicit path; otherwise the file is
    ``{dexjoco_root}/configs/{task_suite}/{task_name}.yaml`` from the DexJoCo repo.
    """
    path = task_config or os.path.join(
        str(dexjoco_root), "configs", str(task_suite), f"{task_name}.yaml"
    )
    with open(path) as f:
        task_cfg = yaml.safe_load(f)
    for key in ("env_name", "camera_mapping", "prompt", "robot_type"):
        if key not in task_cfg:
            raise KeyError(f"DexJoCo task config {path} is missing {key!r}.")
    return task_cfg


class SubprocEnvs:
    """Pipes to ``len(env_kwargs_list)`` DexJoCo subprocesses; commands run in parallel."""

    def __init__(self, env_kwargs_list: list[dict[str, Any]], start_method: str = "spawn"):
        ctx = mp.get_context(start_method)
        self.remotes, self.processes = [], []
        for env_kwargs in env_kwargs_list:
            remote, work_remote = ctx.Pipe()
            proc = ctx.Process(
                target=worker_main,
                args=(work_remote, remote, env_kwargs),
                daemon=True,
            )
            proc.start()
            work_remote.close()
            self.remotes.append(remote)
            self.processes.append(proc)
        self._gather(range(len(self.remotes)))

    def __len__(self) -> int:
        return len(self.remotes)

    def _gather(self, env_ids) -> list[Any]:
        results, errors = [], []
        for env_id in env_ids:
            status, payload = self.remotes[env_id].recv()
            if status == "error":
                errors.append(f"[dexjoco env {env_id}]\n{payload}")
            results.append(payload)
        if errors:
            raise RuntimeError("DexJoCo subprocess failed:\n" + "\n".join(errors))
        return results

    def call(self, cmd: str, env_ids, payloads) -> list[Any]:
        env_ids = list(env_ids)
        payloads = list(payloads)
        assert len(env_ids) == len(payloads), (
            f"Expected one payload per env, got {len(payloads)} for {len(env_ids)} envs."
        )
        for env_id, payload in zip(env_ids, payloads):
            self.remotes[env_id].send((cmd, payload))
        return self._gather(env_ids)

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self.remotes, self.processes = [], []
