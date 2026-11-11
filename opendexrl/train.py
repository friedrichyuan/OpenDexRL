"""DSRL training on DexJoCo: SAC over pi0.5's initial noise, pi0.5 frozen.

Follows ``dsrl_pi0/examples/train_utils_sim.py`` (Wagenmaker et al., 2025):

* the SAC action is ``noise_rows`` noise vectors of size ``model_action_dim``:
  row k is pi0.5's initial noise for action k and the last row is repeated
  over the rest of the action horizon (``noise_rows=1``, the default of
  ``dsrl_pi0``, repeats one vector everywhere); ``action_magnitude`` bounds it
  via tanh;
* one RL step = one pi0.5 query; ``query_freq`` of the returned actions are
  executed open-loop before the next query;
* sparse reward per query step: -1, and 0 on the step that reaches success
  (mask 0); time-outs keep mask 1 (bootstrapped); per-transition discount is
  ``discount ** query_freq``;
* the first ``start_online_updates`` transitions use standard-normal noise (the
  base SFT policy), then ``multi_grad_step`` SAC updates per new transition.

Differences to the reference loop: ``num_envs`` DexJoCo simulators run in
parallel with batched pi0.5 inference; transitions of an env enter the buffer
when its episode ends; early failure terminations (e.g. water_plant spraying
outside the pot, an illegal Hanoi state) get the floor value
``-1 / (1 - discount ** query_freq)`` with mask 0 instead of being bootstrapped
(``--failure_terminal floor``; ``bootstrap`` reproduces the reference).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections import deque

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a = p.add_argument
    # task / checkpoint
    a("--task", default="water_plant")
    a("--suite", default="rand_obj", choices=["rand_obj", "rand_full"])
    a("--dexjoco_root", default=os.environ.get("DEXJOCO_ROOT", str(_REPO_ROOT / "third_party" / "dexjoco")))
    a("--models_root", default=os.environ.get(
        "DEXJOCO_MODELS_ROOT", str(_REPO_ROOT / "checkpoints" / "DexJoCo-Pi05")),
      help="local copy of the DexJoCo/DexJoCo-Pi05 release (JAX/LoRA SFT checkpoints)")
    a("--checkpoint_dir", default=None, help="default: <models_root>/<release dir>/<task>")
    a("--randomize_dynamics", type=int, default=0)
    a("--max_episode_steps", type=int, default=None, help="default: the simulator's own limit")
    # pi0.5 / execution
    a("--num_denoise_steps", type=int, default=10)
    a("--query_freq", type=int, default=20, help="env steps executed per pi0.5 query")
    a("--action_offset", type=int, default=0,
      help="skip the first k actions of each chunk (DexJoCo's evaluator never executes the "
           "earliest indices: they elapse during inference latency)")
    a("--action_magnitude", type=float, default=1.0, help="|noise| bound of the SAC action")
    a("--noise_rows", type=int, default=1,
      help="noise vectors the SAC action holds; row k drives pi0.5 action k, the last row is "
           "repeated over the rest of the horizon (1 = dsrl_pi0)")
    # data collection
    a("--num_envs", type=int, default=8)
    a("--num_eval_envs", type=int, default=16)
    a("--eval_episodes", type=int, default=32)
    a("--eval_seed", type=int, default=100000)
    a("--seed", type=int, default=0)
    # SAC (defaults from dsrl_pi0/examples/scripts/run_libero.sh + launch_train_sim.py)
    a("--max_steps", type=int, default=500000, help="SAC gradient steps")
    a("--batch_size", type=int, default=256)
    a("--multi_grad_step", type=int, default=20, help="SAC updates per new transition (UTD)")
    a("--start_online_updates", type=int, default=500, help="transitions before SAC updates start")
    a("--discount", type=float, default=0.999)
    a("--failure_terminal", default="floor", choices=["floor", "bootstrap"])
    a("--resize_image", type=int, default=64)
    a("--cameras", default="base,wrist", help="pi0.5 camera keys fed to SAC (comma list)")
    a("--add_states", type=int, default=1)
    a("--actor_lr", type=float, default=1e-4)
    a("--critic_lr", type=float, default=3e-4)
    a("--temp_lr", type=float, default=3e-4)
    a("--hidden_dims", type=int, nargs="+", default=[128, 128, 128])
    a("--num_qs", type=int, default=10)
    a("--target_entropy", default="auto")
    a("--encoder_type", default="small")
    a("--latent_dim", type=int, default=50)
    a("--tau", type=float, default=0.005)
    a("--critic_reduction", default="min", choices=["mean", "min"],
      help="how Q targets combine the critic ensemble: min (pessimistic, default here; on water_plant it avoided the early collapse mean showed) or mean (dsrl_pi0 default)")
    # logging / checkpoints
    a("--log_interval", type=int, default=500)
    a("--eval_interval", type=int, default=25000, help="in SAC gradient steps")
    a("--checkpoint_interval", type=int, default=25000)
    a("--eval_base_policy", type=int, default=1, help="evaluate pi0.5 with N(0, I) noise first")
    a("--base_noise_rows", type=int, default=0,
      help="base-policy eval noise: 0 = i.i.d. N(0, I) over the whole horizon (pi0.5's own "
           "sampling); K > 0 = K i.i.d. rows laid out like the SAC action (DSRL's start point)")
    a("--eval_only", type=int, default=0, help="only run the base-policy evaluation, then exit")
    a("--eval_deterministic", type=int, default=0,
      help="1: SAC policy mode at eval; 0: sample, as DSRL's perform_control_eval does")
    a("--num_videos", type=int, default=4, help="eval episodes saved as mp4 per evaluation")
    a("--output_dir", default=None)
    a("--wandb_project", default="", help="enable wandb (respects WANDB_MODE)")
    a("--exp_name", default=None)
    return p.parse_args(argv)


# Simulator episode limits (dexjoco/dexjoco/sim/envs/*.py).
SIM_EPISODE_STEPS = {
    "water_plant": 1000, "click_mouse": 1200, "hammer_nail": 1000, "fold_glasses": 1200,
    "pick_bucket": 1200, "pinch_tongs": 1000, "bimanual_assembly": 1500,
    "bimanual_hanoi": 1500, "bimanual_microwave_cook": 1100, "bimanual_photograph": 1000,
    "bimanual_unlock_ipad": 1200,
}


class MetricLogger:
    """stdout + ``metrics.jsonl`` (+ optional wandb)."""

    def __init__(self, output_dir: pathlib.Path, args, enable_wandb: bool):
        self.path = output_dir / "metrics.jsonl"
        self.wandb = None
        if enable_wandb:
            import wandb

            self.wandb = wandb.init(
                project=args.wandb_project, name=output_dir.name, config=vars(args),
                dir=str(output_dir),
            )

    def log(self, metrics: dict, step: int, echo: bool = True):
        metrics = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
        if echo:
            body = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
            print(f"[step {step}] {body}", flush=True)


def main(argv=None):
    args = parse_args(argv)

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(_REPO_ROOT))

    import jax
    import numpy as np
    from gym.spaces import Box, Dict
    from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
    from jaxrl2.data import ReplayBuffer
    from jaxrl2.utils.general_utils import add_batch_dim

    from opendexrl.obs import SACObsBuilder, policy_input
    from opendexrl.pi0_policy import FrozenPi0, default_checkpoint_dir, openpi_config_name
    from opendexrl.envs.subproc import SubprocEnvs, load_task_cfg

    # ---------------------------------------------------------------- setup
    task_cfg = load_task_cfg(dexjoco_root=args.dexjoco_root, task_suite=args.suite, task_name=args.task)
    prompt = str(task_cfg["prompt"])
    max_episode_steps = args.max_episode_steps or SIM_EPISODE_STEPS[args.task]
    exp_name = args.exp_name or f"dsrl_{args.suite}_{args.task}_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = pathlib.Path(args.output_dir or _REPO_ROOT / "logs" / exp_name)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    logger = MetricLogger(output_dir, args, enable_wandb=bool(args.wandb_project))
    print(f"[dsrl] output dir {output_dir}", flush=True)

    ckpt_dir = pathlib.Path(args.checkpoint_dir or default_checkpoint_dir(args.models_root, args.task, args.suite))
    t0 = time.time()
    pi0 = FrozenPi0(
        dexjoco_root=args.dexjoco_root, checkpoint_dir=ckpt_dir,
        config_name=openpi_config_name(args.task, args.suite),
        num_steps=args.num_denoise_steps, seed=args.seed,
    )
    print(f"[dsrl] loaded pi0.5 from {ckpt_dir} in {time.time() - t0:.0f}s "
          f"(horizon {pi0.action_horizon}, model action dim {pi0.model_action_dim})", flush=True)
    assert args.action_offset + args.query_freq <= pi0.action_horizon, (
        f"--action_offset {args.action_offset} + --query_freq {args.query_freq} exceeds the "
        f"pi0.5 action horizon {pi0.action_horizon}."
    )
    act_slice = slice(args.action_offset, args.action_offset + args.query_freq)
    for flag in ("noise_rows", "base_noise_rows"):
        value = getattr(args, flag)
        lo = 1 if flag == "noise_rows" else 0
        assert lo <= value <= pi0.action_horizon, (
            f"--{flag} must be in [{lo}, {pi0.action_horizon}] (pi0.5 action horizon), got {value}."
        )

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if task_cfg["robot_type"] == "dual_arm":
        cameras = [{"wrist": "wrist_left"}.get(c, c) for c in cameras]
    obs_builder = SACObsBuilder(
        cameras=cameras, resize=args.resize_image,
        state_stats=pi0.norm_stats()["state"], add_states=bool(args.add_states),
    )
    env_kwargs = dict(
        task_cfg=task_cfg, rand_full=args.suite == "rand_full",
        randomize_dynamics=bool(args.randomize_dynamics), image_size=224,
        pad_state_dim46=False, max_episode_steps=max_episode_steps,
    )
    rng = np.random.default_rng(args.seed)
    train_envs = SubprocEnvs([{**env_kwargs, "seed": args.seed * 1000 + i} for i in range(args.num_envs)])
    eval_envs = SubprocEnvs([{**env_kwargs, "seed": args.eval_seed + i} for i in range(args.num_eval_envs)])

    obs_space = Dict(obs_builder.observation_spaces(Box))
    action_shape = (args.noise_rows, pi0.model_action_dim)
    act_space = Box(low=-args.action_magnitude, high=args.action_magnitude, shape=action_shape, dtype=np.float32)
    sample_obs = add_batch_dim(obs_space.sample())
    sample_action = add_batch_dim(act_space.sample())
    agent = PixelSACLearner(
        args.seed, sample_obs, sample_action,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, temp_lr=args.temp_lr,
        hidden_dims=tuple(args.hidden_dims), cnn_features=(32, 32, 32, 32), cnn_strides=(2, 1, 1, 1),
        cnn_padding="VALID", latent_dim=args.latent_dim, discount=args.discount, tau=args.tau,
        critic_reduction=args.critic_reduction, dropout_rate=0.0, aug_next=True, use_bottleneck=True,
        encoder_type=args.encoder_type, encoder_norm="group", use_spatial_softmax=True,
        softmax_temperature=-1, target_entropy=args.target_entropy, num_qs=args.num_qs,
        action_magnitude=args.action_magnitude, num_cameras=len(cameras),
    )
    buffer = ReplayBuffer(obs_space, act_space, max(1024, args.max_steps // args.multi_grad_step))
    buffer.seed(args.seed)
    batch_iter = None
    step_discount = args.discount ** args.query_freq

    def repeat_noise(rows: np.ndarray) -> np.ndarray:
        """[B, K, D] noise rows -> [B, horizon, D] pi0.5 noise; the last row fills the tail."""
        tail = np.repeat(rows[:, -1:], pi0.action_horizon - rows.shape[1], axis=1)
        return np.concatenate([rows, tail], axis=1).astype(np.float32)

    def query_actions(task_obs_list, noise):
        return pi0.infer([policy_input(o, prompt) for o in task_obs_list], noise)

    # ---------------------------------------------------------------- eval
    def evaluate(step: int, base_policy: bool) -> dict:
        tag = "base" if base_policy else "dsrl"
        n_envs = len(eval_envs)
        successes, lengths, seeds_used = [], [], []
        video_budget = args.num_videos
        next_seed = args.eval_seed
        t_eval = time.time()
        while len(successes) < args.eval_episodes:
            n = min(n_envs, args.eval_episodes - len(successes))
            ids = list(range(n))
            seeds = [next_seed + i for i in ids]
            next_seed += n
            obs = eval_envs.call("reset", ids, seeds)
            done = np.zeros(n, dtype=bool)
            frames = [[o["base"]] for o in obs]
            last = [None] * n
            noise_rng = np.random.default_rng(seeds[0])
            while not done.all():
                if base_policy:
                    base_rows = args.base_noise_rows or pi0.action_horizon
                    noise = repeat_noise(noise_rng.standard_normal((n, base_rows, pi0.model_action_dim)))
                else:
                    sac_obs = obs_builder.batch(obs)
                    rows = agent.eval_actions(sac_obs) if args.eval_deterministic else agent.sample_actions(sac_obs)
                    noise = repeat_noise(np.reshape(rows, (n, *action_shape)))
                actions = query_actions(obs, noise)[:, act_slice]
                active = [i for i in ids if not done[i]]
                results = eval_envs.call(
                    "chunk_step", active,
                    [{"actions": actions[i], "return_frames": i < video_budget} for i in active],
                )
                for i, r in zip(active, results):
                    obs[i] = r["obs"]
                    if i < video_budget:
                        frames[i].extend(r["frames"])
                    if r["terminated"][-1] or r["truncated"][-1]:
                        done[i] = True
                        last[i] = r
            for i in ids:
                successes.append(bool(last[i]["success_once"]))
                lengths.append(int(last[i]["elapsed"]))
                seeds_used.append(seeds[i])
                if i < video_budget:
                    _save_video(frames[i], output_dir / "videos" / f"{tag}_step{step:07d}_seed{seeds[i]}_"
                                f"{'success' if successes[-1] else 'failure'}.mp4")
            video_budget = max(0, video_budget - n)
        metrics = {
            f"eval_{tag}/success_rate": float(np.mean(successes)),
            f"eval_{tag}/episode_len": float(np.mean(lengths)),
            f"eval_{tag}/episodes": len(successes),
            f"eval_{tag}/time_s": time.time() - t_eval,
        }
        logger.log(metrics, step)
        return metrics

    # ---------------------------------------------------------------- train
    n_envs = len(train_envs)
    obs = train_envs.call("reset", range(n_envs), [int(s) for s in rng.integers(0, 2**31 - 1, n_envs)])
    trajs = [{"obs": [], "actions": []} for _ in range(n_envs)]
    recent_success = deque(maxlen=100)
    recent_len = deque(maxlen=100)
    grad_steps = 0
    grad_budget = 0
    env_steps = 0
    episodes = 0
    next_eval = 0 if args.eval_base_policy else args.eval_interval
    next_ckpt = args.checkpoint_interval
    next_log = args.log_interval
    t_start = time.time()
    t_policy = t_env = t_sac = 0.0
    update_info = {}

    if args.eval_base_policy or args.eval_only:
        evaluate(0, base_policy=True)
        next_eval = args.eval_interval
    if args.eval_only:
        train_envs.close()
        eval_envs.close()
        return

    def finish_episode(env_id: int, final_obs, result) -> None:
        nonlocal episodes
        traj = trajs[env_id]
        traj["obs"].append(obs_builder.single(final_obs))
        q = len(traj["actions"])
        success = bool(result["success_once"])
        failure_terminal = (not success) and bool(result["terminated"][-1]) and int(result["elapsed"]) < max_episode_steps
        rewards = -np.ones(q, dtype=np.float32)
        masks = np.ones(q, dtype=np.float32)
        if success:
            rewards[-1] = 0.0
            masks[-1] = 0.0
        elif failure_terminal and args.failure_terminal == "floor":
            rewards[-1] = -1.0 / (1.0 - step_discount)
            masks[-1] = 0.0
        for t in range(q):
            buffer.insert(dict(
                observations=traj["obs"][t], next_observations=traj["obs"][t + 1],
                actions=traj["actions"][t], next_actions=traj["actions"][min(t + 1, q - 1)],
                rewards=rewards[t], masks=masks[t], discount=step_discount,
            ))
        buffer.increment_traj_counter()
        episodes += 1
        recent_success.append(success)
        recent_len.append(int(result["elapsed"]))
        trajs[env_id] = {"obs": [], "actions": []}
        logger.log({
            "rollout/success": float(success), "rollout/episode_len": int(result["elapsed"]),
            "rollout/failure_terminal": float(failure_terminal), "rollout/episodes": episodes,
            "rollout/success_rate_100": float(np.mean(recent_success)),
            "buffer/size": len(buffer),
        }, grad_steps, echo=False)
        return q

    while grad_steps < args.max_steps:
        sac_obs = obs_builder.batch(obs)
        t = time.time()
        if grad_steps == 0:
            rows = rng.standard_normal((n_envs, *action_shape)).astype(np.float32)
        else:
            rows = np.reshape(agent.sample_actions(sac_obs), (n_envs, *action_shape)).astype(np.float32)
        actions = query_actions(obs, repeat_noise(rows))[:, act_slice]
        t_policy += time.time() - t
        for i in range(n_envs):
            trajs[i]["obs"].append({k: v[i] for k, v in sac_obs.items()})
            trajs[i]["actions"].append(rows[i])
        t = time.time()
        results = train_envs.call("chunk_step", range(n_envs), list(actions))
        t_env += time.time() - t
        reset_ids = []
        for i, r in enumerate(results):
            env_steps += int(r["executed"].sum())
            obs[i] = r["obs"]
            if r["terminated"][-1] or r["truncated"][-1]:
                q = finish_episode(i, r["obs"], r)
                # As in DSRL, updates are only owed for data collected once
                # online updates have started.
                if len(buffer) > args.start_online_updates:
                    grad_budget += q * args.multi_grad_step
                reset_ids.append(i)
        if reset_ids:
            new_obs = train_envs.call("reset", reset_ids, [int(s) for s in rng.integers(0, 2**31 - 1, len(reset_ids))])
            for i, o in zip(reset_ids, new_obs):
                obs[i] = o

        if len(buffer) < args.start_online_updates or grad_budget == 0:
            continue
        if batch_iter is None:
            batch_iter = buffer.get_iterator(args.batch_size)
        t = time.time()
        while grad_budget > 0 and grad_steps < args.max_steps:
            update_info = agent.update(next(batch_iter))
            grad_steps += 1
            grad_budget -= 1
            if grad_steps >= next_log:
                info = {f"train/{k}": float(jax.device_get(v)) for k, v in update_info.items()
                        if np.ndim(jax.device_get(v)) == 0}
                elapsed = time.time() - t_start
                info.update({
                    "rollout/success_rate_100": float(np.mean(recent_success)) if recent_success else 0.0,
                    "rollout/episode_len_100": float(np.mean(recent_len)) if recent_len else 0.0,
                    "rollout/episodes": episodes, "rollout/env_steps": env_steps,
                    "buffer/size": len(buffer),
                    "buffer/mean_reward": float(buffer.data["rewards"][: len(buffer)].mean()),
                    "buffer/terminal_frac": float(1.0 - buffer.data["masks"][: len(buffer)].mean()),
                    "time/elapsed_min": elapsed / 60,
                    "time/policy_s": t_policy, "time/env_s": t_env, "time/sac_s": t_sac + time.time() - t,
                })
                logger.log(info, grad_steps)
                next_log += args.log_interval
            if grad_steps >= next_eval:
                t_sac += time.time() - t
                evaluate(grad_steps, base_policy=False)
                next_eval += args.eval_interval
                t = time.time()
            if args.checkpoint_interval > 0 and grad_steps >= next_ckpt:
                agent.save_checkpoint(str(output_dir / "checkpoints"), grad_steps, args.checkpoint_interval)
                next_ckpt += args.checkpoint_interval
        t_sac += time.time() - t

    evaluate(grad_steps, base_policy=False)
    # The periodic save may already have written this step.
    if not (output_dir / "checkpoints" / f"checkpoint{grad_steps}").exists():
        agent.save_checkpoint(str(output_dir / "checkpoints"), grad_steps, args.checkpoint_interval or 1)
    train_envs.close()
    eval_envs.close()
    print(f"[dsrl] done: {grad_steps} SAC steps, {episodes} episodes, {env_steps} env steps", flush=True)


def _save_video(frames, path: pathlib.Path, fps: int = 30) -> None:
    import imageio

    imageio.mimsave(str(path), frames, fps=fps)


if __name__ == "__main__":
    main()
