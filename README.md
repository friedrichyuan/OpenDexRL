# OpenDexRL: Reinforcement Fine-tuning for High-DoF Dexterous Manipulation

OpenDexRL fine-tunes vision-language-action policies for dexterous, high-DoF manipulation with online reinforcement learning. It applies **DSRL** (Diffusion Steering via Reinforcement Learning, [Wagenmaker et al., 2025](https://arxiv.org/abs/2506.15799)) to the **π0.5** policies released with the [DexJoCo](https://github.com/brave-eai/dexjoco) benchmark: π0.5 stays frozen, and a small pixel-based SAC agent learns the initial noise of its flow-matching action sampler. The DexJoCo SFT checkpoints (JAX, LoRA) are used as released; no weight conversion and no gradients through the 3B-parameter model.

## Method

On top of the reference DSRL loop ([dsrl_pi0](https://github.com/nakamotoo/dsrl_pi0)):

- **Parallel rollouts.** `N` DexJoCo simulators run in subprocesses; π0.5 is queried once per step for the whole batch.
- **Pessimistic critic.** SAC targets use the minimum over a 10-head Q ensemble (`--critic_reduction min`). With the mean used by DSRL, Q-values on water_plant were about 10× too optimistic (−3.5 against returns of −30 to −45).
- **Failure-aware rewards.** Sparse −1 per decision; episodes that end in an early failure (e.g. spraying outside the pot) get the worst-case value instead of being bootstrapped, so the agent cannot profit from failing fast.
- **Per-step noise** (`--noise_rows K`). The SAC action can carry one noise vector for each of the first `K` executed actions instead of one vector repeated over the whole chunk (`K=1`, DSRL's default).

## Results

DexJoCo `rand_obj`, single-arm tasks, 32 evaluation episodes per checkpoint on fixed seeds, one H20 GPU, default settings (`--noise_rows 1`, `--critic_reduction min`). *SFT* is the released π0.5 policy sampled with standard-normal noise; *OpenDexRL* is the mean over the last five checkpoints (400k–500k SAC steps).

| Task | SFT success | OpenDexRL success | Episode length (SFT → OpenDexRL) |
|---|---|---|---|
| click_mouse | 62.5% | **90.6%** | 611 → 416 |
| pinch_tongs | 21.9% | **38.8%** | 915 → 820 |
| hammer_nail | 96.9% | 91.2% | 274 → 301 |
| water_plant* | 90.6% | 83.8% | 297 → 296 |

\*Stopped at 375k steps (window 275k–375k). On hammer_nail and water_plant the SFT policy is already near the ceiling and RL did not improve it. Evaluations fluctuate by 10–20 points between checkpoints (best single checkpoints: click_mouse 96.9%, pinch_tongs 59.4%), so select checkpoints by evaluation rather than taking the last one.

Fewer SAC updates per transition (`--multi_grad_step 5` instead of 20, 125k updates for the same amount of environment data) stabilized click_mouse: **98.8%** over the last five checkpoints (100k–125k updates, 56.3% at the start), with episodes shortened to 224 steps. pinch_tongs stayed at 35.6% and keeps oscillating.

Per-step noise raises the starting point of the search before any training:

| Task | 1 noise vector per chunk | 20 per-step vectors |
|---|---|---|
| click_mouse | 56.3% | **81.3%** |
| pinch_tongs | 18.8% | **40.6%** |

## Installation

```bash
git clone https://github.com/friedrichyuan/OpenDexRL.git
cd OpenDexRL
git submodule update --init      # not --recursive: dsrl_pi0's own submodules are not needed
bash scripts/install.sh          # uv venv in .venv (Python 3.11, JAX 0.5.3, MuJoCo 3.4.0)
```

Download the DexJoCo π0.5 checkpoints (the full release is ~230 GB; fetch only the tasks you train):

```bash
hf download DexJoCo/DexJoCo-Pi05 --local-dir checkpoints/DexJoCo-Pi05 --include "pi05_dexjoco_ckpt/pinch_tongs/*"
```

Paths are set in `scripts/env.sh` and can be overridden through environment variables (`DEXJOCO_MODELS_ROOT`, `VENV_DIR`, `OPENPI_DATA_HOME`, ...).

## Usage

```bash
GPU=0 bash scripts/smoke_test.sh pinch_tongs                    # end-to-end check, ~5 min
GPU=0 bash scripts/train.sh pinch_tongs rand_obj                 # one task, one GPU
GPU=0 bash scripts/train.sh pinch_tongs rand_obj --noise_rows 20 # per-step noise
bash scripts/train_multi.sh rand_obj click_mouse pinch_tongs     # one task per visible GPU
GPU=1 bash scripts/queue.sh rand_obj fold_glasses pick_bucket    # start each run when the GPU has room
```

A run writes `logs/<run>/`: `train.log`, `metrics.jsonl`, `videos/` (evaluation rollouts) and `checkpoints/` (SAC actor and critic). All options: `python -m opendexrl.train -h`. Each run needs ~25 GB of GPU memory and ~32 GB of host RAM with the default 8 training and 16 evaluation environments; several runs can share a GPU.

Tasks: `water_plant`, `click_mouse`, `hammer_nail`, `fold_glasses`, `pick_bucket`, `pinch_tongs` (single-arm) and `bimanual_{assembly,hanoi,microwave_cook,photograph,unlock_ipad}` (dual-arm). Dual-arm tasks run end-to-end but have not been trained to convergence yet.

## Repository layout

```
opendexrl/
  train.py            DSRL training loop (SAC over pi0.5 noise, evaluation, checkpoints)
  pi0_policy.py       frozen DexJoCo pi0.5 with batched, noise-conditioned inference
  obs.py              SAC observations (64x64 camera stack + normalized proprioception)
  envs/               DexJoCo task wrapper and subprocess vector env
scripts/              install, train, train_multi, queue, smoke_test, tune_stability, watch_runs
tests/                simulator and pi0.5-inference checks
third_party/
  dexjoco/            DexJoCo benchmark and its OpenPI fork (submodule)
  dsrl_pi0/           DSRL's jaxrl2 SAC implementation (submodule)
```

## Acknowledgements

OpenDexRL builds on [DSRL](https://github.com/nakamotoo/dsrl_pi0), [DexJoCo](https://github.com/brave-eai/dexjoco) and [OpenPI](https://github.com/Physical-Intelligence/openpi).

```bibtex
@article{wagenmaker2025steering,
  title   = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  author  = {Wagenmaker, Andrew and Nakamoto, Mitsuhiko and Zhang, Yunchu and Park, Seohong and Yagoub, Waleed and Nagabandi, Anusha and Gupta, Abhishek and Levine, Sergey},
  journal = {Conference on Robot Learning (CoRL)},
  year    = {2025}
}
@misc{wang2026dexjoco,
  title         = {DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo},
  author        = {Wang, Hanwen and Zhao, Weizhi and Wang, Xiangyu and Huang, Siyuan and Lin, He and Zheng, Boyuan and Xu, Rongtao and Wang, Gang and Mu, Yao and Wang, He and Fan, Lue and Li, Hongsheng and Zhang, Zhaoxiang and Tan, Tieniu},
  year          = {2026},
  eprint        = {2605.16257},
  archivePrefix = {arXiv}
}
```
