# Foresight Forcing

**Learning Spatial Foresight for Robotic Manipulation**

Foresight Forcing is a geometry-aware World Action Model built on ImageWAM. It
learns spatial transition queries from correspondence-aligned changes in a
frozen 3D feature space, then uses those queries with the current visual
observation to generate actions.

<p align="center">
  <img src="assets/teaser.svg" alt="Foresight Forcing overview" width="100%" />
</p>

The key training signal comes from a frozen VGGT teacher applied to paired
current and future observations. At deployment, the teacher and future image
are removed: transition queries are predicted from the current observation,
language instruction, and proprioceptive state, and the action expert uses the
current visual tokens together with those queries.

**Project resources:** [project website](https://anson2333.github.io/foresight-forcing/) · [code](https://github.com/anson2333/ForesightForcing) · [model weights (Hugging Face)](https://huggingface.co/anson233/foresight-forcing)

## Results

The table reports the main simulation results from the paper. Success rates are
percentages; the RoboTwin columns report clean, randomized, and average scores.

| Benchmark | Evaluation protocol | Foresight Forcing |
| --- | --- | ---: |
| LIBERO | Four-suite average (Spatial, Object, Goal, Long) | **98.5** |
| LIBERO-Plus | Seven perturbation dimensions, overall | **84.5** |
| RoboTwin 2.0 Full | Clean / randomized / average | **94.90 / 95.12 / 95.01** |
| RoboTwin 2.0 Clean-to-Random | Clean / randomized / average | **87.84 / 67.06 / 77.45** |

## Method

Foresight Forcing adds a spatial transition pathway to the ImageWAM action
generation pipeline:

1. During training, a frozen VGGT extracts patch features from aligned current
   and future observations.
2. Their feature transition supervises spatial queries laid out on the visual
   token grid.
3. A shared updater refines the queries using current visual features,
   language, and proprioception.
4. The action expert conditions on both the current visual path and the
   predicted transition queries.

The default inference path keeps the reference visual features visible together
with the transition queries. Future observations, the VGGT teacher, and
distillation heads are training-only components.

## Installation

The release uses Python 3.11, CUDA, and `uv`. Clone the repository with the
pinned VGGT submodule:

```bash
git clone --recurse-submodules https://github.com/anson2333/ForesightForcing.git
cd ForesightForcing
git submodule update --init --recursive
uv sync --python 3.11 --extra shared
source .venv/bin/activate
cp .env.example .env.local
```

The VGGT submodule is pinned to commit
`a288dd0f14786c93483e45524328726ab7b1b4ce`. Keep this revision when comparing
results. Prepare the pinned FLUX.2 source checkout and model paths separately:

```bash
git clone https://github.com/black-forest-labs/flux2 third_party/flux2
git -C third_party/flux2 checkout 50fe5162777813d869182b139e83b10743caef15

export FLUX2_SRC=$(pwd)/third_party/flux2
export FLUX2_MODEL_PATH=/path/to/flux-2-klein-base-4b.safetensors
export FLUX2_AE_MODEL_PATH=/path/to/ae.safetensors
export FLUX2_QWEN3_MODEL_SPEC=Qwen/Qwen3-4B
```

Model weights must be downloaded separately under their respective licenses.

Set local data and output locations in the shell or `.env.local`:

```bash
export DATA_ROOT=/path/to/datasets
export MODEL_ROOT=/path/to/model/checkpoints
export OUTPUT_ROOT=$(pwd)/runs
```

For simulator evaluation, use an external RoboTwin checkout and set:

```bash
export ROBOTWIN_REPO_ROOT=/path/to/RoboTwin
```

RoboTwin is intentionally not vendored in this repository. Install its runtime
environment with:

```bash
bash scripts/setup/install_robotwin_env.sh
```

## Data and checkpoints

The training entrypoints expect the preprocessed LIBERO and RoboTwin layouts.
The dataset roots can be downloaded and arranged under `DATA_ROOT` as follows:

```bash
huggingface-cli download yuanty/LIBERO-fastwam \
  --repo-type dataset --local-dir data/libero_mujoco3.3.2

huggingface-cli download yuanty/robotwin2.0-fastwam \
  --repo-type dataset --local-dir data/robotwin2.0
(cd data/robotwin2.0 && cat robotwin2.0.tar.gz.part-* | tar -xzf -)
```

Set the matching root before training:

```bash
export DATA_ROOT=$(pwd)/data/libero_mujoco3.3.2
# For RoboTwin:
export DATA_ROOT=$(pwd)/data/robotwin2.0
export ROBOTWIN_ROOT=$DATA_ROOT/robotwin2.0
```

The released checkpoints can be downloaded from Hugging Face with:

```bash
huggingface-cli download anson233/foresight-forcing \
  --local-dir checkpoints/foresight-forcing
```

## Training

The standard FLUX.2 ImageWAM launcher uses the 4B base configuration and the
geometry-aware transition pathway by default.

### LIBERO

```bash
export DATA_ROOT=$(pwd)/data/libero_mujoco3.3.2

GPU_PER_NODE=8 \
TASK_TYPE=libero \
FLUX2_VARIANT=4b \
PRECOMPUTE_QWEN3_CACHE=true \
bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

### RoboTwin

```bash
export DATA_ROOT=$(pwd)/data/robotwin2.0
export ROBOTWIN_ROOT=$DATA_ROOT/robotwin2.0

GPU_PER_NODE=8 \
TASK_TYPE=robotwin \
FLUX2_VARIANT=4b \
PRECOMPUTE_QWEN3_CACHE=true \
bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

Useful environment overrides include `ZERO_STAGE`, `ACTION_INIT`, and
`QWEN_CACHE_DIR`. If `ACTION_INIT` is not supplied, the launcher creates the
compatible ActionDiT initialization weights automatically.

## Evaluation

Point the evaluator at a checkpoint and the statistics file used by its dataset
processor:

```bash
export CKPT_PATH=/path/to/checkpoint.pt
export DATASET_STATS_PATH=/path/to/dataset_stats.json
export NUM_GPUS=8
```

### LIBERO

```bash
FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero.sh
```

### LIBERO-Plus

```bash
FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero_plus.sh
```

### RoboTwin 2.0

```bash
export ROBOTWIN_REPO_ROOT=/path/to/RoboTwin
FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_robotwin.sh
```

The RoboTwin evaluator covers the 50-task Full and Clean-to-Random suites. For
the 16D action contract, use the dedicated `ee16` launcher and do not mix its
checkpoints with the ordinary 14D entrypoints.

## Repository layout

```text
configs/                  Hydra model, data, and task configurations
experiments/              LIBERO and RoboTwin evaluation integrations
scripts/                  Training, evaluation, and setup entrypoints
src/imagewam/             ImageWAM and Foresight Forcing implementation
third_party/vggt/         Pinned VGGT submodule
```

## Verification

Run the lightweight checks after installing the environment:

```bash
python3 -m compileall -q src scripts experiments
```

## License

See [LICENSE](LICENSE).
