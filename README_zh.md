# Foresight Forcing

Foresight Forcing 是一个面向 ImageWAM 的几何感知 transition-query 方法，
默认推理路径为 `transition_and_ref`：保留 reference image 条件，同时加入
由 reference image 和文本生成的 transition queries。

## 发布版入口

仓库只保留三个 benchmark 入口：

- LIBERO：`configs/task/libero.yaml`
- LIBERO-Plus：`configs/task/libero_plus.yaml`
- RoboTwin：`configs/task/robotwin.yaml`

4B/9B 模型配置位于 `configs/model/`，数据配置位于 `configs/data/`。
RoboTwin 的 16D 预训练动作空间可以通过 `configs/data/robotwin_ee16.yaml`
显式选择，普通 RoboTwin 仍使用 14D 动作空间。

## 安装

```bash
uv sync --python 3.11 --extra shared
source .venv/bin/activate
cp .env.example .env.local
```

在 `.env.local` 或 shell 中设置本地数据、模型和 FLUX.2 源码路径：

```bash
export DATA_ROOT=/path/to/datasets
export MODEL_ROOT=/path/to/checkpoints
export FLUX2_SRC=/path/to/flux2
export FLUX2_MODEL_PATH=/path/to/flux-2-klein-base-4b.safetensors
export FLUX2_AE_MODEL_PATH=/path/to/ae.safetensors
export FLUX2_QWEN3_MODEL_SPEC=Qwen/Qwen3-4B
```

## 训练

LIBERO：

```bash
TASK_TYPE=libero FLUX2_VARIANT=4b GPU_PER_NODE=8 \
  bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

RoboTwin：

```bash
TASK_TYPE=robotwin FLUX2_VARIANT=4b GPU_PER_NODE=8 \
  bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

没有提供 `ACTION_INIT` 时，脚本会自动生成对应的 ActionDiT 初始化权重。

## 评测

先设置：

```bash
export CKPT_PATH=/path/to/checkpoint.pt
export DATASET_STATS_PATH=/path/to/dataset_stats.json
```

然后运行：

```bash
FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero.sh
FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero_plus.sh
FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_robotwin.sh
```

RoboTwin 的 16D checkpoint 使用：

```bash
bash scripts/flux2/run_eval_flux2_robotwin_ee16.sh
```

## Geometry transition

4B 默认配置：

```yaml
enabled: true
mode: bottleneck
objective: geometry_forcing
inference_action_condition: transition_and_ref
projector_mid_channels: 1024
```

checkpoint 加载同时兼容 projector mid channels 为 `256` 和 `1024` 的权重。
`mode: aux` 仍保留为训练阶段的可选路径，但不是默认部署路径。

## 许可证和上游项目

使用时请同时遵守 ImageWAM、FLUX.2、VGGT、RoboTwin、LIBERO 和 LIBERO-Plus
等上游项目的许可证与引用要求。
