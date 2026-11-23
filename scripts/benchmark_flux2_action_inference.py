#!/usr/bin/env python3
"""Benchmark FLUX.2 ImageWAM action inference without a simulator."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("native", "geometry"),
        default="native",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default="Perform the task described by the instruction.",
    )
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = q * (len(ordered) - 1)
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(ordered[0]),
        "p50": float(percentile(0.50)),
        "p90": float(percentile(0.90)),
        "p95": float(percentile(0.95)),
        "max": float(ordered[-1]),
    }


def _configure_mode(model_cfg, mode: str) -> None:
    geometry_cfg = model_cfg.loss.geometry_transition
    with open_dict(geometry_cfg):
        if mode == "native":
            geometry_cfg.enabled = False
            return
        geometry_cfg.enabled = True


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))

    from experiments.robotwin.imagewam_policy.deploy_policy import _filter_model_cfg_for_target

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    train_cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg.model, resolve=True))
    with open_dict(model_cfg):
        model_cfg.load_text_encoder = True
        model_cfg.qwen_context_len = 128
    _configure_mode(model_cfg, args.mode)
    model_cfg = _filter_model_cfg_for_target(model_cfg)

    load_start = time.perf_counter()
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=str(device))
    model.load_checkpoint(str(args.checkpoint))
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    model_load_s = time.perf_counter() - load_start

    # Text is constant during a rollout, so cache it exactly as the deployment policy does.
    text_context, text_mask = model._prepare_flux2_infer_text(args.prompt, None, None)
    torch.cuda.synchronize(device)
    resident_allocated = torch.cuda.memory_allocated(device)
    resident_reserved = torch.cuda.memory_reserved(device)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_image = torch.rand(
        (1, 3, args.height, args.width),
        generator=generator,
        dtype=torch.float32,
    ).mul_(2.0).sub_(1.0).to(device=device, dtype=torch.bfloat16)
    proprio = torch.zeros((1, 14), device=device, dtype=torch.float32)

    def infer(profile: bool):
        return model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=args.action_horizon,
            proprio=proprio,
            context=text_context,
            context_mask=text_mask,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            rand_device="cpu",
            profile_infer_timing=profile,
        )

    for _ in range(args.warmup):
        infer(profile=False)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    latencies: list[float] = []
    timing_records: list[dict] = []
    output_checksum = None
    for _ in range(args.repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = infer(profile=True)
        torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - start)
        timing_records.append(result["timing"])
        output_checksum = float(result["action"].double().sum().item())

    segment_names = sorted(
        {name for record in timing_records for name in record.get("segments", {})}
    )
    segment_summaries = {
        name: _summary([record["segments"][name] for record in timing_records])
        for name in segment_names
    }
    action_predict_s = [record["action_predict_total_s"] for record in timing_records]
    total_profiled_s = [record["total_profiled_s"] for record in timing_records]

    payload = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "train_config": str(args.config.resolve()),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_dtype": "bfloat16",
        "input_shape": [1, 3, args.height, args.width],
        "action_horizon": args.action_horizon,
        "num_inference_steps": args.num_inference_steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cached_text_context": True,
        "model_load_s": model_load_s,
        "latency_s": _summary(latencies),
        "raw_latency_s": latencies,
        "profiled_total_s": _summary(total_profiled_s),
        "action_predict_total_s": _summary(action_predict_s),
        "segments_s": segment_summaries,
        "resident_memory_gib": {
            "allocated": resident_allocated / (1024**3),
            "reserved": resident_reserved / (1024**3),
        },
        "peak_memory_gib": {
            "allocated": torch.cuda.max_memory_allocated(device) / (1024**3),
            "reserved": torch.cuda.max_memory_reserved(device) / (1024**3),
        },
        "output_checksum": output_checksum,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
