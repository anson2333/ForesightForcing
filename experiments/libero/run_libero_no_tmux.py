#!/usr/bin/env python3
"""Run LIBERO workers directly without tmux.

The legacy LIBERO scheduler creates one tmux pane per chunk and can leave
orphaned sessions when a MuJoCo worker hangs.  This manager uses ordinary
process groups, bounded chunk timeouts, and a small retry/split path instead.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def _parse_ids(value: str) -> list[int]:
    ids = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not ids:
        raise ValueError("--gpu-ids must contain at least one id")
    return ids


def _parse_suites(value: str) -> list[str]:
    value = value.strip().strip("[]")
    return [x.strip().strip("'\"") for x in value.split(",") if x.strip()]


def _make_tasks(path: Path, suites: list[str], ratio: float | None, seed: int) -> list[tuple[str, int]]:
    from libero.libero import benchmark

    import random

    rng = random.Random(seed)
    tasks: list[tuple[str, int]] = []
    with path.open("w", encoding="utf-8") as f:
        for suite_name in suites:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            ids = list(range(int(suite.n_tasks)))
            if ratio is not None and ratio < 1.0:
                n = max(1, int((len(ids) * ratio) + 0.999999))
                ids = sorted(rng.sample(ids, n))
            for task_id in ids:
                f.write(f"{suite_name},{task_id}\n")
                tasks.append((suite_name, task_id))
    return tasks


def _read_tasks(path: Path) -> list[tuple[str, int]]:
    tasks = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid task line {line_no}: {line!r}")
        tasks.append((parts[0], int(parts[1])))
    if not tasks:
        raise ValueError(f"No tasks in {path}")
    return tasks


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _result_files(output_dir: Path) -> list[Path]:
    return list(output_dir.glob("libero_*/gpu*_task*_results.json"))


def _completed_tasks(output_dir: Path) -> set[tuple[str, int]]:
    completed = set()
    for path in _result_files(output_dir):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            completed.add((str(record["task_suite"]), int(record["task_id"])))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return completed


def _tagged_output_dir(raw: Path, ckpt: Path, root: Path) -> Path:
    """Match eval_libero_single's checkpoint-tagged output convention."""
    parts = ckpt.resolve().parts
    if "runs" in parts and parts.index("runs") + 2 < len(parts):
        idx = parts.index("runs")
        tag = f"{parts[idx + 1]}_{parts[idx + 2]}"
    else:
        tag = ckpt.stem
    eval_root = (root / "evaluate_results").resolve()
    raw = raw.resolve()
    try:
        rel = raw.relative_to(eval_root)
    except ValueError:
        return raw if raw.parent.name == tag else raw.parent / tag / raw.name
    if len(rel.parts) >= 2 and rel.parts[1] == tag:
        return raw
    family = rel.parts[0] if rel.parts else "libero"
    run_name = rel.parts[-1] if rel.parts else raw.name
    return eval_root / family / tag / run_name


def _write_summary(output_dir: Path, task_count: int, failed: list[dict]) -> None:
    records = []
    for path in _result_files(output_dir):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_suite[str(record.get("task_suite", "unknown"))].append(record)
    summary = {
        "task_count_requested": task_count,
        "task_count_results": len(records),
        "failed_chunks": failed,
        "successes": sum(int(r.get("successes", 0)) for r in records),
        "episodes": sum(int(r.get("total_episodes", 0)) for r in records),
        "by_suite": {},
    }
    for suite, rows in sorted(by_suite.items()):
        successes = sum(int(r.get("successes", 0)) for r in rows)
        episodes = sum(int(r.get("total_episodes", 0)) for r in rows)
        summary["by_suite"][suite] = {
            "tasks": len(rows),
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes if episodes else None,
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["suite", "tasks", "successes", "episodes", "success_rate"])
        for suite, values in summary["by_suite"].items():
            writer.writerow([suite, values["tasks"], values["successes"], values["episodes"], values["success_rate"]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--task-suite-names", default="libero_10,libero_goal,libero_spatial,libero_object")
    parser.add_argument("--task-sample-ratio", type=float, default=1.0)
    parser.add_argument("--task-sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config-name", default="sim_libero")
    parser.add_argument("--task-choice", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--gpu-ids", default="1,2,3,4,5,6,7")
    parser.add_argument("--max-tasks-per-gpu", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--chunk-timeout-sec", type=int, default=3600)
    parser.add_argument("--single-timeout-sec", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--launch-stagger-sec", type=float, default=0.0)
    parser.add_argument("--manager-lock-file", type=Path, default=Path("/tmp/imagewam_libero_no_tmux.lock"))
    parser.add_argument("--allow-concurrent-manager", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.overrides and args.overrides[0] == "--":
        args.overrides = args.overrides[1:]

    manager_lock = None
    if not args.allow_concurrent_manager:
        args.manager_lock_file.parent.mkdir(parents=True, exist_ok=True)
        manager_lock = args.manager_lock_file.open("a+", encoding="utf-8")
        try:
            fcntl.flock(manager_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            manager_lock.seek(0)
            owner = manager_lock.read().strip() or "unknown"
            parser.error(
                f"another LIBERO manager holds {args.manager_lock_file} (pid={owner}); "
                "wait for it to finish or explicitly pass --allow-concurrent-manager"
            )
        manager_lock.seek(0)
        manager_lock.truncate()
        manager_lock.write(str(os.getpid()))
        manager_lock.flush()

    root = Path(__file__).resolve().parents[2]
    out = _tagged_output_dir(args.output_dir, Path(args.ckpt), root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "task_logs").mkdir(exist_ok=True)
    (out / "task_chunks").mkdir(exist_ok=True)
    (out / "task_status").mkdir(exist_ok=True)

    task_file = args.task_file.resolve() if args.task_file else out / "tasks.txt"
    if args.task_file is None or not task_file.exists():
        ratio = None if args.task_sample_ratio >= 1.0 else args.task_sample_ratio
        tasks = _make_tasks(task_file, _parse_suites(args.task_suite_names), ratio, args.task_sample_seed)
    else:
        tasks = _read_tasks(task_file)
    requested_task_count = len(tasks)
    if args.resume:
        completed = _completed_tasks(out)
        tasks = [task for task in tasks if task not in completed]
        print(
            f"resume completed={len(completed)} remaining={len(tasks)} "
            f"requested={requested_task_count}",
            flush=True,
        )
        active_task_file = out / "remaining_tasks.txt"
    else:
        active_task_file = out / "tasks.txt"
    active_task_file.write_text("\n".join(f"{s},{i}" for s, i in tasks) + "\n", encoding="utf-8")

    queue: list[tuple[int, list[tuple[str, int]], int]] = []
    next_id = 0
    for start in range(0, len(tasks), max(1, args.chunk_size)):
        queue.append((next_id, tasks[start : start + max(1, args.chunk_size)], 0))
        next_id += 1
    slots = [gpu for gpu in _parse_ids(args.gpu_ids) for _ in range(max(1, args.max_tasks_per_gpu))]
    active: dict[int, dict] = {}
    failed: list[dict] = []
    worker_script = root / "experiments/libero/eval_libero_single.py"
    stop_requested = False

    def _request_stop(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"received signal {signum}; stopping workers", flush=True)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    print(f"tasks={len(tasks)} chunks={len(queue)} slots={len(slots)} output={out}", flush=True)

    def launch(slot: int, item: tuple[int, list[tuple[str, int]], int]) -> None:
        chunk_id, chunk_tasks, retry = item
        chunk_file = out / "task_chunks" / f"chunk_{chunk_id:06d}.txt"
        chunk_file.write_text("\n".join(f"{s},{i}" for s, i in chunk_tasks) + "\n", encoding="utf-8")
        log_path = out / "task_logs" / f"chunk_{chunk_id:06d}_slot_{slot}_retry_{retry}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(slots[slot])
        # Pin the backend in every worker.  OSMesa is the default on hosts
        # where it is installed; GPU-only hosts can select EGL explicitly.
        render_backend = os.environ.get("LIBERO_MUJOCO_GL", os.environ.get("MUJOCO_GL", "osmesa"))
        pyopengl_backend = os.environ.get("LIBERO_PYOPENGL_PLATFORM", os.environ.get("PYOPENGL_PLATFORM", render_backend))
        env["MUJOCO_GL"] = render_backend
        env["PYOPENGL_PLATFORM"] = pyopengl_backend
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable, str(worker_script),
            f"--config-name={args.config_name}",
            f"task={args.task_choice}",
            f"ckpt={args.ckpt}",
            f"gpu_id={slot}",
            f"EVALUATION.output_dir={out}",
            f"EVALUATION.task_chunk_file={chunk_file}",
            f"EVALUATION.num_trials={args.num_trials}",
            *args.overrides,
        ]
        log = log_path.open("wb")
        proc = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[slot] = {"proc": proc, "item": item, "start": time.monotonic(), "log": log, "log_path": log_path}
        print(f"launch chunk={chunk_id} tasks={len(chunk_tasks)} slot={slot} gpu={slots[slot]} pid={proc.pid}", flush=True)

    while (queue or active) and not stop_requested:
        for slot in range(len(slots)):
            if slot not in active and queue:
                launch(slot, queue.pop(0))
                if args.launch_stagger_sec > 0:
                    time.sleep(args.launch_stagger_sec)
        time.sleep(2)
        for slot, state in list(active.items()):
            proc = state["proc"]
            elapsed = time.monotonic() - state["start"]
            item = state["item"]
            timeout = args.single_timeout_sec if len(item[1]) == 1 else args.chunk_timeout_sec
            if proc.poll() is None and elapsed <= timeout:
                continue
            state["log"].close()
            if proc.poll() is None:
                _kill_group(proc)
                rc = "timeout"
            else:
                rc = proc.returncode
            chunk_id, chunk_tasks, retry = item
            result_count = 0
            for suite, task_id in chunk_tasks:
                if (out / suite / f"gpu{slot}_task{task_id}_results.json").exists():
                    result_count += 1
            print(f"finish chunk={chunk_id} slot={slot} rc={rc} elapsed={elapsed:.0f}s results={result_count}/{len(chunk_tasks)}", flush=True)
            if rc != 0 or result_count < len(chunk_tasks):
                remaining = [(s, i) for s, i in chunk_tasks if not (out / s / f"gpu{slot}_task{i}_results.json").exists()]
                if retry < args.max_retries and len(remaining) > 1:
                    for task in reversed(remaining):
                        next_id += 1
                        queue.insert(0, (next_id, [task], retry + 1))
                elif retry < args.max_retries and remaining:
                    queue.insert(0, (next_id, remaining, retry + 1))
                    next_id += 1
                else:
                    failed.append({"chunk": chunk_id, "slot": slot, "rc": rc, "tasks": remaining, "log": str(state["log_path"])})
            del active[slot]

    if stop_requested:
        for slot, state in list(active.items()):
            proc = state["proc"]
            _kill_group(proc)
            state["log"].close()
            del active[slot]
        print("stopped before summary; worker process groups reaped", flush=True)
        return 143

    _write_summary(out, requested_task_count, failed)
    print(
        f"completed results={len(_result_files(out))}/{requested_task_count} "
        f"failed_chunks={len(failed)}",
        flush=True,
    )
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
