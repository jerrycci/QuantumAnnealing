#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 100% fixed tsp-demo-optimize-ray.py
#
# - 使用 Ray + skopt 對外部 TSP solver (tsp-demo-simple.py) 的參數 A 做 BO
# - TSP script 需支援：
#       --A=<float>
#       --ga_timeout=<float>   (optional, 若你有加這個參數)
#       並印出一行：cost=<float>
#
# 修正點：
#   1. Ray worker 不再用自己的 os.environ，而是用「driver 啟動當下的 env」跑 subprocess
#   2. 支援 --ga_timeout，轉傳給 tsp-demo-simple.py
#   3. 絕對路徑安全：在 Ray working_dir 中執行拷貝過去的 tsp-demo-simple.py
#   4. 強健的 cost parsing + inf / nan sanitization
#

import argparse
import os
import sys
import re
import time
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import List, Optional, Dict

import ray
from skopt import Optimizer
from skopt.space import Real


# -------- Cost Parsing --------

COST_PATTERNS = [
    re.compile(r"cost\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"\b([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"),
]


def parse_cost_from_text(text: str) -> float:
    """從 stdout 文字中抓出 cost 數值。"""
    for pat in COST_PATTERNS:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    tail = "\n".join(text.splitlines()[-10:])
    raise ValueError(f"Failed to parse cost from text. Tail:\n{tail}")


def sanitize_cost(cost: float) -> float:
    """確保 y 是有限的浮點數；inf / nan / 超大值一律變成 1e12 penalty。"""
    if cost is None:
        return 1e12
    try:
        c = float(cost)
    except Exception:
        return 1e12
    if math.isnan(c) or math.isinf(c):
        return 1e12
    if abs(c) > 1e20:
        return 1e12
    return c


# -------- Ray Remote Worker --------

@ray.remote (num_gpus=1)
def evaluate_A_mean_remote(
    tsp_rel_path: str,
    A: float,
    repeats: int,
    timeout: int,
    ga_timeout: float,
    base_env: Dict[str, str],
) -> float:
    """
    在 Ray worker 裡多次執行 tsp solver，取平均 cost。
    這裡不再使用 worker 的 os.environ，而是使用 driver 傳進來的 base_env。
    """

    import os
    import sys
    import subprocess
    import statistics

    # Ray 會把 driver 的 working_dir 打包到 worker 的 runtime_resources/working_dir_files 裡
    # 在那裡，tsp_rel_path 是相對 driver cwd 的路徑，例如 "./tsp-demo-simple.py"
    # 我們在 worker 這邊把它 resolve 成 executable path
    workdir = os.getcwd()  # Ray 的 working_dir
    exec_file = os.path.join(workdir, tsp_rel_path.lstrip("./"))
    exec_file = os.path.abspath(exec_file)

    print(f"[RAY DEBUG] A={A}")
    print(f"[RAY DEBUG] workdir={workdir}")
    print(f"[RAY DEBUG] exec_file={exec_file}")
    print(f"[RAY DEBUG] repeats={repeats}, timeout={timeout}, ga_timeout={ga_timeout}")

    def run_once() -> float:
        # 使用 driver capture 的環境變數，而不是 worker 的 os.environ
        env = dict(base_env)

        # 不主動去改 OMP / BLAS threads，盡量跟 CLI 環境一致
        # （如果你要手動控制 threads，可以改成下面這樣）
        # if omp_threads is not None:
        #     env["OMP_NUM_THREADS"] = str(omp_threads)

        cmd = [
            sys.executable,
            exec_file,
            f"--A={A}",
            f"--ga_timeout={ga_timeout}",
        ]
        print(f"[RAY DEBUG] Running cmd: {cmd}")

        try:
            p = subprocess.run(
                cmd,
                cwd=os.path.dirname(exec_file),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"[RAY DEBUG] subprocess.TimeoutExpired for A={A}")
            return float("inf")

        print(f"[RAY DEBUG] Return code: {p.returncode}")
        print(f"[RAY DEBUG] STDOUT:\n{p.stdout}")
        print(f"[RAY DEBUG] STDERR:\n{p.stderr}")

        if p.returncode != 0:
            print(f"[RAY DEBUG] Non-zero exit code for A={A}, treating as inf.")
            return float("inf")

        try:
            cost = parse_cost_from_text(p.stdout)
            print(f"[RAY DEBUG] Parsed cost for A={A}: {cost}")
        except Exception as e:
            print(f"[RAY DEBUG] Failed to parse cost for A={A}: {e}")
            cost = float("inf")

        return cost

    vals = []
    for _ in range(max(1, repeats)):
        vals.append(run_once())

    vals_sanitized = [sanitize_cost(v) for v in vals]
    mean_cost = statistics.mean(vals_sanitized)
    print(f"[RAY DEBUG] Final mean cost for A={A}: {mean_cost}")
    return mean_cost


# -------- Driver / Main --------

def main():
    ap = argparse.ArgumentParser(
        description="Bayesian optimization of A for TSP solver via Ray + skopt"
    )
    ap.add_argument("--tsp", required=True, help="Path to tsp solver script (relative or absolute)")
    ap.add_argument("--ray-address", default=None, help="Ray address (e.g., auto)")
    ap.add_argument("--A-low", type=float, default=0.5)
    ap.add_argument("--A-high", type=float, default=3.0)
    ap.add_argument(
        "--prior",
        choices=["uniform", "log-uniform"],
        default="log-uniform",
        help="Prior for sampling A",
    )
    ap.add_argument("--n-initial", type=int, default=10, help="Initial random evaluations")
    ap.add_argument("--n-iter", type=int, default=20, help="Number of BO iterations")
    ap.add_argument("--batch-size", type=int, default=4, help="Parallel evaluations in each batch")
    ap.add_argument("--repeats", type=int, default=1, help="Repeat each A multiple times")
    ap.add_argument("--timeout", type=int, default=600, help="Subprocess timeout (seconds)")
    ap.add_argument("--ga_timeout", type=float, default=15.0, help="Inner GA timeout (seconds)")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--csv", type=str, default=None, help="CSV log file (append)")
    ap.add_argument("--jsonlog", type=str, default=None, help="JSONL log file (append)")

    args = ap.parse_args()

    # Driver cwd & env（這份 env 是我們要傳給 Ray worker subprocess 用的）
    driver_cwd = os.getcwd()
    print(f"[INFO] Driver cwd: {driver_cwd}")

    # 這裡 capture 的 base_env 就是你在 Terminal（CLI）跑時的環境
    base_env = dict(os.environ)

    # 檢查 tsp script 存在（從 driver 角度）
    tsp_path = Path(args.tsp)
    if not tsp_path.exists():
        print(f"[ERROR] tsp script not found: {tsp_path}", file=sys.stderr)
        sys.exit(1)

    # 如果是絕對路徑，轉成相對 driver_cwd 的路徑，讓 Ray working_dir 打包得比較乾淨
    tsp_rel = os.path.relpath(str(tsp_path), start=driver_cwd)

    # ---- Ray init ----
    if args.ray_address:
        ray.init(address=args.ray_address, runtime_env={"working_dir": driver_cwd})
        print(f"[INFO] Connected to Ray at {args.ray_address}, working_dir={driver_cwd}")
    else:
        ray.init(runtime_env={"working_dir": driver_cwd})
        print(f"[INFO] Started local Ray with working_dir={driver_cwd}")

    # ---- skopt Optimizer ----
    space = [Real(args.A_low, args.A_high, name="A", prior=args.prior)]
    opt = Optimizer(
        dimensions=space,
        base_estimator="GP",
        acq_func="EI",
        random_state=args.random_state,
    )

    def log_result(A_val: float, cost_val: float, phase: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts},{phase},{A_val:.8f},{cost_val:.8f}"
        print(line)

        if args.csv:
            csv_path = Path(args.csv)
            header_needed = not csv_path.exists()
            with csv_path.open("a", encoding="utf-8") as f:
                if header_needed:
                    f.write("timestamp,phase,A,cost\n")
                f.write(line + "\n")

        if args.jsonlog:
            json_path = Path(args.jsonlog)
            with json_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"ts": time.time(), "phase": phase, "A": A_val, "cost": cost_val}
                    )
                    + "\n"
                )

    def submit_points(points: List[List[float]]):
        futures = [
            evaluate_A_mean_remote.remote(
                tsp_rel,          # 相對路徑，在 Ray working_dir 裡會是同樣結構
                float(x[0]),
                args.repeats,
                args.timeout,
                args.ga_timeout,
                base_env,        # *關鍵*：把 driver 的 env 傳進 worker
            )
            for x in points
        ]
        costs = ray.get(futures)
        return [sanitize_cost(c) for c in costs]

    # ---- Initial random evaluations ----
    remaining = max(1, args.n_initial)
    batch = max(1, args.batch_size)

    while remaining > 0:
        take = min(batch, remaining)
        xs = opt.ask(n_points=take)
        costs = submit_points(xs)
        for x, c in zip(xs, costs):
            A_val = float(x[0])
            opt.tell([A_val], c)
            log_result(A_val, c, "init")
        remaining -= take

    # ---- BO iterations ----
    for it in range(max(0, args.n_iter)):
        xs = opt.ask(n_points=batch)
        costs = submit_points(xs)
        for x, c in zip(xs, costs):
            A_val = float(x[0])
            opt.tell([A_val], c)
            log_result(A_val, c, f"iter{it+1}")

        best_idx = int(min(range(len(opt.yi)), key=lambda i: opt.yi[i]))
        best_A = opt.Xi[best_idx][0]
        best_cost = opt.yi[best_idx]
        print(f"[BEST so far] A={best_A:.8f}, cost={best_cost:.8f} (after iter {it+1})")

    # ---- Final best ----
    best_idx = int(min(range(len(opt.yi)), key=lambda i: opt.yi[i]))
    best_A = opt.Xi[best_idx][0]
    best_cost = opt.yi[best_idx]
    print(f"[FINAL] Best A = {best_A:.8f}, Best cost = {best_cost:.8f}")
    print(f"best_A={best_A}")
    print(f"best_cost={best_cost}")


if __name__ == "__main__":
    main()
