#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tsp-demo-optimize.py — Bayesian optimization over A using skopt + Dask
# Usage examples:
#   Local (no Dask): python3 tsp-demo-optimize.py --tsp ./tsp-demo.py
#   With Dask:       python3 tsp-demo-optimize.py --tsp ./tsp-demo.py --scheduler tcp://192.168.32.109:8786
# Advanced:
#   python3 tsp-demo-optimize.py --tsp ./tsp-demo.py --A-low 0.5 --A-high 3.0 --prior log-uniform #       --n-initial 12 --n-iter 30 --batch-size 6 --repeats 3 --omp-threads 4 --csv results.csv

import argparse
import os
import re
import sys
import time
import json
import statistics
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from skopt import Optimizer
from skopt.space import Real

try:
    from dask.distributed import Client, as_completed
    DASK_AVAILABLE = True
except Exception:
    DASK_AVAILABLE = False

COST_PATTERNS = [
    re.compile(r"cost\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"\b([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")  # fallback: last number in the output
]

def parse_cost(stdout: str) -> float:
    # Try multiple patterns to be robust against different prints
    for pat in COST_PATTERNS:
        m = pat.search(stdout)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    raise ValueError("Failed to parse cost from program output. "
                     "Ensure tsp-demo.py prints 'cost=...'. Output snippet:\n"
                     + "\n".join(stdout.splitlines()[-10:]))

def run_tsp_once(tsp_path: str, A: float, timeout: int, omp_threads: Optional[int]) -> Tuple[float, str]:
    env = os.environ.copy()
    if omp_threads is not None and omp_threads > 0:
        env["OMP_NUM_THREADS"] = str(omp_threads)
    cmd = [sys.executable, tsp_path, f"--A={A}"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"tsp run timed out after {timeout}s for A={A}") from e
    if p.returncode != 0:
        raise RuntimeError(f"tsp-demo.py exited with code {p.returncode} for A={A}. "
                           f"stderr:\n{p.stderr}")
    cost = parse_cost(p.stdout)
    return cost, p.stdout

def evaluate_A_mean(tsp_path: str, A: float, repeats: int, timeout: int, omp_threads: Optional[int]) -> float:
    vals = []
    for i in range(repeats):
        cost, _ = run_tsp_once(tsp_path, A, timeout, omp_threads)
        vals.append(cost)
    return statistics.mean(vals) if repeats > 1 else vals[0]

def main():
    ap = argparse.ArgumentParser(description="Bayesian optimization of A for tsp-demo.py using skopt + Dask")
    ap.add_argument("--tsp", required=True, help="Path to tsp-demo.py (must accept --A=<float> and print 'cost=...')")
    ap.add_argument("--scheduler", default=None, help="Dask scheduler address, e.g., tcp://192.168.32.109:8786. Omit for local.")
    ap.add_argument("--A-low", type=float, default=0.5, help="Lower bound for A")
    ap.add_argument("--A-high", type=float, default=3.0, help="Upper bound for A")
    ap.add_argument("--prior", choices=["uniform", "log-uniform"], default="log-uniform", help="Prior for A sampling")
    ap.add_argument("--n-initial", type=int, default=10, help="Initial random evaluations")
    ap.add_argument("--n-iter", type=int, default=20, help="BO iterations after initial phase")
    ap.add_argument("--batch-size", type=int, default=4, help="Parallel evaluations per ask() batch")
    ap.add_argument("--repeats", type=int, default=1, help="Repeat each A multiple times and average")
    ap.add_argument("--timeout", type=int, default=600, help="Per evaluation timeout (seconds)")
    ap.add_argument("--omp-threads", type=int, default=None, help="Set OMP_NUM_THREADS for tsp runs")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed for skopt")
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV path to append results")
    ap.add_argument("--jsonlog", type=str, default=None, help="Optional JSONL path to append detailed logs")
    args = ap.parse_args()

    tsp_path = Path(args.tsp)
    if not tsp_path.exists():
        print(f"[ERROR] tsp script not found: {tsp_path}", file=sys.stderr)
        sys.exit(1)

    # Setup Dask client (optional)
    client = None
    if args.scheduler:
        if not DASK_AVAILABLE:
            print("[ERROR] Dask is not available but scheduler was provided.", file=sys.stderr)
            sys.exit(1)
        client = Client(args.scheduler)
        print(f"[INFO] Connected to Dask scheduler at {args.scheduler}")
    else:
        # local execution path (no Dask needed)
        print("[INFO] Running locally without Dask scheduler.")

    # Define skopt optimizer
    space = [Real(args.A_low, args.A_high, name="A", prior=args.prior)]
    opt = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI", random_state=args.random_state)

    # Logging helpers
    def log_result(A_val: float, cost_val: float, phase: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')},{phase},{A_val:.8f},{cost_val:.8f}"
        print(line)
        if args.csv:
            header_needed = not Path(args.csv).exists()
            with open(args.csv, "a", encoding="utf-8") as f:
                if header_needed:
                    f.write("timestamp,phase,A,cost\n")
                f.write(line + "\n")
        if args.jsonlog:
            with open(args.jsonlog, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "phase": phase, "A": A_val, "cost": cost_val}) + "\n")

    def submit_jobs(points: List[List[float]]):
        # Submit a list of A points for evaluation; returns futures or results list.
        if client is None:
            # local synchronous execution
            results = []
            for x in points:
                A_val = float(x[0])
                c = evaluate_A_mean(str(tsp_path), A_val, args.repeats, args.timeout, args.omp_threads)
                results.append((A_val, c))
            return results
        else:
            # Dask parallel execution
            futures = [client.submit(evaluate_A_mean, str(tsp_path), float(x[0]), args.repeats, args.timeout, args.omp_threads)
                       for x in points]
            return futures

    # Initial random evaluations (possibly batched)
    n_init = max(1, args.n_initial)
    batch = max(1, args.batch_size)
    remaining = n_init
    while remaining > 0:
        take = min(batch, remaining)
        xs = opt.ask(n_points=take)
        jobs = submit_jobs(xs)
        if client is None:
            # local results
            for (A_val, c), x in zip(jobs, xs):
                opt.tell([A_val], c)
                log_result(A_val, c, phase="init")
        else:
            for fut, x in zip(as_completed(jobs), xs):
                c = fut.result()
                A_val = float(x[0])
                opt.tell([A_val], c)
                log_result(A_val, c, phase="init")
        remaining -= take

    # Iterative BO phase
    n_iter = max(0, args.n_iter)
    for it in range(n_iter):
        xs = opt.ask(n_points=batch)
        jobs = submit_jobs(xs)
        if client is None:
            for (A_val, c), x in zip(jobs, xs):
                opt.tell([A_val], c)
                log_result(A_val, c, phase=f"iter{it+1}")
        else:
            for fut, x in zip(as_completed(jobs), xs):
                c = fut.result()
                A_val = float(x[0])
                opt.tell([A_val], c)
                log_result(A_val, c, phase=f"iter{it+1}")

        # Report best so far each outer iteration
        best_idx = int(min(range(len(opt.yi)), key=lambda i: opt.yi[i]))
        best_A = opt.Xi[best_idx][0]
        best_cost = opt.yi[best_idx]
        print(f"[BEST so far] A={best_A:.8f}, cost={best_cost:.8f} (after iter {it+1})")

    # Final best
    best_idx = int(min(range(len(opt.yi)), key=lambda i: opt.yi[i]))
    best_A = opt.Xi[best_idx][0]
    best_cost = opt.yi[best_idx]
    print(f"[FINAL] Best A = {best_A:.8f}, Best cost = {best_cost:.8f}")
    # Also print plain values for easy scraping
    print(f"best_A={best_A}")
    print(f"best_cost={best_cost}")

if __name__ == "__main__":
    main()
