#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Optimized version of tsp-demo.py
# Supports command-line arguments:
#   --A <float>
#   --ga_timeout <seconds>
#
# Designed to be Ray-safe and BO-safe.

import argparse
import itertools
import numpy as np
import os, math, time
from datetime import datetime

import dimod
from pyqubo import Array, Constraint

import sys

print("[RAY DEBUG] python:", sys.executable)

try:
    from compal_solver import compal_solver as solver
    print("[RAY DEBUG] compal_solver:", solver)
except Exception as e:
    print("[RAY DEBUG] compal_solver IMPORT FAILED:", e)
    solver = None


# --------------------------
# Utility: Haversine (meters)
# --------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# --------------------------------
# Main solver wrapper for CGA
# --------------------------------

def solve_tsp_compal_solver(coords, timeout=5, A=1.2):
    """Solve TSP using compal_solver.Quantix_GA with tunable A."""

    if solver is None:
        print("[RAY DEBUG] ERROR: compal_solver Not Available!")
        return None, 1e12

    try:
        n = len(coords)
        x = Array.create("x", (n, n), "BINARY")

        # TSP constraints
        time_const = sum((sum(x[i, j] for j in range(n)) - 1)**2 for i in range(n))
        city_const = sum((sum(x[i, j] for i in range(n)) - 1)**2 for j in range(n))

        def euclidean(i, j):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            return ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5

        # Route cost term
        distance = 0.0
        for i in range(n):
            for j in range(n):
                d_ij = euclidean(i, j)
                for k in range(n):
                    distance += d_ij * x[k, i] * x[(k+1) % n, j]

        H = distance + A * (time_const + city_const)
        model = H.compile()
        qubo, offset = model.to_qubo(index_label=True)
        variables = model.variables
        nvars = len(variables)

        # ----- Determine integer scale -----
        abs_key = max(qubo, key=lambda y: abs(float(qubo[y])))
        abs_value = abs(float(qubo[abs_key]))
        order_upper = 13 - len(str(round(abs_value)))

        len_key = max(qubo, key=lambda y: len(str(abs(float(qubo[y]))).split(".")[-1]))
        len_value = len(str(abs(float(qubo[len_key]))).split(".")[-1])
        N = min(len_value, order_upper)
        if N < 0:
            N = 0

        # ----- Dump QUBO integer file -----
        os.makedirs("qubo_int", exist_ok=True)
        file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        qubo_int_path = f"qubo_int/{file_name}_qubo_int.txt"
        with open(qubo_int_path, "w") as f:
            print(nvars, offset, file=f)
            for (i, j), value in qubo.items():
                print(i, j, round(float(value) * (10**N)), file=f)

        # ----- Run GA -----
        ga = solver.Quantix_GA(qubo_int_path)
        result, energy, count, timeout_flag = ga.run(
            batch_factor=10,
            main_factor=0.2,
            run_time=timeout
        )

        # ----- Handle no solution -----
        result = np.array(result)
        if result.size == 0:
            print(f"[Warning] GA returned empty result for A={A}. Returning max cost.")
            return None, 1e12

        # Normalize shape
        if result.ndim == 1:
            inferred_count = result.size // nvars
            result = result.reshape((inferred_count, nvars))
            count = inferred_count
        elif result.ndim == 2:
            count = result.shape[0]

        # Build Q matrix for energy evaluation
        Q = np.zeros((nvars, nvars))
        for (i, j), val in qubo.items():
            Q[i, j] = float(val)

        sample_list, energy_list = [], []
        for c in range(count):
            vec = np.asarray(result[c, :], dtype=np.uint8)
            sample = {variables[i]: int(vec[i]) for i in range(nvars)}
            sample_list.append(sample)

            b = vec.reshape((nvars, 1))
            e_val = float((b.T @ Q @ b)[0, 0])
            energy_list.append(e_val)

        sampleset = dimod.SampleSet.from_samples(
            sample_list, dimod.BINARY, energy_list
        )
        decoded = model.decode_sampleset(sampleset)

        # ----- No valid decoded solutions -----
        if not decoded:
            print(f"[Warning] No valid decoded samples for A={A}. Returning max cost.")
            return None, 1e12

        # ----- Decode best path -----
        def decode_to_path(sol):
            path = [None] * n
            for i in range(n):
                for j in range(n):
                    if sol.array("x", (i, j)) == 1:
                        path[i] = j
            return path

        best = min(decoded, key=lambda d: d.energy)
        path = decode_to_path(best)

        # Fix missing assignments if needed
        if None in path or len(set(path)) != n:
            assigned = [p for p in path if p is not None]
            missing = [i for i in range(n) if i not in assigned]
            path = [p if p is not None else missing.pop(0) for p in path]

        # Final route length (meters)
        def dist_m(i, j):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            return haversine_m(lat1, lon1, lat2, lon2)

        cost = sum(dist_m(path[i], path[(i+1) % n]) for i in range(n))
        return path, cost

    except Exception as e:
        print("[ERROR] Exception inside solve_tsp_compal_solver:", e)
        return None, 1e12


# ---------------------
# Main entry point
# ---------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP demo with tunable A parameter")
    parser.add_argument("--A", type=float, default=1.2)
    parser.add_argument("--ga_timeout", type=float, default=5.0,
                        help="GA runtime (seconds) passed to Quantix_GA.run(run_time=...)")
    args = parser.parse_args()

    print(f"[RAY DEBUG] Using GA timeout = {args.ga_timeout}")

    coords = [
        (25.08285, 121.59045),
        (25.07845, 121.57532),
        (25.08723, 121.59112),
        (25.08276, 121.57789),
        (25.08098, 121.58234),
        (25.08412, 121.58623),
        (25.08512, 121.59034),
        (25.08645, 121.58876),
        (25.07987, 121.56123),
        (25.07567, 121.56189),
        (25.07345, 121.56987),
        (25.07789, 121.56543),
        (25.06865, 121.58695),
        (25.08239, 121.55933),
        (25.08022, 121.59382),
        (25.08302, 121.56471)
    ]

    path, cost = solve_tsp_compal_solver(coords, timeout=args.ga_timeout, A=args.A)

    # Final safety
    if cost is None or math.isnan(cost) or math.isinf(cost):
        cost = 1e12

    print(f"cost={cost}")
