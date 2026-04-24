import numpy as np
from glob import glob
import time
from compal_solver import compal_solver as solver


# ================================================================
# 1. Load Gset instance + build QUBO matrix
# ================================================================
def maxcut_qubo_matrix(path="./", file="G39"):
    """
    Load Gset and build QUBO matrix Q for MaxCut:
        Q_ii = -sum_j w_ij
        Q_ij = 2*w_ij   (i < j)
    """
    data_list = sorted(glob(f"{path}/{file}*"))
    if not data_list:
        raise FileNotFoundError("No Gset file found")

    data = data_list[0]
    lines = open(data).read().strip().split("\n")

    n = int(lines[0].split()[0])
    wmat = np.zeros((n, n), dtype=float)

    for line in lines[1:]:
        i, j, w = line.split()
        u = int(i) - 1
        v = int(j) - 1
        weight = float(w)

        wmat[u, v] += weight
        wmat[v, u] += weight

    Q = np.zeros((n, n), dtype=float)

    for i in range(n):
        Q[i, i] = -np.sum(wmat[i])
        for j in range(i + 1, n):
            if wmat[i, j] != 0:
                Q[i, j] = 2 * wmat[i, j]
                Q[j, i] = Q[i, j]

    return Q, wmat


# ================================================================
# 2. Convert NumPy Q → compal QUBO dict
# ================================================================
def qubo_matrix_to_dict(Q):
    n = Q.shape[0]
    Qdict = {}
    for i in range(n):
        for j in range(i, n):
            if Q[i, j] != 0:
                Qdict[(i, j)] = float(Q[i, j])
    return Qdict


# ================================================================
# 3. Wrapper to run compal GA solver directly from Q matrix
# ================================================================
class QUBO_Solver_from_matrix:
    def __init__(self, Q):
        self.Q = Q
        self.n = Q.shape[0]
        self.qubo = qubo_matrix_to_dict(Q)
        self.variables = [f"x[{i}]" for i in range(self.n)]
        self.offset = 0.0
        self.ga = solver.CGA_Solver(self.qubo, self.offset, self.variables)

    def run(self, **kwargs):
        return self.ga.run(**kwargs)


# ================================================================
# 4. Convert GA sample → bitstring array
# ================================================================
def sample_to_array(sample, n):
    """ sample['x[0]'], 'x[1]' ... → numpy array """
    x = np.zeros(n, dtype=int)
    for i in range(n):
        x[i] = sample[f"x[{i}]"]
    return x


# ================================================================
# 5. Compute QUBO energy x^T Q x
# ================================================================
def qubo_energy(Q, x):
    """x^T Q x using upper-triangle only, consistent with qubo_matrix_to_dict."""
    return float(x @ np.triu(Q) @ x)

# ================================================================
# 6. True MaxCut cut value
# ================================================================
def maxcut_cut_value(wmat, x):
    cut = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            if wmat[i, j] != 0 and x[i] != x[j]:
                cut += wmat[i, j]         # w_ij (usually -1)
    return cut


# ================================================================
# 7. MAIN EXAMPLE
# ================================================================
if __name__ == "__main__":
    FILE = "G39"
    PATH = "./"

    print(f"Loading Gset instance {FILE}...")
    Q, wmat = maxcut_qubo_matrix(PATH, FILE)
    n = Q.shape[0]
    print("Variables:", n)

    # GA parameters (your recommended defaults)
    if n < 5000:
        batch = 10
        step = 0.2
        runtime = 10
    else:
        batch = 5
        step = 0.16
        runtime = 60

    print("\nStarting compal GA solver...")
    solver_obj = QUBO_Solver_from_matrix(Q)

    t0 = time.time()
    sampleset, count, timeout_flag, overflow_flag = solver_obj.run(
        init_spin=f"{PATH}/{FILE}_init2408.txt",
        batch_factor=batch,
        main_factor=step,
        run_time=runtime,
        debug_info=0,
        num_results=1000
    )
    t1 = time.time()

    print("\n=== GA Solver Output ===")

    if not sampleset:
        print("Solver timeout.")
        exit()

    print(f"Time cost: {t1 - t0:.3f} sec")
    print("Raw sample:", sampleset)

    record = sampleset.first
    sample = record.sample
    x = np.array([sample[f"x[{i}]"] for i in range(n)], dtype=int)

    print("Energy from SampleSet =", record.energy)

    E = qubo_energy(Q, x)
    print("Computed QUBO energy x^T Q x =", E)

    cut = maxcut_cut_value(wmat, x)
    print("True MaxCut cut value =", cut)
    
