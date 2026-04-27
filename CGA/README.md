## Compal GPU Annealer (CGA) Usage & Installation

The **CGA** folder contains example programs demonstrating the implementation of the Compal GPU Annealer across different API versions.

### Example Programs
* **Example 1: Traveling Salesperson Problem**
    * `tsp-demo.py`: Demonstrates usage of the **CGA 1.0 API**.
* **Example 2: Max-Cut Problem**
    * `maxcut_sample.py`: Demonstrates usage of the **CGA 2.0 API**, including support for the `init_spin` variable.

---

### Installation Guide (CGA 2.0)
The CGA 2.0 package is located in the `./CGA2.0-release` folder. Follow these steps to upgrade your environment:

> **Environment requirement:** CGA 2.0 should be installed in a **Python 3.10** environment.

If needed, create and activate a Python 3.10 environment before installation:

```bash
conda create -n cga20 python=3.10
conda activate cga20
```

#### Step 1: Extract the Package
Download and unzip the CGA 2.0 package file.
```bash
7z x cga2.0-release.7z -p"password"
```

#### Step 2: Remove Previous Version
Uninstall the existing version of the solver to avoid conflicts.
```bash
pip uninstall compal_solver-nstc
```

#### Step 3: Install CGA 2.0
Install the newly extracted wheel file.
```bash
pip install ./compal_solver_nstc-2.0.1-py3-none-any.whl
```

---

### Supported GPU Architectures
This library is compiled with support for the following NVIDIA GPU architectures and specific hardware models:

| Compute Capability | Architecture | Notable GPU Models Supported |
| :--- | :--- | :--- |
| **sm_61** | **Volta** | RTX 1080ti |
| **sm_70** | **Volta** | Tesla V100, Titan V |
| **sm_80** | **Ampere (DC)** | A100 (SXM4/PCIe) |
| **sm_86** | **Ampere (Consumer)** | RTX 3090, 3080, 3070, A6000, A40 |
| **sm_90** | **Hopper** | H100, H200, H800 |
| **sm_100** | **Blackwell (DC)** | B100, B200, GB200 Superchip |
| **sm_120** | **Blackwell (Consumer)** | RTX 5090, RTX 5080 |


# 📦 Compal Solver – CGA Python Package

Compal Solver 提供 GPU-based QUBO 求解器，支援兩種 API：

* **CGA 1.0** → `Quantix_GA`（基於 lattice file）
* **CGA 2.0** → `CGA_Solver`（直接輸入 QUBO dict）


# 🧠 Overview

| Version | Class        | Input            | 特點             |
| ------- | ------------ | ---------------- | -------------- |
| CGA 1.0 | `Quantix_GA` | lattice file     | 與舊版 backend 相容 |
| CGA 2.0 | `CGA_Solver` | Python QUBO dict | 更彈性、直接         |

---

# 🔹 CGA 1.0 — `Quantix_GA`

## 📌 用途

使用 **lattice file** 作為輸入的 QUBO solver。

---

## 📥 Initialization

```python
from compal_solver import Quantix_GA

solver = Quantix_GA(lattice="qubo.txt")
```

---

## ▶️ Run

```python
result, energy, count, timeout_flag = solver.run(
    init_spin=None,
    batch_factor=10.0,
    main_factor=0.1,
    energy_stop=-9223372036854775808,
    run_time=10,
    debug_info=0,
    N=0,
    check_flag=0,
    coefficient=64,
    num_results=100
)
```

---

## 📤 Return Values

| 參數             | 說明            |
| -------------- | ------------- |
| `result`       | 解 (bit array) |
| `energy`       | 對應 energy     |
| `count`        | 解的數量          |
| `timeout_flag` | 是否 timeout    |

---

## 📄 Lattice File Format

```
<bin_size> <offset>
i j value
i j value
...
```

---

# 🔹 CGA 2.0 — `CGA_Solver`

## 📌 用途

直接使用 Python dict 表示 QUBO（推薦使用）

---

## 📥 Initialization

```python
from compal_solver import CGA_Solver

Q = {
    (0, 0): -1,
    (1, 1): -1,
    (0, 1): 2
}

variables = ["x0", "x1"]

solver = CGA_Solver(
    quadric=Q,
    offset=0.0,
    variables=variables
)
```

---

## ▶️ Run

```python
sampleset, count, timeout_flag, overflow_flag = solver.run(
    init_spin=None,
    batch_factor=10.0,
    main_factor=0.1,
    energy_stop=-9223372036854775808,
    run_time=10,
    debug_info=0,
    N=0,
    check_flag=0,
    coefficient=64,
    num_results=100
)
```

---

## 📤 Return Values

| 參數              | 說明                |
| --------------- | ----------------- |
| `sampleset`     | `dimod.SampleSet` |
| `count`         | 解的數量              |
| `timeout_flag`  | 是否 timeout        |
| `overflow_flag` | 是否數值 overflow     |

---

## 📊 SampleSet 使用方式

```python
print(sampleset)

best = sampleset.first
print(best.sample)
print(best.energy)
```

---

# ⚙️ Common Parameters

| 參數             | 說明                    |
| -------------- | --------------------- |
| `batch_factor` | 搜尋 batch 強度           |
| `main_factor`  | 主演算法強度                |
| `energy_stop`  | 提前停止條件                |
| `run_time`     | 執行時間（秒）               |
| `debug_info`   | debug 等級              |
| `N`            | scaling（10^N）         |
| `check_flag`   | overflow 檢查           |
| `coefficient`  | backend（16 / 32 / 64） |
| `num_results`  | 回傳解數量                 |

---


## 2️⃣ Timeout Behavior

* solver 會自動等待 backend 完成
* `timeout_flag = 1` 表示超時但仍安全結束

---

## 3️⃣ Overflow Protection

若 QUBO scaling 超過限制：

```python
overflow_flag = 1
```

---

## 4️⃣ On-Prem Usage Time Statistics

`on-prem/` 裡提供 `count_time_script.py`，可用來統計指定月份內所有 `.txt.debug` log 檔案第一行的 `run_time` 總和，方便彙整 CGA on-prem 使用時間。

```bash
cd on-prem
python count_time_script.py --month=3
```

常用參數：

| 參數 | 說明 |
| ---- | ---- |
| `--month` | 要統計的月份（1-12，必填） |
| `--year` | 要統計的年份，未指定時使用當年度 |
| `--dir` | log 資料夾路徑，未指定時讀取 script 同層的 `log/` |
| `--verbose` / `-v` | 顯示每個檔案的詳細 run_time |

更多說明請參考 [`on-prem/README.md`](./on-prem/README.md)。

---

# 🆚 CGA1.0 vs CGA2.0

| Feature | CGA1.0 | CGA2.0      |
| ------- | ------ | ----------- |
| Input   | file   | Python dict |
| 易用性     | ❌      | ✅           |
| 效能      | 相同     | 相同          |
| 推薦      | legacy | ⭐⭐⭐⭐⭐       |

---

# ✅ Quick Example（推薦）

```python
from compal_solver import compal_solver as solver

Q = {
    (0, 0): -1,
    (1, 1): -1,
    (0, 1): 2
}

cga = solver.CGA_Solver(Q, 0.0, ["x0", "x1"])

sampleset, count, timeout_flag, overflow_flag = cga.run()

print(sampleset.first)
```


---

## ❌ overflow_flag = 1

👉 降低：

```python
N=0
```

---

# 📬 Contact

Compal GPU Annealer Team
Jerry Chen

---
