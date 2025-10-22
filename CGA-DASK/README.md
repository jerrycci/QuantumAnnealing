# 🧠 TSP Bayesian Optimization with Dask

本專案示範如何使用 **Dask** 平行化運行 **scikit-optimize (skopt)** 進行貝氏優化（Bayesian Optimization），自動尋找 TSP 問題中最佳的懲罰係數 `A`。

---

## 📦 1. 安裝環境

### ✅ 適用環境
- Ubuntu 20.04 / 22.04 或 macOS
- Python 3.8+（推薦使用 venv）
- 可多台電腦運行（分散式模式）

### 🧰 安裝步驟

#### （1）建立 Python 虛擬環境
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

python3 -m venv dask-env
source dask-env/bin/activate
```

#### （2）安裝必要套件
```bash
pip install --upgrade pip
pip install dask distributed scikit-optimize joblib numpy dimod pyqubo
```
（使用 CGA 求解器，請確保已安裝並可匯入 `compal_solver`。）

---

## ⚙️ 2. 啟動 Dask 分散式運算環境

### Step 1 — 在 Scheduler 節點（例：192.168.31.100）執行：
```bash
source dask-env/bin/activate
dask-scheduler
```
預設會監聽在 TCP 8786 連接埠。

### Step 2 — 在每台 Worker 節點上執行：
```bash
source dask-env/bin/activate
dask-worker tcp://192.168.31.100:8786
```
確保防火牆允許 8786 連線。

---

## 🚀 3. 執行分散式貝氏優化

假設：
- TSP 程式為 `tsp-demo-simple.py`
- 優化器為 `tsp-demo-optimize.py`

執行指令：
```bash
python3 tsp-demo-optimize.py --tsp ./tsp-demo-simple.py   --A-low 0.01 --A-high 0.1 --prior log-uniform   --n-initial 4 --n-iter 3 --batch-size 4 --omp-threads 4   --csv results.csv --jsonlog results.jsonl --scheduler tcp://192.168.31.100:8786
```

---

## 📈 4. 輸出結果

### 優化過程記錄
終端機中會顯示每次評估的 `A` 與 `cost`：
```
2025-10-22 15:53:47,iter1,0.08937662,10156.91171199
[BEST so far] A=0.06876831, cost=10026.10969874 (after iter 1)
```

### 結果儲存
- `results.csv`：紀錄所有實驗的 A 與 cost。
- `results.jsonl`：每筆結果以 JSON 格式紀錄。
- 結尾輸出：
  ```
  [FINAL] Best A = 0.06876831, Best cost = 10026.10969874
  ```

---

## 🧩 5. 常見錯誤與解法

| 問題 | 原因 | 解決方式 |
|------|------|-----------|
| `ValueError: min() arg is an empty sequence` | A 太小導致無效解 調整 A 範圍  |
| `tsp-demo.py exited with code 1` | 無法解碼有效路徑 | 使用安全版程式或提高 timeout |
| `Connection refused` | Worker 無法連 Scheduler | 檢查 IP 與 8786 連接埠 |
| 無結果 | timeout 太短 | 增加 `--timeout 20` |

---

## 🧠 6. 進階設定

- **平行度控制：** `--batch-size` 控制每次同時評估幾個 A。
- **隨機重現性：** 可指定 `--random-state 42`。
- **重複平均：** `--repeats N` 會對同一 A 執行多次取平均。

---

## 📚 7. 資料夾結構範例

```
.
├── tsp-demo-simple.py
├── tsp-demo-optimize.py
├── results.csv
├── results.jsonl
└── qubo_int/
    └── 20251022-150300_qubo_int.txt
```

---
