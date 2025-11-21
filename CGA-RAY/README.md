使用 Ray 進行多節點 GPU TSP 參數優化

本專案使用 Ray 分散式運算 在多台 GPU 主機上並行搜尋最佳參數 A，用於執行你自訂的 CGA TSP 求解器（tsp-demo-simple.py）。
每台機器上會運行 1 個 Ray worker，並 一次只接受 1 個 CGA 任務，讓你的 solver 可以使用該機器 所有 GPU（例如 A100 / V100 / RTX 系列）。

📦 檔案

tsp-demo-optimize-ray.py
多節點 Ray 優化主程式。

tsp-demo-simple.py
CGA TSP solver，每個 worker 都必須有一份。

🖥️ 系統需求

所有節點（包含 head / worker）需要：

Ubuntu 20.04 / 22.04 / 24.04
Python 3.10
Ray 2.51.1（所有 node 版本需一致）
NVIDIA 驅動 + CUDA（符合你的 CGA 需求）

🚀 建立虛擬環境（建議名稱：ray）

在每台 node 執行：

python3 -m venv ray
source ray/bin/activate
pip install -U pip
pip install ray[default]==2.51.1


確認版本一致：

ray --version

🛰️ 多台機器佈署方式

假設：

Head Node IP：192.168.31.100

Worker 1 IP：192.168.31.166


1️⃣ Head Node 啟動 Ray Head
ray stop
ray start --head --port=6379

2️⃣ Worker Node 加入 Head

在每台 Worker：

ray stop
ray start --address="192.168.31.100:6379"


確認成功：

ray status


你應該會看到 2 個 active worker。

📌 每個 Worker 需要一份 TSP Solver

CGA 會用不同型號 GPU，因此 不能打包進 runtime_env。
所以每個 node 上都必須：

tsp-demo-simple.py (相同版本)
compal_solver (依據node機器上的GPU型號安裝對應的CGA)

一律放在當前執行目錄：
EX:
~/CGA-RAY/

🧠 執行多節點 TSP 參數優化

在 Head Node 執行：

python3 tsp-demo-optimize-ray.py \
  --tsp ./tsp-demo-simple.py \
  --ray-address auto \
  --A-low 0.5 --A-high 3.0 \
  --n-initial 5 \
  --n-iter 20 \
  --ga_timeout 5 

🎉 效果：

Head 會同時派給 worker node (以及Head Node)

每個 node 只有 1 個 worker + 独立資源標籤 →
每個 node 同時間只會跑 1 個 CGA 任務

Compal_Solver 可吃到全機所有 GPU

📊 記錄輸出

加上：

--csv results.csv
--jsonlog results.jsonl


產生優化紀錄。


