# count_time_script

統計指定月份內所有 `.txt.debug` log 檔案的 `run_time` 總和。

## 檔案結構

Script 預設會去讀取**同層的 `log/` 資料夾**：

```
project/
├── count_time_script.py
└── log/
    ├── qubo_2026-03-31-13-11-44.txt.debug
    ├── qubo_2026-03-31-13-16-21.txt.debug
    └── ...
```

### Log 檔案格式

- **檔名**：`{name}_{YYYY-MM-DD-HH-mm-SS}.txt.debug`
- **第一行**：包含 `run_time=<秒數>` 的參數列，例如：
  ```
  [CGA] run_time=10 timeout_sec=21 batch=10 ...
  ```

> `run_time` 支援整數與浮點數。

---

## 使用方式

### 基本用法

統計當年度指定月份的總 run_time：

```bash
python count_time_script.py --month=3
```

### 完整參數

| 參數 | 必填 | 說明 | 預設值 |
|------|------|------|--------|
| `--month` | ✅ | 要統計的月份（1–12） | — |
| `--year` | | 要統計的年份 | 當年度 |
| `--dir` | | log 資料夾路徑 | script 同層的 `log/` |
| `--verbose` / `-v` | | 顯示每個檔案的詳細時間 | — |

### 範例

```bash
# 統計 2026 年 3 月（年份預設為當年）
python count_time_script.py --month=3

# 指定年份
python count_time_script.py --month=3 --year=2025

# 指定 log 資料夾路徑
python count_time_script.py --month=3 --dir=/path/to/logs

# 顯示每個檔案的詳細資訊
python count_time_script.py --month=3 --verbose
```

---

## 輸出範例

```
統計目標：2026 年 03 月
資料夾：/home/frank/QA/quantix_ga/log
--------------------------------------------------
符合檔案數：12 個
成功解析：  12 個
--------------------------------------------------
總 run_time：720 秒  (12m 0s)
```

加上 `--verbose`：

```
統計目標：2026 年 03 月
資料夾：/home/frank/QA/quantix_ga/log
--------------------------------------------------
  qubo_2026-03-31-13-11-44.txt.debug  →  run_time = 10s
  qubo_2026-03-31-13-16-21.txt.debug  →  run_time = 60s
  ...
符合檔案數：12 個
成功解析：  12 個
--------------------------------------------------
總 run_time：720 秒  (12m 0s)
```

---

## 需求

- Python 3.6+
- 無需安裝第三方套件