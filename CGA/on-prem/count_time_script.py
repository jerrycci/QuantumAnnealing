#!/usr/bin/env python3
"""
統計指定月份內所有 .debug 檔案的 run_time 總和。

檔案命名格式: {name}_{YYYY-MM-DD-HH-mm-SS}_txt.debug
第一行格式:   [CGA] run_time=<秒數> ...

使用方式:
    python count_time_script.py --month=3
    python count_time_script.py --month=3 --year=2026
    python count_time_script.py --month=3 --dir=/path/to/logs
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="統計指定月份的 .debug 檔案 run_time 總和"
    )
    parser.add_argument(
        "--month", type=int, required=True, choices=range(1, 13),
        metavar="MONTH", help="要統計的月份 (1-12)"
    )
    parser.add_argument(
        "--year", type=int, default=datetime.now().year,
        help=f"要統計的年份 (預設: {datetime.now().year})"
    )
    default_log_dir = Path(__file__).parent / "log"
    parser.add_argument(
        "--dir", type=str, default=str(default_log_dir),
        help="log 資料夾路徑 (預設: script 同層的 log 資料夾)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="顯示每個檔案的詳細資訊"
    )
    return parser.parse_args()


# 從檔名解析日期，格式: {name}_{YYYY-MM-DD-HH-mm-SS}_txt.debug
FILENAME_DATE_PATTERN = re.compile(
    r'_(\d{4})-(\d{2})-(\d{2})-\d{2}-\d{2}-\d{2}\.txt\.debug$'
)

# 從第一行解析 run_time
RUN_TIME_PATTERN = re.compile(r'\brun_time=(\d+(?:\.\d+)?)')


def parse_date_from_filename(filename: str):
    """從檔名取得 (year, month, day)，若格式不符回傳 None。"""
    m = FILENAME_DATE_PATTERN.search(filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def parse_run_time(filepath: Path):
    """讀取檔案第一行，取得 run_time 值（秒）。失敗回傳 None。"""
    try:
        with filepath.open("r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        m = RUN_TIME_PATTERN.search(first_line)
        if m:
            return float(m.group(1))
    except OSError as e:
        print(f"  [警告] 無法讀取 {filepath.name}: {e}", file=sys.stderr)
    return None


def format_duration(total_seconds: float) -> str:
    """將秒數格式化為易讀字串，例如 3h 25m 10s。"""
    total_seconds = int(total_seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def main():
    args = parse_args()
    log_dir = Path(args.dir)

    if not log_dir.is_dir():
        print(f"錯誤：找不到目錄 '{log_dir}'", file=sys.stderr)
        sys.exit(1)

    target_year = args.year
    target_month = args.month

    print(f"統計目標：{target_year} 年 {target_month:02d} 月")
    print(f"資料夾：{log_dir.resolve()}")
    print("-" * 50)

    debug_files = sorted(log_dir.glob("*.txt.debug"))
    if not debug_files:
        print("找不到任何 .debug 檔案。")
        sys.exit(0)

    matched_files = []
    skipped_files = []

    for filepath in debug_files:
        date = parse_date_from_filename(filepath.name)
        if date is None:
            skipped_files.append(filepath.name)
            continue
        year, month, _ = date
        if year == target_year and month == target_month:
            matched_files.append(filepath)

    if not matched_files:
        print(f"在 {target_year}/{target_month:02d} 找不到符合的檔案。")
        if skipped_files:
            print(f"\n（已略過 {len(skipped_files)} 個格式不符的檔案）")
        sys.exit(0)

    total_seconds = 0.0
    file_count = 0
    failed_files = []

    for filepath in matched_files:
        run_time = parse_run_time(filepath)
        if run_time is None:
            failed_files.append(filepath.name)
            if args.verbose:
                print(f"  [跳過] {filepath.name}（找不到 run_time）")
            continue
        total_seconds += run_time
        file_count += 1
        if args.verbose:
            print(f"  {filepath.name}  →  run_time = {run_time:.0f}s")

    print(f"符合檔案數：{len(matched_files)} 個")
    print(f"成功解析：  {file_count} 個")
    if failed_files:
        print(f"解析失敗：  {len(failed_files)} 個")
        for name in failed_files:
            print(f"    - {name}")
    if skipped_files:
        print(f"格式不符（略過）：{len(skipped_files)} 個")

    print("-" * 50)
    print(f"總 run_time：{total_seconds:.0f} 秒  ({format_duration(total_seconds)})")


if __name__ == "__main__":
    main()