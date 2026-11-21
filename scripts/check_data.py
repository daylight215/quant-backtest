# -*- coding: utf-8 -*-
"""数据自检脚本（任务1 交付前跑一遍）

用法：
    1) 把本文件和「全部清洗后数据.csv」放在同一个文件夹，双击/命令行运行：
           python check_data.py
    2) 或者直接把 csv 拖进命令行：
           python check_data.py "C:\\路径\\全部清洗后数据.csv"

检查 5 项，全部通过才算合格：
    ① 文件能读、字段齐全
    ② 只含真实交易日（周末 0 行）
    ③ 每个标的行数正常（约 2459 行）
    ④ 抽查价格正确（茅台 2018-01-02 ≈ 578.82，沪深300 ≈ 4087.40）
    ⑤ 没有涨跌停不可能的波动（单日 |涨跌| 不得超过 11%）

最后会打印「合格 / 不合格」并给出原因，退出码 0 = 合格，1 = 不合格。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REQUIRED = ["trade_date", "code", "name", "open", "high", "low", "close", "volume"]
EXPECT_ROWS = 2459          # 每个标的的交易日数（2016-10-10 ~ 2026-11-20）
ROW_TOLERANCE = 3
EXPECT_PRICE = {"600519": ("2018-01-02", 578.82), "000300": ("2018-01-02", 4087.40)}
MAX_DAILY_MOVE = 0.11       # A 股涨跌停 10%，留 1% 余量
CODES = ["600519", "600036", "000858", "000300"]


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"读不了文件（编码问题）：{path}")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # code 可能是 600519.0 / "858" / 858，统一成 6 位字符串
    code = df["code"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["code"] = code.str.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        path = Path(argv[0].strip('"'))
    else:
        path = Path(__file__).resolve().parent / "全部清洗后数据.csv"
        if not path.exists():
            print("没找到「全部清洗后数据.csv」。")
            print("请把它放到本文件同一目录，或把文件拖到本窗口后回车：")
            raw = input("文件路径：").strip().strip('"')
            if not raw:
                return 1
            path = Path(raw)

    print("=" * 60)
    print("数据自检")
    print(f"文件：{path}")
    if not path.exists():
        print("❌ 文件不存在")
        return 1
    print(f"大小：{path.stat().st_size / 1024:.0f} KB")

    df = normalize(read_csv(path))
    print(f"形状：{df.shape}")
    print("=" * 60)

    results: list[tuple[str, bool, str]] = []

    # ① 字段
    missing = [c for c in REQUIRED if c not in df.columns]
    results.append(("字段齐全", not missing,
                    "缺少 " + ",".join(missing) if missing else "8 个必需字段都在"))

    # ② 周末
    weekend = int((df["trade_date"].dt.dayofweek >= 5).sum())
    results.append(("只含真实交易日", weekend == 0, f"周末行数 {weekend}（应为 0）"))

    # ③ 行数
    rows = df.groupby("code").size().to_dict()
    rows_ok = all(abs(rows.get(c, 0) - EXPECT_ROWS) <= ROW_TOLERANCE for c in CODES)
    results.append(("各标的行数", rows_ok,
                    "，".join(f"{c}={rows.get(c, 0)}" for c in CODES)
                    + f"（应约 {EXPECT_ROWS}）"))

    # ④ 抽查价格
    price_msgs, price_ok = [], True
    for code, (date, expect) in EXPECT_PRICE.items():
        row = df[(df["code"] == code) & (df["trade_date"] == pd.Timestamp(date))]
        got = float(row["close"].iloc[0]) if len(row) else float("nan")
        ok = abs(got - expect) < 1.0
        price_ok &= ok
        price_msgs.append(f"{code} {date}: {got:.2f}（应 {expect}）")
    results.append(("抽查价格", price_ok, "；".join(price_msgs)))

    # ⑤ 涨跌停可行性
    limit_msgs, limit_ok = [], True
    for code in CODES:
        s = df[df["code"] == code].sort_values("trade_date")["close"]
        r = s.pct_change().dropna()
        bad = int((r.abs() > MAX_DAILY_MOVE).sum())
        limit_ok &= bad == 0
        limit_msgs.append(f"{code} {bad} 天（最大 {r.abs().max():.1%}）")
    results.append(("无异常涨跌", limit_ok, "；".join(limit_msgs)))

    for name, ok, detail in results:
        print(f"{'✅' if ok else '❌'} {name}：{detail}")

    passed = all(ok for _, ok, _ in results)
    print("=" * 60)
    if passed:
        print("✅ 全部通过，这份数据可以进回测。")
        print("   直接运行：python run_backtest.py --csv 全部清洗后数据.csv")
    else:
        print("❌ 不合格，数据不能进回测。")
        print("   请用 task1_数据抓取_修复版.py 重新抓取，不要手工改 CSV。")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    code = main()
    # 双击运行时窗口不会一闪而过，停下来让人看清结果
    if len(sys.argv) == 1 and sys.stdin.isatty():
        input("\n按回车键退出……")
    raise SystemExit(code)
