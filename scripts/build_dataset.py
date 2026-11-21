# -*- coding: utf-8 -*-
"""构建全组统一的数据集：真实交易日 + 前复权 + 6 位代码。

为什么需要这个脚本？
    任务1 初版脚本有两处硬伤：
    1) ak.stock_zh_a_daily(symbol="600519") 少了交易所前缀，会直接 KeyError: 'date'；
    2) pd.date_range(freq="D") 把周末和节假日也补进了数据，凭空多出 28.5% 的"交易日"。
    结果就是回测在周六也会成交，年化/夏普/回撤全部失真。

本脚本的做法：
    - 只保留接口返回的真实交易日，绝不做日历补齐；
    - 代码统一 6 位，名称单独一列；
    - 前复权（qfq），分红送转的影响已包含在价格里；
    - **额外多抓约 450 个自然日的历史做"预热"**（is_warmup=True）：
      逆波动率、均线都要 60~200 天历史，如果数据正好从 2018-01-02 开始，
      策略前三个月只能空仓，回测从第一天就失真。预热行只用于算指标，不计入样本。
    - 输出字段：trade_date, code, name, open, high, low, close, volume, return,
      is_warmup, is_in_sample。

用法：
    python scripts/build_dataset.py
    python scripts/build_dataset.py --split 2022-12-31 --out data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ASSETS = {
    "600519": "贵州茅台",
    "600036": "招商银行",
    "000858": "五粮液",
    "000300": "沪深300",
}

EM_MAP = {"日期": "trade_date", "开盘": "open", "收盘": "close",
          "最高": "high", "最低": "low", "成交量": "volume"}


def _prefix(code: str) -> str:
    return "sh" if code.startswith(("5", "6", "9")) else "sz"


def fetch(code: str, start: str, end: str) -> pd.DataFrame:
    """抓单个标的，返回标准列。

    个股优先走新浪（`stock_zh_a_daily`，必须带 sh/sz 前缀），东财兜底。
    原因：新浪与任务1 约定一致、复权口径稳定；东财接口在部分网络下会断连，
    且两种源的 qfq 复权基准不完全相同（收益相关系数约 0.98，而非 1.0），
    混用会让全组数据口径不一致。
    """
    import akshare as ak

    if code == "000300":
        raw = ak.stock_zh_index_daily(symbol="sh000300")
        df = raw.rename(columns={"date": "trade_date"})
    else:
        try:
            raw = ak.stock_zh_a_daily(symbol=f"{_prefix(code)}{code}", adjust="qfq")
            df = raw.rename(columns={"date": "trade_date"})
        except Exception:  # 新浪偶尔抽风时退回东财
            raw = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start.replace("-", ""), end_date=end.replace("-", ""),
                adjust="qfq",
            )
            df = raw.rename(columns=EM_MAP)

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[(df["trade_date"] >= pd.Timestamp(start)) & (df["trade_date"] <= pd.Timestamp(end))]
    df = df[["trade_date", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 去重、排序、剔除异常
    df = (df.drop_duplicates("trade_date", keep="last")
            .sort_values("trade_date")
            .reset_index(drop=True))
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    df["code"] = code
    df["name"] = ASSETS[code]
    df["return"] = df["close"].pct_change()
    return df


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="构建统一数据集")
    p.add_argument("--start", default="2018-01-02")
    p.add_argument("--end", default="2026-11-20")
    p.add_argument("--split", default="2021-12-31", help="样本内截止日")
    p.add_argument("--out", default=str(ROOT / "data" / "processed"))
    args = p.parse_args(argv)

    frames = []
    warmup_start = (pd.Timestamp(args.start) - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    for code in ASSETS:
        print(f"抓取 {code} {ASSETS[code]} …")
        frames.append(fetch(code, warmup_start, args.end))

    df = pd.concat(frames, ignore_index=True)
    df["is_warmup"] = df["trade_date"] < pd.Timestamp(args.start)
    df["is_in_sample"] = (df["trade_date"] >= pd.Timestamp(args.start)) & (
        df["trade_date"] <= pd.Timestamp(args.split))
    df = df.sort_values(["trade_date", "code"]).reset_index(drop=True)

    # —— 体检：三条硬指标，不通过就别拿去回测 ——
    weekend = df[df["trade_date"].dt.dayofweek >= 5]
    print("\n===== 数据体检 =====")
    print(f"总行数 {len(df)}，标的 {df['code'].nunique()} 个，"
          f"区间 {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
    print(f"每个标的行数：{df.groupby('code').size().to_dict()}")
    print(f"周末行数：{len(weekend)}（必须是 0）")
    print(f"重复(日期,代码)：{int(df.duplicated(['trade_date', 'code']).sum())}（必须是 0）")
    mt = df[(df["code"] == "600519") & (df["trade_date"] == "2018-01-02")]
    if len(mt):
        print(f"抽查 茅台 2018-01-02 前复权收盘：{mt['close'].iloc[0]:.2f}（应在 579 附近）")
    hs = df[(df["code"] == "000300") & (df["trade_date"] == "2018-01-02")]
    if len(hs):
        print(f"抽查 沪深300 2018-01-02 收盘：{hs['close'].iloc[0]:.2f}（应在 4080 附近）")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["trade_date", "code", "name", "open", "high", "low", "close",
            "volume", "return", "is_warmup", "is_in_sample"]

    df[cols].to_csv(out / "全部清洗后数据.csv", index=False, encoding="utf-8-sig")
    df[df["is_in_sample"]][cols].to_csv(out / "样本内数据.csv", index=False, encoding="utf-8-sig")
    df[(~df["is_warmup"]) & (~df["is_in_sample"])][cols].to_csv(
        out / "样本外数据.csv", index=False, encoding="utf-8-sig")
    print(f"\n已导出到：{out}")
    print("提示：'全部清洗后数据.csv' 含预热期，最适合直接喂给回测；")
    print("      样本内/样本外两个文件用于报告里的分段说明。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
