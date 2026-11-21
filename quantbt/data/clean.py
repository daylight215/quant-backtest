# -*- coding: utf-8 -*-
"""数据清洗与对齐。

三件事：
1. 校验：把"这份数据能不能用"变成可打印的体检报告（缺口、重复、异常零价）。
2. 对齐：多标的交易日并集 + 前向填充。停牌日 close 沿用上一日（净值不变），
   但 open 保持 NaN —— 引擎看到 NaN 就不下单，天然处理停牌。
3. 面板化：把长表（trade_date/code/...）透视成宽表（index=日期, columns=代码）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STD_COLS = ["date", "open", "high", "low", "close", "volume"]


@dataclass
class DataQualityReport:
    """单标的体检结果，最后汇总进日志/文档，证明数据被真的检查过。"""

    code: str
    rows: int
    start: str
    end: str
    duplicate_dates: int
    zero_volume_days: int
    missing_close: int
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "代码": self.code,
            "行数": self.rows,
            "起始日": self.start,
            "结束日": self.end,
            "重复日期": self.duplicate_dates,
            "零成交日": self.zero_volume_days,
            "缺失收盘": self.missing_close,
            "备注": "；".join(self.notes),
        }


def audit(df: pd.DataFrame, code: str) -> DataQualityReport:
    """对单标的数据做体检。"""
    dup = int(df["date"].duplicated().sum())
    zero_vol = int((df["volume"].fillna(0) <= 0).sum())
    missing = int(df["close"].isna().sum())
    notes = []
    if dup:
        notes.append("存在重复日期，已保留最后一条")
    if zero_vol:
        notes.append(f"{zero_vol} 个交易日成交量为 0（停牌/无成交）")
    if missing:
        notes.append("存在缺失收盘价")
    return DataQualityReport(
        code=code,
        rows=int(len(df)),
        start=str(df["date"].min().date()) if len(df) else "-",
        end=str(df["date"].max().date()) if len(df) else "-",
        duplicate_dates=dup,
        zero_volume_days=zero_vol,
        missing_close=missing,
        notes=notes,
    )


def pivot_long_to_fields(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """长表 -> {字段名: 宽表}。

    支持任务1同学的数据格式：
        trade_date, code, name, open, high, low, close, volume, return
    """
    df = df.copy()
    date_col = "trade_date" if "trade_date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.sort_values([date_col, "code"]).drop_duplicates(["code", date_col], keep="last")

    out = {}
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            out[col] = df.pivot(index=date_col, columns="code", values=col).sort_index()
    out["names"] = (
        df.drop_duplicates("code").set_index("code")["name"].to_dict()
        if "name" in df.columns
        else {}
    )
    return out


def align_panel(
    fields: dict[str, pd.DataFrame], extra_dates: pd.Index | None = None
) -> dict[str, pd.DataFrame]:
    """把多标的宽表对齐到同一交易日历。

    - 日历 = 各标的日期并集（可并入基准指数日期）
    - close / open 前向填充：停牌期间净值不变
    - volume 填 0：停牌即无成交
    """
    close = fields["close"]
    dates = close.index
    if extra_dates is not None:
        dates = dates.union(pd.DatetimeIndex(extra_dates)).sort_values()

    aligned = {}
    for col in ["open", "close", "volume"]:
        if col not in fields:
            continue
        panel = fields[col].reindex(dates)
        if col == "volume":
            panel = panel.fillna(0.0)
        else:
            # 只前向填充：绝不能用未来价格回填（那是最典型的未来函数）
            panel = panel.ffill()
        aligned[col] = panel

    # 某标的上市前的整段 NaN 仍然保留，引擎会自动跳过
    return aligned


def to_wide(frames: list[tuple[str, pd.DataFrame]], field_name: str) -> pd.DataFrame:
    """把 [('600519', df), ...] 拼成宽表。"""
    series = {}
    for code, df in frames:
        if df is None or len(df) == 0:
            continue
        series[code] = df.set_index("date")[field_name]
    if not series:
        return pd.DataFrame()
    out = pd.DataFrame(series).sort_index()
    return out


def sanity_check(prices: pd.DataFrame, max_daily_move: float = 0.35) -> list[str]:
    """粗筛异常：单日涨跌幅超过 max_daily_move 的（未复权/拆股错位常见）。"""
    ret = prices.pct_change()
    hot = (ret.abs() > max_daily_move) & prices.notna()
    msgs = []
    for code in prices.columns:
        cnt = int(hot[code].sum())
        if cnt:
            msgs.append(f"{code} 有 {cnt} 个交易日涨跌幅超过 {max_daily_move:.0%}，请核查复权")
    return msgs


def constant_price_days(prices: pd.DataFrame) -> dict[str, int]:
    """连续多日收盘价完全不变的天数（停牌/数据错位的告警）。"""
    out = {}
    for code in prices.columns:
        s = prices[code].dropna()
        if len(s) < 2:
            out[code] = 0
            continue
        out[code] = int((s.diff() == 0).sum() - np.isnan(s.diff()).sum())
    return out
