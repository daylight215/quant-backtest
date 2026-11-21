# -*- coding: utf-8 -*-
"""策略基类与接口约定。

**这是整个项目最容易埋 bug 的地方，先把约定写死：**

1. 策略只能看到 `MarketData`，看不到账户、现金、持仓，更看不到未来数据。
2. `generate_weights` 返回的 DataFrame：
       index   = 信号日（用该日收盘价计算，绝不使用次日价格）
       columns = 标的代码
       values  = 目标权重（0~1，各列之和 <= 1，剩余部分即现金）
   某格为 NaN 表示"当日对该标的没有新意见"，引擎会保持原持仓。
3. 引擎在信号日的**下一个交易日开盘**执行，即 T 日信号 / T+1 开盘成交，
   符合 A 股 T+1 的现实约束，也彻底杜绝"用当日收盘价成交"的未来函数。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.loader import MarketData


class Strategy(ABC):
    """所有策略的父类。"""

    name: str = "base"
    display_name: str = "策略"   # 中文展示名，用于图表标题和报告

    def __init__(self, **params) -> None:
        self.params = dict(params)

    @abstractmethod
    def generate_weights(self, md: MarketData) -> pd.DataFrame:
        """返回目标权重面板（见模块 docstring 的约定）。"""

    def describe(self) -> str:
        """一句话描述，用于图表标题和日志。"""
        if not self.params:
            return self.name
        body = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({body})"

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"<{self.__class__.__name__} {self.params}>"


def validate_weights(weights: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """统一格式并做基本校验：不能有负权重、单日总仓位不能超过 100%。"""
    w = weights.copy()
    # 补齐缺失列，避免策略少写一个标的就悄悄失配
    for code in codes:
        if code not in w.columns:
            w[code] = np.nan
    w = w[codes].sort_index()

    valid = w.dropna(how="all")
    if (valid.fillna(0) < -1e-9).any().any():
        raise ValueError("策略输出了负权重（本项目不支持做空）")
    gross = valid.fillna(0).sum(axis=1)
    if (gross > 1 + 1e-6).any():
        bad = gross[gross > 1 + 1e-6].index[:3]
        raise ValueError(f"单日总权重超过 100%：{list(bad)}")
    return w


def ma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均。用 min_periods=window 保证前 window-1 天为 NaN（避免早期伪信号）。"""
    return series.rolling(window=window, min_periods=window).mean()


@dataclass
class WeightBook:
    """帮策略记录"每个信号日的目标权重"，最后一次性转成 DataFrame。"""

    dates: list[pd.Timestamp] = field(default_factory=list)
    rows: list[dict[str, float]] = field(default_factory=list)

    def add(self, date: pd.Timestamp, weights: dict[str, float]) -> None:
        self.dates.append(date)
        self.rows.append(weights)

    def to_frame(self, codes: list[str]) -> pd.DataFrame:
        if not self.dates:
            return pd.DataFrame(columns=codes)
        df = pd.DataFrame(self.rows, index=pd.DatetimeIndex(self.dates))
        return df.reindex(columns=codes)
