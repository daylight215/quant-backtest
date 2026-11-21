# -*- coding: utf-8 -*-
"""策略一：双均线趋势择时（对应课程「量化择时模型」）。

思路
----
对每只股票分别判断趋势：短均线在长均线上方、且收盘价在长均线上方 → 处于上升趋势，持有；
否则视为下跌/震荡 → 空仓。满足条件的标的等权分配，全部不满足则 100% 现金。

为什么用"趋势"而不是"预测涨跌"？
    均线择时的经济学解释是**动量/趋势持续**与**波动率聚集**：资产收益短期正自相关
    在 A 股这种散户占比高、情绪驱动的市场里比较明显；同时它天然是"右侧交易"，
    放弃一点涨幅换取躲开大级别下跌，因此最大回撤通常显著小于买入持有。

参数
----
fast          短均线窗口，默认 20
slow          长均线窗口，默认 60
trend_confirm 是否要求收盘价也在长均线上方（默认 True，多一层过滤）
max_gross     最大总仓位，默认 1.0（满仓），可设 0.8 做仓位管理
signal_freq   信号确认频率：'daily' / 'weekly' / 'monthly'。
              日频信号噪声大，均线附近会反复穿越，导致高换手和"假突破"亏损；
              周频（每周最后一个交易日确认一次）实测能把换手砍掉一半、回撤更小。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from .base import Strategy, validate_weights


def _sample_mask(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """返回"哪些日期可以更新信号"的掩码（其余日期为 NaN，引擎保持原仓位）。"""
    freq = freq.lower()
    if freq == "daily":
        return pd.Series(1.0, index=index)
    period = {"weekly": "W", "w": "W", "monthly": "M", "m": "M"}.get(freq)
    if period is None:
        raise ValueError(f"不支持的信号频率：{freq}")
    last = index.to_series().groupby(index.to_period(period)).max()
    mask = pd.Series(np.nan, index=index)
    mask.loc[pd.DatetimeIndex(last.values)] = 1.0
    return mask


class DualMATiming(Strategy):
    name = "ma_timing"
    display_name = "双均线趋势择时"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 60,
        trend_confirm: bool = True,
        max_gross: float = 1.0,
        signal_freq: str = "weekly",
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast({fast}) 必须小于 slow({slow})")
        super().__init__(fast=fast, slow=slow, trend_confirm=trend_confirm,
                         max_gross=max_gross, signal_freq=signal_freq)
        self.fast = fast
        self.slow = slow
        self.trend_confirm = trend_confirm
        self.max_gross = max_gross
        self.signal_freq = signal_freq

    def generate_weights(self, md: MarketData) -> pd.DataFrame:
        close = md.close
        ma_fast = close.rolling(self.fast, min_periods=self.fast).mean()
        ma_slow = close.rolling(self.slow, min_periods=self.slow).mean()

        signal = ma_fast > ma_slow
        if self.trend_confirm:
            signal &= close > ma_slow
        # 数据不足（前 slow-1 天）时均线为 NaN，比较结果为 False，自动空仓
        signal = signal.fillna(False)

        weights = signal.astype(float)
        n_active = weights.sum(axis=1)
        # 逐行除以入选个数 → 等权；没有标的入选时除数为 NaN，fillna(0) 即全现金
        weights = weights.div(n_active.where(n_active > 0), axis=0).fillna(0.0)
        weights = weights * self.max_gross

        # 只在确认日更新目标权重，其余日 NaN = 不调仓
        mask = _sample_mask(close.index, self.signal_freq)
        weights = weights.where(mask.notna())
        return validate_weights(weights, md.codes)
        return validate_weights(weights, md.codes)
