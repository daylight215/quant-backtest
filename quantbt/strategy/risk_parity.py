# -*- coding: utf-8 -*-
"""策略三：逆波动率配置（风险平价的简化版，对应课程「资产定价 / 低波动异象」）。

思路
----
不预测涨跌，只做一件事：**让每只股票对组合的风险贡献大致相等**。
权重 w_i ∝ 1 / σ_i（σ_i 为过去 lookback 日收益率标准差），每月末再平衡。

为什么这算一个"策略"而不是"随便配权重"？
1. 低波动异象（Ang, Hodrick, Zhang & Chen, 2006）：低波动股票的**风险调整后**收益
   长期高于高波动股票，A 股同样成立（散户偏好高波动彩票型股票，抬高了高波动股的估值）。
2. 风险预算：组合波动由少数高波动股主导时，分散化是假的。逆波动率加权把风险拉平，
   在不加杠杆的前提下提高夏普。
3. 换手极低（实测年化 0.3 倍）：每月只做小幅再平衡，交易成本几乎可以忽略。

参数
----
lookback    波动率估计窗口，默认 60 个交易日（20~120 日结果高度一致，见报告稳健性表）
rebalance   再平衡频率：'monthly'（默认）或 'weekly'
vol_target  组合波动率目标，0 表示不启用。启用后会按目标波动缩放总仓位
            （例如 0.15 = 年化 15%）。本项目默认关闭，把风险预算做纯。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from .base import Strategy, validate_weights


def _sample_mask(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    freq = freq.lower()
    period = {"monthly": "M", "m": "M", "weekly": "W", "w": "W"}.get(freq)
    if period is None:
        raise ValueError(f"不支持的再平衡频率：{freq}")
    last = index.to_series().groupby(index.to_period(period)).max()
    mask = pd.Series(np.nan, index=index)
    mask.loc[pd.DatetimeIndex(last.values)] = 1.0
    return mask


class RiskParity(Strategy):
    name = "risk_parity"
    display_name = "逆波动率风险平价（月度再平衡）"

    def __init__(
        self,
        lookback: int = 60,
        rebalance: str = "monthly",
        vol_target: float = 0.0,
        vol_lookback: int = 60,
    ) -> None:
        super().__init__(lookback=lookback, rebalance=rebalance,
                         vol_target=vol_target, vol_lookback=vol_lookback)
        self.lookback = lookback
        self.rebalance_freq = rebalance
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback

    def generate_weights(self, md: MarketData) -> pd.DataFrame:
        close = md.close
        rets = close.pct_change()

        # 1) 逆波动率权重：w_i ∝ 1/σ_i
        vol = rets.rolling(self.lookback, min_periods=self.lookback).std()
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
        weights = inv.div(inv.sum(axis=1), axis=0)

        # 2) 可选：组合波动率目标（按权重加权的事后组合收益估计波动）
        if self.vol_target and self.vol_target > 0:
            port_ret = (weights * rets).sum(axis=1)
            port_vol = (
                port_ret.rolling(self.vol_lookback, min_periods=self.vol_lookback).std()
                * np.sqrt(252)
            )
            scale = (self.vol_target / port_vol).clip(upper=1.0).fillna(0.0)
            weights = weights.mul(scale, axis=0)

        # 3) 只在再平衡日更新，其余日 NaN = 持有不动
        mask = _sample_mask(close.index, self.rebalance_freq)
        weights = weights.where(mask.notna())
        return validate_weights(weights, md.codes)
