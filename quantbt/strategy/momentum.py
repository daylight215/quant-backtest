# -*- coding: utf-8 -*-
"""策略二：横截面动量轮动 + 市场择时（对应课程「多因子选股」中的动量因子）。

思路
----
每隔一段时间（周/月）计算每只股票过去 lookback 日的收益率（动量因子），
取最强的 top_k 只等权持有；同时叠加一层大盘择时：沪深300 在 200 日均线下方时
整体空仓，避免在系统性下跌里"轮动到跌得少的股票"这种伪防御。

经济逻辑
--------
动量效应（Jegadeesh & Titman, 1993）在 A 股中短期（1~6 个月）依然存在，
本质是信息扩散缓慢 + 处置效应导致的价格延迟反应；而大盘均线过滤承担了
"风险开关"的作用，两者结合是业界最常见的"选股 + 择时"两层结构。

参数
----
lookback     动量回看窗口，默认 60 个交易日
top_k        持有几只，默认 1
rebalance    调仓频率：'weekly' 或 'monthly'
market_ma    大盘均线过滤窗口，默认 200；设为 0 表示不做择时
min_momentum 只买动量为正的标的，默认 True
"""

from __future__ import annotations

import pandas as pd

from ..data.loader import MarketData
from .base import Strategy, validate_weights


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """取每个周期最后一个交易日作为调仓信号日（周五/月末）。"""
    freq = freq.lower()
    if freq in ("weekly", "w", "w-fri"):
        period = index.to_period("W")
    elif freq in ("monthly", "m", "month"):
        period = index.to_period("M")
    else:
        raise ValueError(f"不支持的调仓频率：{freq}")
    last = pd.Series(index, index=index).groupby(period).max()
    return pd.DatetimeIndex(last.values)


class MomentumRotation(Strategy):
    name = "momentum"
    display_name = "横截面动量轮动 + 大盘择时"

    def __init__(
        self,
        lookback: int = 60,
        top_k: int = 1,
        rebalance: str = "weekly",
        market_ma: int = 200,
        min_momentum: bool = True,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k 至少为 1")
        super().__init__(lookback=lookback, top_k=top_k, rebalance=rebalance,
                         market_ma=market_ma, min_momentum=min_momentum)
        self.lookback = lookback
        self.top_k = top_k
        self.rebalance_freq = rebalance
        self.market_ma = market_ma
        self.min_momentum = min_momentum

    def generate_weights(self, md: MarketData) -> pd.DataFrame:
        close = md.close
        score = close / close.shift(self.lookback) - 1.0

        # 大盘择时开关：只在基准收盘价高于其均线时允许持仓
        if self.market_ma and self.market_ma > 0:
            bench_ma = md.benchmark.rolling(self.market_ma, min_periods=self.market_ma).mean()
            risk_on = (md.benchmark > bench_ma).reindex(close.index).ffill()
        else:
            risk_on = pd.Series(True, index=close.index)

        rows: dict[pd.Timestamp, dict[str, float]] = {}
        for d in rebalance_dates(close.index, self.rebalance_freq):
            gate = risk_on.loc[d]
            # 均线数据不足时 gate 为 NaN：按"风险关闭"处理，宁可空仓也不瞎买
            if pd.isna(gate) or not bool(gate):
                rows[d] = {c: 0.0 for c in md.codes}
                continue
            s = score.loc[d].dropna()
            if self.min_momentum:
                s = s[s > 0]
            if s.empty:
                rows[d] = {c: 0.0 for c in md.codes}
                continue
            top = s.nlargest(self.top_k)
            w = 1.0 / len(top)
            rows[d] = {c: (w if c in top.index else 0.0) for c in md.codes}

        if not rows:
            return pd.DataFrame(index=close.index, columns=md.codes, dtype=float)

        book = pd.DataFrame.from_dict(rows, orient="index").reindex(
            index=close.index, columns=md.codes
        )
        # 调仓日之间沿用上一次的目标权重（买完就不动，直到下一个调仓日）
        book = book.ffill()
        return validate_weights(book, md.codes)
