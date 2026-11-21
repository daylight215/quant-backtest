# -*- coding: utf-8 -*-
"""事件驱动日频回测引擎。

时间轴（务必看懂这一张图，答辩必问"你怎么避免未来函数"）：

    交易日 t-1 收盘：策略用 t-1 及之前的数据算出目标权重 w(t-1)
    交易日 t   开盘：引擎按 w(t-1) 调仓，成交价 = 开盘价 × (1 ± 滑点)
    交易日 t   收盘：用收盘价给组合估值，得到净值 NA V(t)

因此任何一天用到的价格都严格早于或等于当天开盘，不存在"用当天收盘价决定当天成交"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..data.loader import MarketData
from ..strategy.base import Strategy
from .broker import Portfolio


@dataclass
class BacktestResult:
    """回测产物：后面所有指标、图表、报告都从这里取数。"""

    nav: pd.Series                  # 策略净值（起点 1.0 之前是初始资金口径）
    benchmark: pd.Series            # 基准净值（起点 1.0）
    exposure: pd.Series             # 每日股票仓位占比（0~1）
    weights: pd.DataFrame           # 实际执行的目标权重
    trades: pd.DataFrame            # 成交明细
    strategy_name: str
    params: dict
    config: BacktestConfig
    initial_cash: float
    benchmark_name: str = "沪深300"
    strategy_label: str = ""
    cash: pd.Series = field(default=None)  # type: ignore[assignment]

    @property
    def returns(self) -> pd.Series:
        return self.nav.pct_change().dropna()

    @property
    def benchmark_returns(self) -> pd.Series:
        return self.benchmark.pct_change().dropna()

    @property
    def start(self) -> pd.Timestamp:
        return self.nav.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.nav.index[-1]


class BacktestEngine:
    """把 行情 + 策略 + 配置 跑成一条净值曲线。"""

    def __init__(self, config: BacktestConfig, data: MarketData, strategy: Strategy) -> None:
        self.config = config
        self.data = data
        self.strategy = strategy

    # ------------------------------------------------------------------ #
    def run(self) -> BacktestResult:
        cfg = self.config
        md = self.data

        # 只交易 [start, end] 区间；但行情保留了 start 之前的一段，用来预热均线
        trade_dates = md.close.index[
            (md.close.index >= pd.Timestamp(cfg.start)) & (md.close.index <= pd.Timestamp(cfg.end))
        ]
        if len(trade_dates) < 2:
            raise ValueError(f"回测区间 {cfg.start} ~ {cfg.end} 内没有足够交易日")

        # 策略信号 -> 次日执行（shift(1) 就是"T 日信号、T+1 成交"的代码化）
        raw_weights = self.strategy.generate_weights(md)
        raw_weights = raw_weights.reindex(md.close.index)
        exec_weights = raw_weights.shift(1)

        portfolio = Portfolio(cfg.initial_cash, cfg.cost, cfg.lot_size)
        prev_close = md.close.shift(1)

        navs, cash_list, expo_list = [], [], []
        executed = pd.DataFrame(np.nan, index=trade_dates, columns=md.codes)

        for date in trade_dates:
            target = exec_weights.loc[date]
            # 全为 NaN 表示"今天没有新信号"（如买入持有策略），不调仓
            if not target.isna().all():
                target = target.fillna(0.0)
                val_prices = prev_close.loc[date]
                current = portfolio.weights(val_prices)
                equity = portfolio.total_value(val_prices)
                tolerance = self._tolerance(target, val_prices, equity)
                if self._need_rebalance(target, current, tolerance):
                    portfolio.rebalance(
                        date, target, md.open.loc[date], val_prices,
                        min_trade_value=cfg.min_trade_value,
                        weight_tolerance=tolerance,
                    )
                    executed.loc[date] = target

            close_prices = md.close.loc[date]
            navs.append(portfolio.total_value(close_prices))
            cash_list.append(portfolio.cash)
            equity = navs[-1]
            expo_list.append(0.0 if equity <= 0 else (equity - portfolio.cash) / equity)

        nav = pd.Series(navs, index=trade_dates, name="nav")
        cash = pd.Series(cash_list, index=trade_dates, name="cash")
        exposure = pd.Series(expo_list, index=trade_dates, name="exposure")

        # 基准：同一区间、同一交易日历，起点归一化到 1.0
        bench = md.benchmark.reindex(trade_dates).ffill()
        bench = bench / bench.iloc[0]

        return BacktestResult(
            nav=nav,
            benchmark=bench,
            exposure=exposure,
            weights=executed,
            trades=portfolio.trades_frame(),
            strategy_name=self.strategy.describe(),
            params=dict(self.strategy.params),
            config=cfg,
            initial_cash=cfg.initial_cash,
            benchmark_name=getattr(md, "benchmark_name", "沪深300"),
            strategy_label=getattr(self.strategy, "display_name", self.strategy.name),
            cash=cash,
        )

    # ------------------------------------------------------------------ #
    def _tolerance(self, target: pd.Series, prices: pd.Series, equity: float) -> pd.Series:
        """每只标的的权重容忍带 = max(配置带宽, 一手占组合的比例)。

        茅台一手约 15 万元，占 100 万组合的 15%，目标权重 1/3 根本买不到——
        若用固定的 2% 带宽，引擎会天天买 100 股其它股票来"凑"权重，
        白白贡献几千元手续费。按可交易粒度设阈值才是对的。
        """
        band = self.config.rebalance_band
        lot_weight = (prices.reindex(target.index) * self.config.lot_size / equity).fillna(0.0)
        return pd.Series(np.maximum(band, lot_weight), index=target.index)

    def _need_rebalance(
        self, target: pd.Series, current: pd.Series, tolerance: pd.Series
    ) -> bool:
        """有任意一只标的偏离超过自身容忍带（或发生建仓/清仓）才触发调仓。"""
        diff = (target - current).abs()
        entry = (target > 1e-9) & (current <= 1e-9)   # 空仓 -> 建仓
        exit_ = (target <= 1e-9) & (current > 1e-9)   # 持仓 -> 清仓
        return bool((diff > tolerance).any() or entry.any() or exit_.any())
