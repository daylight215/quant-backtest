# -*- coding: utf-8 -*-
"""单元测试：不需要联网、不依赖 pytest，直接 `python -m unittest discover -s tests -v`。

覆盖三类最容易出错、也最能体现"引擎是真的"的地方：
1. 指标算法（累计/年化/回撤/夏普）与手算结果一致；
2. 交易规则（一手 100 股、最低佣金、现金不透支、印花税时点）；
3. 回测引擎的时间轴（T 日信号、T+1 开盘成交，不存在未来函数）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantbt.config import BacktestConfig, CostConfig
from quantbt.data.loader import MarketData
from quantbt.engine import metrics
from quantbt.engine.backtest import BacktestEngine
from quantbt.engine.broker import Portfolio
from quantbt.strategy.base import Strategy, validate_weights


class TestMetrics(unittest.TestCase):
    def test_total_and_annual_return(self):
        nav = pd.Series([1.0, 1.1, 1.21], index=pd.date_range("2020-01-01", periods=3))
        self.assertAlmostEqual(metrics.total_return(nav), 0.21, places=6)
        # 2 个交易日赚 21%，年化 = 1.21^(252/2) - 1（一个极端但可验证的数）
        expected = 1.21 ** (252 / 2) - 1
        self.assertAlmostEqual(metrics.annual_return(nav), expected, places=4)

    def test_max_drawdown(self):
        nav = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3],
                        index=pd.date_range("2020-01-01", periods=5))
        mdd, peak, trough, recovery = metrics.max_drawdown(nav)
        self.assertAlmostEqual(mdd, 0.9 / 1.2 - 1, places=6)
        self.assertEqual(peak, nav.index[1])
        self.assertEqual(trough, nav.index[2])
        self.assertEqual(recovery, nav.index[4])

    def test_sharpe_zero_vol(self):
        # 净值完全不动 -> 收益率恒为 0、标准差为 0 -> 夏普按约定返回 0，而不是报 ZeroDivisionError
        nav = pd.Series([1.0, 1.0, 1.0, 1.0],
                        index=pd.date_range("2020-01-01", periods=4))
        self.assertAlmostEqual(metrics.sharpe_ratio(nav.pct_change().dropna()), 0.0, places=6)

    def test_beta_alpha_of_self(self):
        rng = np.random.default_rng(0)
        rb = pd.Series(rng.normal(0, 0.01, 500))
        alpha, beta = metrics.beta_alpha(rb, rb)
        self.assertAlmostEqual(beta, 1.0, places=6)
        self.assertAlmostEqual(alpha, 0.0, places=6)


class TestBroker(unittest.TestCase):
    def setUp(self):
        self.cost = CostConfig(commission_rate=0.00025, min_commission=5.0,
                               slippage_bps=0.0, stamp_duty_rate=0.001,
                               stamp_duty_rate_after=0.0005)
        self.p = Portfolio(100_000.0, self.cost, lot_size=100)

    def test_lot_size_and_cash(self):
        prices = pd.Series({"A": 10.0, "B": 20.0})
        self.p.rebalance(pd.Timestamp("2024-01-02"),
                         pd.Series({"A": 0.5, "B": 0.5}), prices, prices)
        for code, shares in self.p.positions.items():
            self.assertEqual(shares % 100, 0, f"{code} 不是整手")
        self.assertGreaterEqual(self.p.cash, -1e-6, "现金不能为负")

    def test_min_commission(self):
        prices = pd.Series({"A": 10.0})
        self.p.rebalance(pd.Timestamp("2024-01-02"), pd.Series({"A": 0.01}), prices, prices)
        trade = self.p.trades[0]
        self.assertGreaterEqual(trade.commission, 5.0)
        self.assertEqual(trade.shares % 100, 0)

    def test_stamp_duty_changes(self):
        self.assertAlmostEqual(self.cost.stamp_rate("2023-08-27"), 0.001)
        self.assertAlmostEqual(self.cost.stamp_rate("2023-08-28"), 0.0005)

    def test_no_negative_cash_when_expensive(self):
        prices = pd.Series({"A": 1500.0})
        self.p.rebalance(pd.Timestamp("2024-01-02"), pd.Series({"A": 1.0}), prices, prices)
        self.assertGreaterEqual(self.p.cash, -1e-6)


class _AlwaysLong(Strategy):
    """测试用：每天都要求满仓 A。"""

    name = "test_always_long"
    display_name = "测试满仓"

    def generate_weights(self, md: MarketData) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=md.close.index, columns=md.codes)
        w["A"] = 1.0
        return validate_weights(w, md.codes)


class _Flip(Strategy):
    """测试用：第 3 个交易日信号翻多，用于验证 T+1 成交。"""

    name = "test_flip"

    def generate_weights(self, md: MarketData) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=md.close.index, columns=md.codes)
        w.iloc[2:, w.columns.get_loc("A")] = 1.0
        return validate_weights(w, md.codes)


def _fake_market(n=10, start="2024-01-01") -> MarketData:
    idx = pd.bdate_range(start, periods=n)
    close = pd.DataFrame({"A": np.linspace(100, 110, n),
                          "B": np.linspace(50, 55, n)}, index=idx)
    open_ = close * 1.0  # 开盘=收盘，方便手算
    vol = pd.DataFrame(1000.0, index=idx, columns=["A", "B"])
    bench = pd.Series(np.linspace(1.0, 1.05, n), index=idx)
    return MarketData(close=close, open=open_, volume=vol, benchmark=bench,
                      names={"A": "A", "B": "B"}, trade_start=idx[0])


class TestEngine(unittest.TestCase):
    def test_no_lookahead_t_plus_1(self):
        """第 3 天（index=2）收盘翻多，最早只能第 4 天（index=3）开盘成交。"""
        cfg = BacktestConfig(start="2024-01-01", end="2024-12-31", split_date="2024-06-30",
                             initial_cash=1_000_000, rebalance_band=0.0, min_trade_value=0.0)
        md = _fake_market(8)
        res = BacktestEngine(cfg, md, _Flip()).run()
        self.assertGreater(len(res.trades), 0)
        first_trade = res.trades.iloc[0]
        self.assertEqual(pd.Timestamp(first_trade["date"]), md.close.index[3])
        self.assertEqual(first_trade["side"], "BUY")

    def test_always_long_holds(self):
        cfg = BacktestConfig(start="2024-01-01", end="2024-12-31", split_date="2024-06-30",
                             initial_cash=1_000_000, rebalance_band=0.0, min_trade_value=0.0)
        res = BacktestEngine(cfg, _fake_market(10), _AlwaysLong()).run()
        # 建仓后仓位应始终接近 1（允许整手取整与现金留存）
        self.assertGreater(res.exposure.iloc[-1], 0.9)
        self.assertLessEqual(res.exposure.max(), 1.0 + 1e-9)

    def test_nav_starts_at_initial_cash(self):
        cfg = BacktestConfig(start="2024-01-01", end="2024-12-31", split_date="2024-06-30",
                             initial_cash=500_000)
        res = BacktestEngine(cfg, _fake_market(5), _AlwaysLong()).run()
        self.assertGreater(res.nav.iloc[0], 0.9 * 500_000)
        self.assertLess(res.nav.iloc[0], 1.1 * 500_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
