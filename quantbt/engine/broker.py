# -*- coding: utf-8 -*-
"""账户与撮合：A 股规则下的最小可用实现。

规则清单（写进报告，答辩时老师大概率会问）：
1. T+1：当日买入次日才能卖 —— 本引擎在"信号日次日开盘"成交，天然满足。
2. 一手 100 股：买入数量向下取整到 100 的倍数；清仓时允许卖出零股。
3. 费用：佣金万 2.5（最低 5 元，双边）+ 印花税（卖出，2023-08-28 前千分之一，
   之后万分之五）+ 滑点 5bp（买入上浮、卖出下调）。
4. 停牌：开盘价为 NaN 的标的不撮合，持仓保留、按最近收盘价估值。
5. 现金不透支：买入时若现金不足，按手数递减，直到买得起或买不了。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from ..config import CostConfig


@dataclass
class Trade:
    """一笔成交记录，最终汇总成 trades.csv。"""

    date: pd.Timestamp
    code: str
    side: str          # BUY / SELL
    shares: int
    price: float       # 含滑点的成交价
    amount: float      # 成交金额（不含费用）
    commission: float
    stamp_duty: float
    slippage_cost: float

    def as_dict(self) -> dict:
        d = asdict(self)
        d["date"] = str(pd.Timestamp(d["date"]).date())
        return d


class Portfolio:
    """现金 + 持仓。只做多、不加杠杆。"""

    def __init__(self, cash: float, cost: CostConfig, lot_size: int = 100) -> None:
        self.initial_cash = float(cash)
        self.cash = float(cash)
        self.cost = cost
        self.lot_size = lot_size
        self.positions: dict[str, int] = {}
        self.trades: list[Trade] = []

    # ------------------------------------------------------------------ #
    def market_value(self, prices: pd.Series) -> float:
        """持仓市值。prices 为估值价（停牌股用最近收盘价）。"""
        total = 0.0
        for code, shares in self.positions.items():
            price = prices.get(code, float("nan"))
            if pd.isna(price):
                continue
            total += shares * float(price)
        return float(total)

    def total_value(self, prices: pd.Series) -> float:
        return self.cash + self.market_value(prices)

    def weights(self, prices: pd.Series) -> pd.Series:
        """当前各标的市场价值占总资产的比例（用于调仓阈值判断）。"""
        total = self.total_value(prices)
        out = {}
        for code in prices.index:
            price = prices.get(code)
            shares = self.positions.get(code, 0)
            out[code] = 0.0 if pd.isna(price) or total <= 0 else shares * float(price) / total
        return pd.Series(out)

    # ------------------------------------------------------------------ #
    def _commission(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return max(amount * self.cost.commission_rate, self.cost.min_commission)

    def rebalance(
        self,
        date: pd.Timestamp,
        target_weights: pd.Series,
        exec_prices: pd.Series,
        valuation_prices: pd.Series,
        min_trade_value: float = 0.0,
        weight_tolerance: pd.Series | None = None,
    ) -> None:
        """按目标权重调仓。

        exec_prices       当日开盘价（NaN=停牌，跳过）
        valuation_prices  估值价（一般用上一日收盘，停牌股为最近收盘）
        min_trade_value   单笔最小成交金额，过滤掉"为了 100 股来回倒"的碎单
        weight_tolerance  每只标的的权重容忍带：偏离小于它就不动这只票。
                          没有它就会出现"茅台一手 15 万买不到 1/3，引擎天天拿别的股票
                          凑权重"的碎单地狱。
        """
        equity = self.total_value(valuation_prices)
        if equity <= 0:
            return
        current = self.weights(valuation_prices)

        # 1) 先算目标股数（偏离在容忍带内的标的不参与本次调仓）
        target_shares: dict[str, int] = {}
        for code, weight in target_weights.items():
            tol = 0.0 if weight_tolerance is None else float(weight_tolerance.get(code, 0.0))
            cur_w = float(current.get(code, 0.0))
            entering = weight > 1e-9 and cur_w <= 1e-9
            exiting = weight <= 1e-9 and cur_w > 1e-9
            if not entering and not exiting and abs(weight - cur_w) <= tol:
                continue

            price = exec_prices.get(code, float("nan"))
            if pd.isna(price) or price <= 0:
                continue  # 停牌：维持原状
            if weight <= 1e-9:
                target_shares[code] = 0
            else:
                lots = math.floor(weight * equity / float(price) / self.lot_size)
                target_shares[code] = int(lots * self.lot_size)

        # 2) 先卖后买（保证买入有现金）
        for code, want in target_shares.items():
            held = self.positions.get(code, 0)
            if want >= held:
                continue
            price = exec_prices.get(code, float("nan"))
            if pd.isna(price) or price <= 0:
                continue
            shares = held - want
            # 清仓允许零股，部分减仓按整手
            if want > 0:
                shares = (shares // self.lot_size) * self.lot_size
            if shares <= 0:
                continue
            # 碎单过滤：不清仓、且金额太小就不动
            if want > 0 and shares * float(price) < min_trade_value:
                continue
            self._execute(date, code, "SELL", shares, float(price))

        # 3) 买入（现金不足则按手递减）
        for code, want in target_shares.items():
            held = self.positions.get(code, 0)
            if want <= held:
                continue
            price = exec_prices.get(code, float("nan"))
            if pd.isna(price) or price <= 0:
                continue
            shares = want - held
            while shares > 0 and self._buy_cost(code, shares, float(price)) > self.cash:
                shares -= self.lot_size
            if shares <= 0:
                continue
            # 建仓（held=0）不受碎单限制，否则小资金永远买不进
            if held > 0 and shares * float(price) < min_trade_value:
                continue
            self._execute(date, code, "BUY", shares, float(price))

    def _buy_cost(self, code: str, shares: int, price: float) -> float:
        slip = self.cost.slippage_bps / 10000.0
        amount = shares * price * (1 + slip)
        return amount + self._commission(amount)

    def _execute(
        self, date: pd.Timestamp, code: str, side: str, shares: int, price: float
    ) -> None:
        slip = self.cost.slippage_bps / 10000.0
        if side == "BUY":
            fill = price * (1 + slip)
            amount = shares * fill
            commission = self._commission(amount)
            stamp = 0.0
            self.cash -= amount + commission
        else:
            fill = price * (1 - slip)
            amount = shares * fill
            commission = self._commission(amount)
            stamp = amount * self.cost.stamp_rate(date)
            self.cash += amount - commission - stamp

        held = self.positions.get(code, 0)
        new_held = held + shares if side == "BUY" else held - shares
        if new_held <= 0:
            self.positions.pop(code, None)
        else:
            self.positions[code] = new_held

        self.trades.append(
            Trade(
                date=pd.Timestamp(date),
                code=code,
                side=side,
                shares=int(shares),
                price=round(fill, 4),
                amount=round(amount, 2),
                commission=round(commission, 2),
                stamp_duty=round(stamp, 2),
                slippage_cost=round(shares * price * slip, 2),
            )
        )

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=["date", "code", "side", "shares", "price", "amount",
                         "commission", "stamp_duty", "slippage_cost"]
            )
        return pd.DataFrame([t.as_dict() for t in self.trades])
