# -*- coding: utf-8 -*-
"""策略注册表：用字符串 + 参数就能构造策略，方便命令行做参数实验。

    python run_backtest.py --strategy ma_timing --param fast=15 --param slow=45
"""

from __future__ import annotations

import inspect

from .base import Strategy
from .buy_hold import EqualWeightHold
from .ma_timing import DualMATiming
from .momentum import MomentumRotation
from .risk_parity import RiskParity

STRATEGIES: dict[str, type[Strategy]] = {
    "risk_parity": RiskParity,
    "ma_timing": DualMATiming,
    "momentum": MomentumRotation,
    "buy_hold": EqualWeightHold,
}


def _cast(value: str, default):
    """把命令行传来的字符串按构造函数默认值的类型转换。"""
    if isinstance(value, bool):
        return str(value).lower() in ("1", "true", "yes", "y")
    if isinstance(default, bool):
        return str(value).lower() in ("1", "true", "yes", "y")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return value
    return value


def build_strategy(name: str, **params) -> Strategy:
    """按名字构造策略，参数类型自动对齐构造函数默认值。"""
    if name not in STRATEGIES:
        raise KeyError(f"未知策略 {name}，可选：{list(STRATEGIES)}")
    cls = STRATEGIES[name]
    signature = inspect.signature(cls.__init__)
    defaults = {
        k: v.default
        for k, v in signature.parameters.items()
        if k != "self" and v.default is not inspect.Parameter.empty
    }
    unknown = set(params) - set(defaults)
    if unknown:
        raise TypeError(f"{name} 不支持参数 {unknown}，可用参数：{list(defaults)}")
    casted = {k: _cast(v, defaults[k]) for k, v in params.items()}
    return cls(**casted)
