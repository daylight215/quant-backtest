# -*- coding: utf-8 -*-
"""全局配置。

为什么不写成 config.yaml？
    本机 quant 环境没有安装 PyYAML，而作业要求"可在 README 中复现运行"。
    用 dataclass 做配置，零额外依赖、IDE 有补全、参数还能被 argparse 覆盖，
    比 yaml 更适合这个规模的项目。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# 项目根目录（quantbt/config.py 的上两级）
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "output"


# ---------------------------------------------------------------------------
# 标的池
# ---------------------------------------------------------------------------
# 代码 -> 名称。数据来自任务1同学提供的样本池（茅台/招行/五粮液/沪深300）。
# 沪深300 只做基准，不参与选股。
ASSETS: dict[str, str] = {
    "600519": "贵州茅台",
    "600036": "招商银行",
    "000858": "五粮液",
}

BENCHMARK_CODE = "000300"
BENCHMARK_NAME = "沪深300"


@dataclass
class CostConfig:
    """交易成本模型（A 股现实规则）。"""

    commission_rate: float = 0.00025   # 佣金：万 2.5，双边
    min_commission: float = 5.0        # 单笔最低 5 元
    slippage_bps: float = 5.0          # 滑点：5 个基点，按成交价方向不利
    stamp_duty_rate: float = 0.001     # 印花税：卖出单边，2023-08-28 前千分之一
    stamp_duty_rate_after: float = 0.0005  # 2023-08-28 起减半为万分之五
    stamp_duty_change_date: str = "2023-08-28"

    def stamp_rate(self, date) -> float:
        """按成交日返回印花税率（卖出才收）。"""
        import pandas as pd

        return (
            self.stamp_duty_rate
            if pd.Timestamp(date) < pd.Timestamp(self.stamp_duty_change_date)
            else self.stamp_duty_rate_after
        )


@dataclass
class BacktestConfig:
    """一次回测的全部设置。修改这里即可复现任意一组实验。"""

    # --- 区间 ---
    start: str = "2018-01-01"
    end: str = "2025-12-31"
    # 样本内/样本外切分点：2022-12-31 之后为样本外（理由见 README「样本切分」）
    split_date: str = "2022-12-31"

    # --- 资金与撮合 ---
    initial_cash: float = 1_000_000.0
    lot_size: int = 100          # A 股一手 100 股
    rebalance_band: float = 0.02  # 目标权重与当前权重偏离超过 2% 才调仓，抑制无效换手
    min_trade_value: float = 5_000.0  # 单笔低于 5000 元不交易（最低 5 元佣金会吃掉收益）

    # --- 绩效 ---
    risk_free_annual: float = 0.02
    trading_days_per_year: int = 252

    # --- 数据 ---
    price_adjust: str = "qfq"    # 前复权
    csv_path: str | None = None  # 指定任务1同学的 CSV 时优先用本地文件

    cost: CostConfig = field(default_factory=CostConfig)

    def to_dict(self) -> dict:
        return asdict(self)
