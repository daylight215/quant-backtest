# -*- coding: utf-8 -*-
"""对照策略：等权买入持有（不做任何择时）。

作用有两个：
1. 作为"配置基准"——如果策略跑不赢等权持有，说明择时在样本期是负贡献；
2. 检验回测引擎是否正确：等权买入持有的净值应当与
   (各股票累计涨幅的等权组合) 基本一致（差异只来自首日建仓的取整和费用）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import MarketData
from .base import Strategy, validate_weights


class EqualWeightHold(Strategy):
    name = "buy_hold"
    display_name = "等权买入持有（对照）"

    def generate_weights(self, md: MarketData) -> pd.DataFrame:
        codes = md.codes
        idx = md.close.index
        w = pd.DataFrame(np.nan, index=idx, columns=codes, dtype=float)

        # 在"回测起点前最后一个交易日"发出信号 -> 引擎在起点当天开盘建仓，
        # 之后全部 NaN（不调仓），这才是真正的买入持有。
        start = md.trade_start if md.trade_start is not None else idx[0]
        pos = int(idx.searchsorted(pd.Timestamp(start)))
        signal_pos = max(pos - 1, 0)
        w.iloc[signal_pos, :] = 1.0 / len(codes)
        return validate_weights(w, codes)
