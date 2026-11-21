# -*- coding: utf-8 -*-
"""绩效指标。

作业硬性要求四项：累计收益率、年化收益率、最大回撤、夏普比率。
这里另外补了卡玛、索提诺、胜率、超额收益、跟踪误差、信息比率、Beta/Alpha、换手率——
不是堆指标，而是为了回答"超额收益从哪来"：
    Beta/Alpha 说明收益是承担市场风险换来的，还是选股/择时带来的；
    跟踪误差/信息比率说明主动偏离的稳定性；
    换手率说明策略是不是靠高频交易磨出来的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 基础指标
# --------------------------------------------------------------------------- #
def total_return(nav: pd.Series) -> float:
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def annual_return(nav: pd.Series, periods: int = 252) -> float:
    n = len(nav) - 1
    if n <= 0:
        return 0.0
    return float((nav.iloc[-1] / nav.iloc[0]) ** (periods / n) - 1.0)


def annual_volatility(returns: pd.Series, periods: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.02, periods: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf_annual / periods
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.02, periods: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf_annual / periods
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods))


def max_drawdown(nav: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    """返回 (最大回撤, 峰值日, 谷底日, 收复日)。"""
    if len(nav) < 2:
        return 0.0, None, None, None
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    trough = dd.idxmin()
    mdd = float(dd.loc[trough])
    peak = nav.loc[:trough].idxmax()

    after = nav.loc[trough:]
    recovered = after[after >= nav.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    return mdd, peak, trough, recovery


def calmar_ratio(nav: pd.Series, periods: int = 252) -> float:
    mdd, *_ = max_drawdown(nav)
    if mdd == 0:
        return 0.0
    return float(annual_return(nav, periods) / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    return float((returns > 0).sum() / len(returns))


def beta_alpha(
    returns: pd.Series, bench_returns: pd.Series, periods: int = 252
) -> tuple[float, float]:
    """CAPM 回归：r_s = alpha + beta * r_b。返回年化 alpha、beta。"""
    df = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()
    if len(df) < 2:
        return 0.0, 0.0
    rs, rb = df.iloc[:, 0], df.iloc[:, 1]
    var = rb.var(ddof=1)
    if var == 0:
        return 0.0, 0.0
    beta = float(rs.cov(rb) / var)
    alpha = float((rs.mean() - beta * rb.mean()) * periods)
    return alpha, beta


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def compute_metrics(
    nav: pd.Series,
    benchmark: pd.Series | None = None,
    rf_annual: float = 0.02,
    periods: int = 252,
) -> dict:
    """一条净值曲线的完整指标。benchmark 可为空（分样本报告基准时用到）。"""
    returns = nav.pct_change().dropna()
    mdd, peak, trough, recovery = max_drawdown(nav)

    out = {
        "total_return": total_return(nav),
        "annual_return": annual_return(nav, periods),
        "annual_vol": annual_volatility(returns, periods),
        "sharpe": sharpe_ratio(returns, rf_annual, periods),
        "sortino": sortino_ratio(returns, rf_annual, periods),
        "max_drawdown": mdd,
        "calmar": calmar_ratio(nav, periods),
        "win_rate": win_rate(returns),
        "mdd_peak": str(peak.date()) if peak is not None else "",
        "mdd_trough": str(trough.date()) if trough is not None else "",
        "mdd_recovery": str(recovery.date()) if recovery is not None else "未收复",
        "n_days": int(len(nav)),
    }

    if benchmark is not None and len(benchmark) > 1:
        rb = benchmark.pct_change().dropna()
        alpha, beta = beta_alpha(returns, rb, periods)
        excess = returns - rb
        te = float(excess.std(ddof=1) * np.sqrt(periods)) if len(excess) > 1 else 0.0
        out.update(
            {
                "benchmark_total_return": total_return(benchmark),
                "benchmark_annual_return": annual_return(benchmark, periods),
                "benchmark_max_drawdown": max_drawdown(benchmark)[0],
                "benchmark_sharpe": sharpe_ratio(rb, rf_annual, periods),
                "excess_total": out["total_return"] - total_return(benchmark),
                "excess_annual": out["annual_return"] - annual_return(benchmark, periods),
                "tracking_error": te,
                "information_ratio": (
                    float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods))
                    if len(excess) > 1 and excess.std(ddof=1) > 0
                    else 0.0
                ),
                "beta": beta,
                "alpha_annual": alpha,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 换手与交易统计
# --------------------------------------------------------------------------- #
def trade_stats(trades: pd.DataFrame, nav: pd.Series, periods: int = 252) -> dict:
    """成交笔数、年化换手率（单边）、平均仓位。"""
    if trades is None or len(trades) == 0:
        return {"n_trades": 0, "turnover_annual": 0.0, "avg_exposure": 0.0,
                "total_cost": 0.0, "cost_drag_annual": 0.0}

    traded = float(trades["amount"].sum())
    years = (len(nav) - 1) / periods
    avg_nav = float(nav.mean())
    # 单边换手率 = (买入+卖出总额 / 2) / 平均资产 / 年数
    turnover = traded / 2.0 / avg_nav / years if years > 0 and avg_nav > 0 else 0.0

    total_cost = float(trades[["commission", "stamp_duty"]].sum().sum())
    return {
        "n_trades": int(len(trades)),
        "turnover_annual": float(turnover),
        "total_cost": total_cost,
        "cost_drag_annual": float(total_cost / avg_nav / years) if years > 0 and avg_nav > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# 表格化（写进报告 / 打印到终端）
# --------------------------------------------------------------------------- #
METRIC_LABELS = {
    "total_return": "累计收益率",
    "annual_return": "年化收益率",
    "annual_vol": "年化波动率",
    "sharpe": "夏普比率",
    "sortino": "索提诺比率",
    "max_drawdown": "最大回撤",
    "calmar": "卡玛比率",
    "win_rate": "日胜率",
    "benchmark_total_return": "基准累计收益",
    "benchmark_annual_return": "基准年化收益",
    "benchmark_max_drawdown": "基准最大回撤",
    "benchmark_sharpe": "基准夏普",
    "excess_total": "累计超额收益",
    "excess_annual": "年化超额收益",
    "tracking_error": "跟踪误差",
    "information_ratio": "信息比率",
    "beta": "Beta",
    "alpha_annual": "年化 Alpha",
    "turnover_annual": "年化换手率",
    "n_trades": "成交笔数",
}

_PCT_KEYS = {
    "total_return", "annual_return", "annual_vol", "max_drawdown", "win_rate",
    "benchmark_total_return", "benchmark_annual_return", "benchmark_max_drawdown",
    "excess_total", "excess_annual", "tracking_error", "alpha_annual",
    "turnover_annual", "cost_drag_annual",
}


def format_value(key: str, value) -> str:
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if key in _PCT_KEYS:
        return f"{value:.2%}"
    if key in ("sharpe", "sortino", "calmar", "information_ratio", "beta"):
        return f"{value:.3f}"
    if key in ("n_trades", "n_days"):
        return f"{int(value)}"
    return f"{value:.4f}"


def metrics_table(scenarios: dict[str, dict]) -> pd.DataFrame:
    """把 {'全样本': {...}, '样本内': {...}} 拼成一张可读的表。"""
    keys: list[str] = []
    for m in scenarios.values():
        for k in m:
            if k not in keys and k in METRIC_LABELS:
                keys.append(k)
    rows = {}
    for name, m in scenarios.items():
        rows[name] = {METRIC_LABELS[k]: format_value(k, m.get(k)) for k in keys}
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    """不依赖 tabulate 的 Markdown 表格（少一个隐形依赖，队友环境也能跑）。"""
    cols = [df.index.name or "指标"] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for name, row in df.iterrows():
        lines.append("| " + " | ".join([str(name)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)
