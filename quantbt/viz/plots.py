# -*- coding: utf-8 -*-
"""净值曲线与回撤图。

作业要求：
- 归一化净值曲线（策略与基准都从 1.0 起算）
- 标注关键买卖点
- 样本内/样本外分段展示

实现要点：
- 买卖点不是把每一笔成交都画出来（那样满屏三角），而是只画"空仓->建仓"和"清仓"两类
  关键事件，图才看得懂。
- 中文字体：Windows 上优先微软雅黑，找不到就退回黑体；再找不到只影响显示不影响计算。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面后端，服务器/命令行都能出图

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.transforms import blended_transform_factory


def _setup_chinese_font() -> None:
    """挑一个系统里真实存在的中文字体，避免图里全是方框。"""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Source Han Sans SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in candidates if name in installed]
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()

COLOR_STRATEGY = "#C0392B"
COLOR_BENCHMARK = "#2C3E50"
COLOR_DD = "#E67E22"


def _key_events(exposure: pd.Series) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """仓位从 0 变正 = 建仓；从正变 0 = 清仓。阈值 1e-6 过滤取整噪声。"""
    prev = exposure.shift(1).fillna(0.0)
    buys = exposure.index[(prev <= 1e-6) & (exposure > 1e-6)]
    sells = exposure.index[(prev > 1e-6) & (exposure <= 1e-6)]
    return buys, sells


def _largest_trades(result, n: int = 8) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """没有明显的建仓/清仓事件时（如月度再平衡），用成交额最大的若干笔当关键买卖点。"""
    trades = getattr(result, "trades", None)
    if trades is None or len(trades) == 0:
        return pd.DatetimeIndex([]), pd.DatetimeIndex([])
    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("amount", ascending=False).head(n)
    buys = pd.DatetimeIndex(df.loc[df["side"] == "BUY", "date"]).sort_values()
    sells = pd.DatetimeIndex(df.loc[df["side"] == "SELL", "date"]).sort_values()
    return buys, sells


def plot_nav(
    result,
    path: str | Path,
    split_date: str | pd.Timestamp | None = None,
    title: str | None = None,
    dpi: int = 160,
) -> Path:
    """上图净值、下图回撤，一张图讲完策略与基准的对比。"""
    # 作业要求：净值必须归一化到 1.0 起算（引擎内部存的是资金量，这里换算）
    nav = result.nav / result.nav.iloc[0]
    bench = result.benchmark / result.benchmark.iloc[0]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # --- 样本内/样本外底纹 ---
    if split_date is not None:
        split = pd.Timestamp(split_date)
        if nav.index[0] < split < nav.index[-1]:
            ax1.axvspan(split, nav.index[-1], color="#F2F2F2", zorder=0)
            ax2.axvspan(split, nav.index[-1], color="#F2F2F2", zorder=0)
            ax1.axvline(split, color="#7F8C8D", linestyle="--", linewidth=1.0)
            ax2.axvline(split, color="#7F8C8D", linestyle="--", linewidth=1.0)
            # 用"数据坐标 x + 轴坐标 y"，把文字放在图底部，避免和图例打架
            trans = blended_transform_factory(ax1.transData, ax1.transAxes)
            ax1.text(nav.index[3], 0.03, "← 样本内（调参）", transform=trans,
                     fontsize=9, color="#777777", va="bottom")
            ax1.text(split + pd.Timedelta(days=10), 0.03, "样本外（检验）→", transform=trans,
                     fontsize=9, color="#777777", va="bottom")

    ax1.plot(nav.index, nav.values, color=COLOR_STRATEGY, linewidth=1.6, label="策略净值")
    ax1.plot(bench.index, bench.values, color=COLOR_BENCHMARK, linewidth=1.2,
             alpha=0.85, label=f"基准（{getattr(result, 'benchmark_name', '沪深300')}）")

    # --- 关键买卖点：优先用建仓/清仓事件；没有就用成交额最大的几笔 ---
    buys, sells = _key_events(result.exposure)
    buy_label, sell_label = "建仓", "清仓"
    if len(buys) + len(sells) < 3:
        buys, sells = _largest_trades(result, n=8)
        buy_label, sell_label = "关键买入（成交额前 8 笔）", "关键卖出（成交额前 8 笔）"

    for idx, label, marker, color in (
        (buys, buy_label, "^", "#27AE60"),
        (sells, sell_label, "v", "#8E44AD"),
    ):
        if len(idx):
            idx = pd.DatetimeIndex(idx).intersection(nav.index)
            if len(idx):
                ax1.scatter(idx, nav.reindex(idx), marker=marker, s=55, color=color,
                            zorder=5, label=label)

    # --- 终值标注 ---
    ax1.annotate(
        f"{nav.iloc[-1]:.2f}", xy=(nav.index[-1], nav.iloc[-1]),
        xytext=(6, 6), textcoords="offset points", color=COLOR_STRATEGY, fontsize=9,
    )
    ax1.annotate(
        f"{bench.iloc[-1]:.2f}", xy=(bench.index[-1], bench.iloc[-1]),
        xytext=(6, -12), textcoords="offset points", color=COLOR_BENCHMARK, fontsize=9,
    )

    ax1.set_ylabel("归一化净值（起点 = 1.0）")
    title = title or f"{getattr(result, 'strategy_label', '') or result.strategy_name} 净值对比"
    ax1.set_title(title, fontsize=13, pad=10)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", fontsize=10, frameon=False)

    # --- 回撤 ---
    for series, color, label in ((nav, COLOR_STRATEGY, "策略回撤"), (bench, COLOR_BENCHMARK, "基准回撤")):
        dd = series / series.cummax() - 1.0
        ax2.fill_between(dd.index, dd.values, 0, color=color, alpha=0.18)
        ax2.plot(dd.index, dd.values, color=color, linewidth=0.9, label=label, alpha=0.9)

    ax2.set_ylabel("回撤")
    ax2.set_ylim(min(-0.05, ax2.get_ylim()[0] * 1.1), 0.01)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower left", fontsize=9, frameon=False, ncol=2)

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=0, ha="center")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_monthly_heatmap(result, path: str | Path, dpi: int = 160) -> Path:
    """月度收益热力图：一眼看出策略在哪些月份/年份吃亏。"""
    ret = result.nav.resample("ME").last().pct_change()
    ret.iloc[0] = result.nav.resample("ME").last().iloc[0] / result.nav.iloc[0] - 1.0
    df = pd.DataFrame({"year": ret.index.year, "month": ret.index.month, "ret": ret.values})
    pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="last")

    fig, ax = plt.subplots(figsize=(11, 3.2))
    data = pivot.values.astype(float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 0.1
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12), [f"{m}月" for m in range(1, 13)])
    ax.set_yticks(range(len(pivot.index)), [str(y) for y in pivot.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center", fontsize=8)
    ax.set_title("策略月度收益热力图", fontsize=12)
    fig.colorbar(im, ax=ax, format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"), shrink=0.8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
