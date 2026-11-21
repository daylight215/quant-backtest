# -*- coding: utf-8 -*-
"""主程序入口：一条命令跑完"取数 -> 回测 -> 指标 -> 出图 -> 落盘"。

用法示例（在 quant 环境下执行）：

    python run_backtest.py --strategy ma_timing --name baseline
    python run_backtest.py --strategy momentum --param lookback=20 --param top_k=2
    python run_backtest.py --strategy ma_timing --param fast=15 --param slow=45 --name ma_15_45
    python run_backtest.py --strategy buy_hold --name buyhold
    python run_backtest.py --offline --strategy ma_timing   # 断网也能跑（用本地缓存）

产物落在 output/<name>/：
    nav.csv        日期、净值、基准净值、仓位
    trades.csv     每一笔成交
    metrics.json   全样本/样本内/样本外指标
    metrics.md     指标表（可直接贴进报告）
    nav.png        净值曲线 + 回撤（含买卖点、样本内外面板）
    monthly.png    月度收益热力图
    panel.csv      清洗后的行情面板（长表，方便任务3/4复用）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 允许直接 `python run_backtest.py`（把项目根加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantbt.config import ASSETS, BENCHMARK_CODE, BENCHMARK_NAME, OUTPUT_DIR, BacktestConfig
from quantbt.data.loader import DataLoader, MarketData
from quantbt.engine.backtest import BacktestEngine, BacktestResult
from quantbt.engine.metrics import (
    METRIC_LABELS,
    compute_metrics,
    df_to_markdown,
    format_value,
    metrics_table,
    trade_stats,
)
from quantbt.strategy.registry import STRATEGIES, build_strategy
from quantbt.viz.plots import plot_monthly_heatmap, plot_nav


# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="量化回测系统 —— 任务2 工程骨架")
    p.add_argument("--strategy", default="risk_parity", help=f"策略名：{list(STRATEGIES)}")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="策略参数，可重复，如 --param fast=15 --param slow=45")
    p.add_argument("--start", default=BacktestConfig.start, help="回测开始日 YYYY-MM-DD")
    p.add_argument("--end", default=BacktestConfig.end, help="回测结束日 YYYY-MM-DD")
    p.add_argument("--split", default=BacktestConfig.split_date, help="样本内/外切分日")
    p.add_argument("--name", default=None, help="输出目录名，默认 <策略>_<开始>_<结束>")
    p.add_argument("--csv", default=None, help="用任务1同学给的 CSV（长表）而不是联网抓取")
    p.add_argument("--benchmark", default="index", choices=["index", "buyhold"],
                   help="基准：index=沪深300（市场基准）；buyhold=三只股票等权买入持有（剥离选股，检验配置层）")
    p.add_argument("--offline", action="store_true", help="只用本地缓存，不联网")
    p.add_argument("--force-refresh", action="store_true", help="强制重新抓取数据")
    p.add_argument("--no-plot", action="store_true", help="跳过绘图（无图形环境时用）")
    p.add_argument("--list-strategies", action="store_true", help="列出可用策略后退出")
    return p.parse_args(argv)


def parse_params(pairs: list[str]) -> dict:
    out = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--param 格式应为 key=value，收到：{item}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


# --------------------------------------------------------------------------- #
def slice_result(result: BacktestResult, start: str | pd.Timestamp,
                 end: str | pd.Timestamp) -> BacktestResult:
    """取一段区间。

    注意：**不做净值归一化**。所有指标（累计收益、回撤、夏普）都是比值/差分，
    与净值量纲无关；归一化反而会让换手率（成交额 / 平均净值）失真。
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    nav = result.nav[(result.nav.index >= s) & (result.nav.index <= e)]
    bench = result.benchmark[(result.benchmark.index >= s) & (result.benchmark.index <= e)]
    if len(nav) < 2:
        raise ValueError(f"区间 {start} ~ {end} 内数据不足")

    trades = result.trades
    if len(trades):
        mask = (pd.to_datetime(trades["date"]) >= s) & (pd.to_datetime(trades["date"]) <= e)
        trades = trades[mask]

    return BacktestResult(
        nav=nav,
        benchmark=bench,
        exposure=result.exposure.loc[nav.index],
        weights=result.weights.loc[nav.index],
        trades=trades,
        strategy_name=result.strategy_name,
        params=result.params,
        config=result.config,
        initial_cash=result.initial_cash,
        benchmark_name=result.benchmark_name,
        strategy_label=result.strategy_label,
    )


def equal_weight_buyhold(md: MarketData) -> pd.Series:
    """等权买入持有基准：期初每只 1/3，此后权重随涨跌漂移，不再平衡。

    它回答的问题是："如果什么都不做、只是把同一批股票各买 1/3，会怎样？"
    与沪深300 对比说明选股贡献，与它对比说明配置/择时贡献——两个基准一起看，
    才能把超额收益拆开讲清楚。
    """
    px = md.close.dropna(how="all")
    # 期初等权买入后权重随涨跌自然漂移：组合净值 = 各股"归一化价格"的算术平均。
    # 注意不能用 (1 + Σ w_i r_i) 连乘，那等价于每天再平衡回 1/3，会凭空多出再平衡收益。
    nav = (px / px.iloc[0]).mean(axis=1)
    nav.name = "buyhold"
    return nav


def evaluate(result: BacktestResult) -> dict:
    """全样本 + 样本内 + 样本外，各算一套指标。"""
    cfg = result.config
    rf = cfg.risk_free_annual
    periods = cfg.trading_days_per_year

    full = compute_metrics(result.nav, result.benchmark, rf, periods)
    full.update(trade_stats(result.trades, result.nav, periods))

    # 样本外从切分日的下一个交易日算起，避免同一天被两段重复统计
    after_split = result.nav.index[result.nav.index > pd.Timestamp(cfg.split_date)]
    if len(after_split) < 2:
        raise ValueError(f"切分日 {cfg.split_date} 之后数据不足，无法做样本外检验")
    out_start = after_split[0]

    in_r = slice_result(result, result.nav.index[0], cfg.split_date)
    out_r = slice_result(result, out_start, result.nav.index[-1])

    def _seg(r: BacktestResult) -> dict:
        m = compute_metrics(r.nav, r.benchmark, rf, periods)
        m.update(trade_stats(r.trades, r.nav, periods))
        return m

    return {
        "strategy": result.strategy_name,
        "params": result.params,
        "period": {"start": str(result.nav.index[0].date()), "end": str(result.nav.index[-1].date())},
        "split_date": cfg.split_date,
        "full": full,
        "in_sample": _seg(in_r),
        "out_sample": _seg(out_r),
    }


def _benchmark_metrics(metrics: dict) -> dict:
    """从指标字典里挑出基准自身的四项核心指标。"""
    mapping = {
        "total_return": "benchmark_total_return",
        "annual_return": "benchmark_annual_return",
        "max_drawdown": "benchmark_max_drawdown",
        "sharpe": "benchmark_sharpe",
    }
    return {k: metrics.get(v) for k, v in mapping.items() if v in metrics}


def build_report_tables(metrics: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (策略表, 基准对比表)。列 = 全样本/样本内/样本外。"""
    strat = {"全样本": metrics["full"], "样本内": metrics["in_sample"], "样本外": metrics["out_sample"]}
    table = metrics_table(strat)

    bench = {name: _benchmark_metrics(m) for name, m in strat.items()}
    bench_table = metrics_table(bench)
    return table, bench_table


# --------------------------------------------------------------------------- #
def save_outputs(result: BacktestResult, metrics: dict, out_dir: Path, data: MarketData,
                 skip_plot: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    nav_df = pd.DataFrame({
        "date": result.nav.index.strftime("%Y-%m-%d"),
        "nav": result.nav.values.round(2),
        "nav_norm": (result.nav / result.nav.iloc[0]).values.round(6),
        "benchmark": result.benchmark.values.round(6),
        "benchmark_norm": (result.benchmark / result.benchmark.iloc[0]).values.round(6),
        "exposure": result.exposure.values.round(6),
    })
    nav_df.to_csv(out_dir / "nav.csv", index=False, encoding="utf-8-sig")
    result.trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")

    # 行情面板（长表），方便任务3/4直接复用，不必再抓一次数据
    panel = pd.concat(
        [data.close.stack().rename("close"),
         data.open.stack().rename("open"),
         data.volume.stack().rename("volume")],
        axis=1,
    ).reset_index()
    panel.columns = ["trade_date", "code", "close", "open", "volume"]
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.strftime("%Y-%m-%d")
    panel.to_csv(out_dir / "panel.csv", index=False, encoding="utf-8-sig")

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    table, bench_table = build_report_tables(metrics)
    md_lines = [
        f"# 回测指标 —— {metrics['strategy']}",
        "",
        f"- 区间：{metrics['period']['start']} ~ {metrics['period']['end']}",
        f"- 样本内/外切分：{metrics['split_date']}",
        f"- 基准：{result.benchmark_name}",
        "",
        "## 策略指标",
        "",
        df_to_markdown(table),
        "",
        "## 基准指标",
        "",
        df_to_markdown(bench_table),
        "",
    ]
    (out_dir / "metrics.md").write_text("\n".join(md_lines), encoding="utf-8")

    if not skip_plot:
        label = result.strategy_label or result.strategy_name
        plot_nav(result, out_dir / "nav.png", split_date=result.config.split_date,
                 title=f"{label} —— 策略 vs {result.benchmark_name}")
        plot_monthly_heatmap(result, out_dir / "monthly.png")

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": result.strategy_name,
        "params": result.params,
        "config": result.config.to_dict(),
        "data_source": data.report.to_dict("records") if data.report is not None else [],
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def print_summary(result: BacktestResult, metrics: dict) -> None:
    full, bench = metrics["full"], metrics["full"]
    print("\n" + "=" * 72)
    print(f"策略：{result.strategy_label or result.strategy_name}")
    print(f"参数：{result.params}")
    print(f"区间：{result.nav.index[0].date()} ~ {result.nav.index[-1].date()}"
          f"（样本外起点 {result.config.split_date}）")
    print(f"基准：{result.benchmark_name}")
    print("-" * 72)
    print(f"{'指标':<12}{'策略':>10}{'基准':>12}{'超额':>12}")
    rows = [
        ("累计收益率", "total_return", "benchmark_total_return"),
        ("年化收益率", "annual_return", "benchmark_annual_return"),
        ("最大回撤", "max_drawdown", "benchmark_max_drawdown"),
        ("夏普比率", "sharpe", "benchmark_sharpe"),
    ]
    for label, sk, bk in rows:
        sv, bv = full.get(sk), full.get(bk)
        print(f"{label:<12}{format_value(sk, sv):>10}{format_value(sk, bv):>12}"
              f"{format_value(sk, (sv or 0) - (bv or 0)):>12}")
    print("-" * 72)
    print(f"年化波动 {full['annual_vol']:.2%} | 卡玛 {full['calmar']:.2f} | "
          f"信息比率 {full.get('information_ratio', 0):.2f} | Beta {full.get('beta', 0):.2f}")
    print(f"年化换手 {full.get('turnover_annual', 0):.1f}x | 成交 {full.get('n_trades', 0)} 笔 | "
          f"交易成本侵蚀 {full.get('cost_drag_annual', 0):.2%}/年")
    print("=" * 72)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_strategies:
        for name, cls in STRATEGIES.items():
            print(f"- {name}: {cls.__doc__.strip().splitlines()[0] if cls.__doc__ else ''}")
        return 0

    cfg = BacktestConfig(start=args.start, end=args.end, split_date=args.split,
                         csv_path=args.csv)
    params = parse_params(args.param)
    strategy = build_strategy(args.strategy, **params)

    name = args.name or f"{args.strategy}_{args.start}_{args.end}".replace("-", "")
    out_dir = OUTPUT_DIR / name

    print(f"[1/4] 加载数据 ……（{', '.join(ASSETS)}）")
    loader = DataLoader(cfg.start, cfg.end, adjust=cfg.price_adjust,
                        force_refresh=args.force_refresh, offline=args.offline)
    md = loader.load(list(ASSETS), benchmark_code=BENCHMARK_CODE, names=ASSETS,
                     csv_path=cfg.csv_path)
    if args.benchmark == "buyhold":
        md.benchmark = equal_weight_buyhold(md)
        md.benchmark_name = "等权买入持有（3 只股票）"
    print(f"      数据源：{loader.last_source}，{len(md.dates)} 个交易日，"
          f"{md.close.index[0].date()} ~ {md.close.index[-1].date()}")

    print(f"[2/4] 生成信号并回测 ……（{strategy.describe()}）")
    result = BacktestEngine(cfg, md, strategy).run()

    print("[3/4] 计算指标 ……")
    metrics = evaluate(result)

    print("[4/4] 保存结果 ……")
    save_outputs(result, metrics, out_dir, md, skip_plot=args.no_plot)

    print_summary(result, metrics)
    print(f"输出目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
