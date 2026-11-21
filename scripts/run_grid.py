# -*- coding: utf-8 -*-
"""参数网格实验工具（任务3 用）。

一条命令跑完一组参数，按样本内/样本外分别报指标，并输出 CSV 方便画热力图。

示例：
    python scripts/run_grid.py --strategy ma_timing \\
        --grid fast=10,20,30 --grid slow=60,90,120 \\
        --grid signal_freq=daily,weekly,monthly

    python scripts/run_grid.py --strategy risk_parity --grid lookback=20,40,60,90,120

输出：output/grid_<strategy>/grid.csv 与 grid.md

**怎么用它做研究（也是报告里"参数与样本处理"那一节该写的内容）：**
1. 先按样本内指标挑参数，再只看样本外的表现；
2. 如果样本外最优参数和样本内最优参数完全对不上，说明参数是在拟合噪声；
3. 看"参数邻域是否平滑"：好策略应当在相邻参数上表现接近，而不是孤岛式的高点。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantbt.config import ASSETS, BENCHMARK_CODE, OUTPUT_DIR, BacktestConfig
from quantbt.data.loader import DataLoader
from quantbt.engine.backtest import BacktestEngine, BacktestResult
from quantbt.engine.metrics import compute_metrics, df_to_markdown, trade_stats
from quantbt.strategy.registry import STRATEGIES, build_strategy


def parse_grid(pairs: list[str]) -> dict[str, list[str]]:
    grid = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--grid 格式应为 key=v1,v2，收到：{item}")
        key, values = item.split("=", 1)
        grid[key.strip()] = [v.strip() for v in values.split(",") if v.strip()]
    return grid


def segment(result: BacktestResult, start, end) -> BacktestResult:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    nav = result.nav[(result.nav.index >= s) & (result.nav.index <= e)]
    bench = result.benchmark[(result.benchmark.index >= s) & (result.benchmark.index <= e)]
    trades = result.trades
    if len(trades):
        dates = pd.to_datetime(trades["date"])
        trades = trades[(dates >= s) & (dates <= e)]
    return BacktestResult(nav=nav, benchmark=bench,
                          exposure=result.exposure.loc[nav.index],
                          weights=result.weights.loc[nav.index], trades=trades,
                          strategy_name=result.strategy_name, params=result.params,
                          config=result.config, initial_cash=result.initial_cash,
                          benchmark_name=result.benchmark_name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="参数网格实验")
    p.add_argument("--strategy", default="ma_timing", choices=list(STRATEGIES))
    p.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2")
    p.add_argument("--start", default=BacktestConfig.start)
    p.add_argument("--end", default=BacktestConfig.end)
    p.add_argument("--split", default=BacktestConfig.split_date)
    args = p.parse_args(argv)

    grid = parse_grid(args.grid)
    if not grid:
        raise SystemExit("至少给一个 --grid，例如 --grid slow=60,90")

    cfg = BacktestConfig(start=args.start, end=args.end, split_date=args.split)
    md = DataLoader(cfg.start, cfg.end).load(list(ASSETS), BENCHMARK_CODE, ASSETS)

    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        strategy = build_strategy(args.strategy, **params)
        res = BacktestEngine(cfg, md, strategy).run()

        full = compute_metrics(res.nav, res.benchmark, cfg.risk_free_annual)
        after = res.nav.index[res.nav.index > pd.Timestamp(cfg.split_date)]
        ins = compute_metrics(segment(res, res.nav.index[0], cfg.split_date).nav,
                              segment(res, res.nav.index[0], cfg.split_date).benchmark,
                              cfg.risk_free_annual)
        oos = compute_metrics(segment(res, after[0], res.nav.index[-1]).nav,
                              segment(res, after[0], res.nav.index[-1]).benchmark,
                              cfg.risk_free_annual)
        stats = trade_stats(res.trades, res.nav)
        rows.append({
            **params,
            "全样本累计": round(full["total_return"], 4),
            "全样本夏普": round(full["sharpe"], 4),
            "样本内累计": round(ins["total_return"], 4),
            "样本内夏普": round(ins["sharpe"], 4),
            "样本外累计": round(oos["total_return"], 4),
            "样本外夏普": round(oos["sharpe"], 4),
            "全样本最大回撤": round(full["max_drawdown"], 4),
            "年化换手": round(stats["turnover_annual"], 2),
        })
        print(f"{params}  样本内 {ins['total_return']:+.1%}  样本外 {oos['total_return']:+.1%}")

    df = pd.DataFrame(rows)
    out_dir = OUTPUT_DIR / f"grid_{args.strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "grid.csv", index=False, encoding="utf-8-sig")

    show = df.sort_values("样本外夏普", ascending=False).head(15)
    (out_dir / "grid.md").write_text(
        "# 参数网格结果（按样本外夏普排序，前 15）\n\n" + df_to_markdown(show) + "\n",
        encoding="utf-8",
    )
    print("\n按样本外夏普排序（前 10）：")
    print(show.head(10).to_string(index=False))
    print(f"\n结果已保存：{out_dir / 'grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
