# -*- coding: utf-8 -*-
"""数据加载器：缓存、抓取、清洗、面板化，一次到位。

对外只暴露 `DataLoader.load()`，返回统一的 `MarketData` 对象：
    md.close      收盘价宽表（index=日期, columns=代码）
    md.open       开盘价宽表
    md.volume     成交量宽表
    md.benchmark  基准指数收盘（Series）
    md.report     数据体检表（DataFrame，可直接写进文档）

缓存策略：
    data/raw/stock_600519.csv、data/raw/index_000300.csv
    缓存覆盖所需区间就直接读，缺失才联网。`force_refresh=True` 强制重抓。
    断网时用 `offline=True`，只读缓存，保证答辩现场一定能跑。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import RAW_DIR
from . import clean, sources


@dataclass
class MarketData:
    """回测所需的最小行情集合。"""

    close: pd.DataFrame
    open: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series
    names: dict[str, str]
    report: pd.DataFrame | None = None
    # 回测正式起点：行情会多留一段预热数据（均线/动量需要），但净值从这个日期算
    trade_start: pd.Timestamp | None = None
    benchmark_name: str = "沪深300"

    @property
    def codes(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    def slice(self, start: str, end: str) -> "MarketData":
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        mask = (self.close.index >= start) & (self.close.index <= end)
        bench = self.benchmark[(self.benchmark.index >= start) & (self.benchmark.index <= end)]
        rep = self.report
        return MarketData(
            close=self.close.loc[mask],
            open=self.open.loc[mask],
            volume=self.volume.loc[mask],
            benchmark=bench,
            names=self.names,
            report=rep,
            trade_start=max(start, self.trade_start) if self.trade_start is not None else start,
        )


def _read_market_csv(path: Path) -> pd.DataFrame:
    """读任务1同学给的 CSV，自动兼容 UTF-8 / GBK 编码。"""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码：{path}")


class DataLoader:
    """统一入口。"""

    def __init__(
        self,
        start: str,
        end: str,
        adjust: str = "qfq",
        cache_dir: Path | str = RAW_DIR,
        force_refresh: bool = False,
        offline: bool = False,
        warmup_days: int = 450,
    ) -> None:
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        # 多抓一年多的历史，保证 MA200 / 动量 在回测起点就已经有值
        self.fetch_start = self.start - pd.Timedelta(days=warmup_days)
        self.adjust = adjust
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_refresh = force_refresh
        self.offline = offline
        self.last_source = "cache"

    # ------------------------------------------------------------------ #
    # 单标的数据
    # ------------------------------------------------------------------ #
    def _cache_path(self, kind: str, code: str) -> Path:
        return self.cache_dir / f"{kind}_{code}.csv"

    def _load_cached(self, kind: str, code: str, min_date: pd.Timestamp) -> pd.DataFrame | None:
        path = self._cache_path(kind, code)
        if not path.exists():
            return None
        df = pd.read_csv(path, parse_dates=["date"])
        if len(df) == 0:
            return None

        # 覆盖区间检查：
        # 起点用 20 天容差——fetch_start 是自然日，而行情首日是交易日，中间隔着周末/假期；
        # 终点要求覆盖到 end（若 end 是未来日期，则以今天为准，避免永远重抓）。
        if df["date"].min() > min_date + pd.Timedelta(days=20):
            return None
        effective_end = min(self.end, pd.Timestamp.today())
        if df["date"].max() < effective_end - pd.Timedelta(days=10):
            return None
        return df

    def _get_stock(self, code: str) -> pd.DataFrame:
        cached = None if self.force_refresh else self._load_cached("stock", code, self.fetch_start)
        if cached is not None:
            return cached
        if self.offline:
            raise FileNotFoundError(
                f"离线模式下没有 {code} 的可用缓存（需覆盖 {self.fetch_start.date()} ~ "
                f"{self.end.date()}）：{self._cache_path('stock', code)}\n"
                f"请先联网跑一次，或去掉 --offline。"
            )
        df = sources.fetch_stock(code, self.fetch_start.strftime("%Y-%m-%d"),
                                 self.end.strftime("%Y-%m-%d"), adjust=self.adjust)
        df.to_csv(self._cache_path("stock", code), index=False, encoding="utf-8-sig")
        self.last_source = "akshare"
        return df

    def _get_index(self, code: str) -> pd.DataFrame:
        cached = None if self.force_refresh else self._load_cached("index", code, self.fetch_start)
        if cached is not None:
            return cached
        if self.offline:
            raise FileNotFoundError(f"离线模式下没有基准 {code} 的缓存")
        df = sources.fetch_index(code, self.fetch_start.strftime("%Y-%m-%d"),
                                 self.end.strftime("%Y-%m-%d"))
        df.to_csv(self._cache_path("index", code), index=False, encoding="utf-8-sig")
        self.last_source = "akshare"
        return df

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def load(
        self,
        codes: list[str],
        benchmark_code: str = "000300",
        names: dict[str, str] | None = None,
        csv_path: str | Path | None = None,
    ) -> MarketData:
        names = dict(names or {})

        if csv_path:
            return self._load_from_csv(csv_path, codes, benchmark_code, names)

        frames: list[tuple[str, pd.DataFrame]] = []
        reports = []
        for code in codes:
            df = self._get_stock(code)
            reports.append(clean.audit(df, code).as_row())
            frames.append((code, df))

        bench_df = self._get_index(benchmark_code)
        reports.append(clean.audit(bench_df, benchmark_code).as_row())

        close = clean.to_wide(frames, "close")
        open_ = clean.to_wide(frames, "open")
        volume = clean.to_wide(frames, "volume")
        bench = bench_df.set_index("date")["close"].sort_index()
        bench.name = benchmark_code

        aligned = clean.align_panel(
            {"close": close, "open": open_, "volume": volume}, extra_dates=bench.index
        )
        bench = bench.reindex(aligned["close"].index).ffill()

        # 报告里再加一列涨跌幅异常提示
        for msg in clean.sanity_check(aligned["close"]):
            reports.append({"代码": "-", "行数": "-", "起始日": "-", "结束日": "-",
                            "重复日期": "-", "零成交日": "-", "缺失收盘": "-", "备注": msg})

        md = MarketData(
            close=aligned["close"],
            open=aligned["open"],
            volume=aligned["volume"],
            benchmark=bench.dropna(),
            names={c: names.get(c, c) for c in codes},
            report=pd.DataFrame(reports),
            trade_start=self.start,
        )
        # 注意：这里返回的是"含预热期"的完整行情，回测区间由引擎按 config.start 截取
        return md

    def _load_from_csv(
        self, csv_path: str | Path, codes: list[str], benchmark_code: str,
        names: dict[str, str],
    ) -> MarketData:
        """用任务1同学提供的长表数据。字段：trade_date, code, name, open, high, low, close, volume。"""
        raw = _read_market_csv(Path(csv_path))
        fields = clean.pivot_long_to_fields(raw)
        if names:
            fields["names"] = {**fields.get("names", {}), **names}

        missing = [c for c in codes if c not in fields["close"].columns]
        if missing:
            raise ValueError(f"CSV 中缺少标的 {missing}")
        if benchmark_code not in fields["close"].columns:
            raise ValueError(f"CSV 中缺少基准 {benchmark_code}")

        sub = {k: fields[k][codes] for k in ("close", "open", "volume")}
        bench = fields["close"][benchmark_code]
        aligned = clean.align_panel(sub, extra_dates=bench.index)
        bench = bench.reindex(aligned["close"].index).ffill()

        reports = []
        for code in codes:
            df = pd.DataFrame({
                "date": fields["close"].index,
                "close": fields["close"][code].values,
                "volume": fields["volume"][code].values if "volume" in fields else 0,
            })
            reports.append(clean.audit(df, code).as_row())

        md = MarketData(
            close=aligned["close"],
            open=aligned["open"],
            volume=aligned["volume"],
            benchmark=bench.dropna(),
            names={c: fields["names"].get(c, names.get(c, c)) for c in codes},
            report=pd.DataFrame(reports),
            trade_start=self.start,
        )
        self.last_source = "csv"
        return md
