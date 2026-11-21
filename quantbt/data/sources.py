# -*- coding: utf-8 -*-
"""数据源适配：把不同接口的返回统一成标准 OHLCV 表。

统一后的列：date, open, high, low, close, volume（date 为 datetime64）。

为什么准备两个源？
    实测同一台机器上 akshare 的东财接口（stock_zh_a_hist / index_zh_a_hist）
    会间歇性 RemoteDisconnected，而新浪接口（stock_zh_a_daily / stock_zh_index_daily）
    稳定。生产化的数据层必须能降级，否则演示当天抓不到数据就翻车。
"""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

# 东财接口的中文列名 -> 标准列名
_EM_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
}

# 新浪接口本身就是英文列名，只需要挑出需要的列
_STD_COLS = ["date", "open", "high", "low", "close", "volume"]


def _prefix(code: str) -> str:
    """A 股 6 位代码 -> 交易所前缀（新浪/腾讯接口需要）。"""
    if code.startswith(("5", "6", "9")):
        return "sh"
    if code.startswith(("0", "2", "3")):
        return "sz"
    if code.startswith(("4", "8")):
        return "bj"
    raise ValueError(f"无法判断交易所：{code}")


def _index_symbol(code: str) -> str:
    """指数代码 -> 带前缀的完整代码。

    注意：指数不能套用个股规则。沪深300 是 000300 但挂在**上交所**（sh000300），
    深证成指 399001 才是 sz。用个股的 _prefix 会把 000300 错判成 sz000300。
    """
    if code.startswith("399"):
        return f"sz{code}"
    return f"sh{code}"


def _standardize(df: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """转标准列、转日期、排序去重。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_STD_COLS)
    df = df.rename(columns=mapping or {})
    missing = [c for c in _STD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"数据源缺少字段 {missing}，实际字段 {list(df.columns)}")
    out = df[_STD_COLS].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = (
        out.dropna(subset=["close"])
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def _with_retry(fn: Callable, *args, retries: int = 3, wait: float = 2.0, **kwargs):
    """简单重试：网络接口偶发断连时多试几次。"""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 数据源异常类型不可控
            last_err = exc
            if attempt < retries - 1:
                time.sleep(wait * (attempt + 1))
    raise RuntimeError(f"数据接口连续 {retries} 次失败：{last_err}")


def fetch_stock(code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """抓单只 A 股日线：东财优先，新浪兜底。"""
    import akshare as ak

    s, e = pd.Timestamp(start), pd.Timestamp(end)

    # 源 1：东方财富（字段是中文，成交量单位=手）
    def _em():
        raw = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=s.strftime("%Y%m%d"),
            end_date=e.strftime("%Y%m%d"),
            adjust=adjust,
        )
        return _standardize(raw, _EM_MAP)

    # 源 2：新浪（全历史，需要自己截区间）
    def _sina():
        raw = ak.stock_zh_a_daily(symbol=f"{_prefix(code)}{code}", adjust=adjust)
        df = _standardize(raw)
        return df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)

    errors = []
    for source in (_em, _sina):
        try:
            df = _with_retry(source, retries=2, wait=1.5)
            if len(df) > 0:
                return df
            errors.append(f"{source.__name__} 返回空表")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source.__name__}: {exc}")
    raise RuntimeError(f"{code} 所有数据源均失败：{'; '.join(errors)}")


def fetch_index(code: str, start: str, end: str) -> pd.DataFrame:
    """抓指数日线（沪深300 等）。新浪源最稳，无需复权。"""
    import akshare as ak

    s, e = pd.Timestamp(start), pd.Timestamp(end)

    def _sina():
        raw = ak.stock_zh_index_daily(symbol=_index_symbol(code))
        df = _standardize(raw)
        return df[(df["date"] >= s) & (df["date"] <= e)].reset_index(drop=True)

    # 备用：东财指数接口（部分网络环境下可用）
    def _em():
        raw = ak.index_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=s.strftime("%Y%m%d"),
            end_date=e.strftime("%Y%m%d"),
        )
        return _standardize(raw, _EM_MAP)

    errors = []
    for source in (_sina, _em):
        try:
            df = _with_retry(source, retries=2, wait=1.5)
            if len(df) > 0:
                return df
            errors.append(f"{source.__name__} 返回空表")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source.__name__}: {exc}")
    raise RuntimeError(f"指数 {code} 抓取失败：{'; '.join(errors)}")
