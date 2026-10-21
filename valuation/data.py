"""Data loaders and cache — docs/07 §7.4.

Order for every loader: local cache (``data/raw``) → download → synthetic data with a
fixed seed and a printed banner, so the notebook always runs.

Market data comes from yfinance. ``^SET.BK`` (SET index) and the SET TRI are not
available there, so ``TDEX.BK`` (ThaiDEX SET50 ETF, adjusted close) is the market proxy.
Adjusted close includes dividends, so it is an approximate total-return series.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .spreads import read_spread_file

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SPREAD_TTM_FILE = DATA_DIR / "CorporateAverageSpread(ttm)_2026-08-26.xlsx"
SPREAD_DURATION_FILE = DATA_DIR / "CorporateAverageSpread(duration)_2026-08-26.xlsx"
BOT_RATES_FILE = RAW_DIR / "bot_rates.csv"

MARKET_PROXY = "TDEX.BK"
DEFAULT_STOCKS = ("PTT.BK", "PTTEP.BK", "ADVANC.BK", "AOT.BK", "CPALL.BK", "KBANK.BK", "BBL.BK", "SCC.BK", "BDMS.BK")
FIELDS = ("adj_close", "close", "dividends")


def banner(message: str) -> None:
    """Print a hard-to-miss notice (used whenever synthetic data replaces real data)."""
    line = "!" * 78
    print(f"{line}\n!! {message}\n{line}")


def load_spread_tables(basis: str = "ttm") -> dict[str, pd.DataFrame]:
    """ThaiBMA spread tables (``spread``, ``std`` in bp; ``maturity`` in years) by ``ttm`` or ``duration`` bucket."""
    return read_spread_file(SPREAD_TTM_FILE if basis == "ttm" else SPREAD_DURATION_FILE)


# ---------------------------------------------------------------- market data
@dataclass
class MarketData:
    adj_close: pd.DataFrame
    close: pd.DataFrame
    dividends: pd.DataFrame
    source: str  # "cache", "yfinance" or "synthetic"


def _cache_paths(name: str) -> dict[str, Path]:
    return {f: RAW_DIR / f"{name}_{f}.parquet" for f in FIELDS}


def _download(tickers: Sequence[str], start: str, end: str, market: str = MARKET_PROXY,
              attempts: int = 3) -> MarketData:
    """yfinance download with retries; tickers still empty afterwards are dropped (the market proxy is required)."""
    import yfinance as yf

    frames: dict[str, pd.DataFrame] = {}
    todo = list(tickers)
    for _ in range(attempts):
        raw = yf.download(todo, start=start, end=end, auto_adjust=False, actions=True, progress=False, threads=False)
        got = [t for t in todo if t in raw["Adj Close"] and raw["Adj Close"][t].notna().any()]
        for field, col in (("adj_close", "Adj Close"), ("close", "Close"), ("dividends", "Dividends")):
            part = raw[col][got]
            frames[field] = part if field not in frames else frames[field].join(part, how="outer")
        todo = [t for t in todo if t not in got]
        if not todo:
            break
    if market in todo or len(todo) == len(tickers) or "adj_close" not in frames:
        raise RuntimeError(f"no data for {todo}")
    if todo:
        banner(f"No yfinance data for {todo}; continuing without them.")
    kept = [t for t in tickers if t not in todo]
    frames = {k: v[kept] for k, v in frames.items()}
    frames["dividends"] = frames["dividends"].fillna(0.0)
    return MarketData(**frames, source="yfinance")


def synthetic_market_data(tickers: Sequence[str], start: str, end: str, market: str = MARKET_PROXY,
                          seed: int = 42) -> MarketData:
    """Market-model prices with random betas and semi-annual dividends (fixed seed)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end, inclusive="left")
    n = len(idx)
    mkt = rng.normal(0.0003, 0.011, n)
    rets = {}
    for t in tickers:
        if t == market:
            rets[t] = mkt
        else:
            beta = rng.uniform(0.6, 1.4)
            rets[t] = 0.0001 + beta * mkt + rng.normal(0.0, 0.014, n)
    adj = pd.DataFrame({t: 30 * np.exp(np.cumsum(r)) for t, r in rets.items()}, index=idx)
    divs = pd.DataFrame(0.0, index=idx, columns=adj.columns)
    pay_days = idx[(idx.month.isin([3, 9])) & (idx.day <= 7)].to_series().groupby(lambda d: (d.year, d.month)).first()
    divs.loc[pay_days.to_numpy()] = adj.loc[pay_days.to_numpy()].to_numpy() * 0.02
    return MarketData(adj_close=adj, close=adj.copy(), dividends=divs, source="synthetic")


def load_market_data(tickers: Sequence[str] = DEFAULT_STOCKS + (MARKET_PROXY,), start: str = "2015-01-01",
                     end: str = "2026-10-01", name: str = "set_daily", refresh: bool = False,
                     offline: bool = False) -> MarketData:
    """Daily adjusted close, close and dividends (cache → yfinance → synthetic) — docs/07 §7.4."""
    paths = _cache_paths(name)
    if not refresh and all(p.exists() for p in paths.values()):
        frames = {f: pd.read_parquet(p) for f, p in paths.items()}
        if set(tickers) <= set(frames["adj_close"].columns):
            frames = {k: v[list(tickers)] for k, v in frames.items()}
            return MarketData(**frames, source="cache")
    if not offline:
        try:
            data = _download(tickers, start, end)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            for f, p in paths.items():
                getattr(data, f).to_parquet(p)
            return data
        except Exception as exc:  # network blocked, ticker gone, …
            banner(f"Download failed ({exc.__class__.__name__}: {exc}).")
    banner("SYNTHETIC MARKET DATA (seed 42) — results are illustrative, not real SET prices.")
    return synthetic_market_data(tickers, start, end)


def trailing_dps(dividends: pd.Series, asof=None, days: int = 365) -> float:
    """Sum of dividends per share with ex-date in the ``days`` before ``asof`` (inclusive)."""
    asof = pd.Timestamp(asof) if asof is not None else dividends.index.max()
    window = dividends.loc[(dividends.index > asof - pd.Timedelta(days=days)) & (dividends.index <= asof)]
    return float(window.sum())


def annual_dps(dividends: pd.Series) -> pd.Series:
    """Dividends per share by calendar year of the ex-date."""
    d = dividends[dividends > 0]
    return d.groupby(d.index.year).sum().rename("dps")


def ex_dividend_events(dividends: pd.DataFrame) -> list[tuple[str, pd.Timestamp]]:
    """``(ticker, ex_date)`` pairs for every positive dividend — input for ``emh.event_study``."""
    return [(t, d) for t in dividends.columns for d in dividends.index[dividends[t] > 0]]


# ---------------------------------------------------------------- Thai rates (BOT)
BOT_COLUMNS = ("policy", "interbank", "gov_3m", "gov_1y", "gov_10y", "mlr", "savings", "cpi_yoy")


def synthetic_bot_rates(start: str = "2005-01-31", end: str = "2026-08-31", seed: int = 42) -> pd.DataFrame:
    """Illustrative monthly rate panel (decimals) with the qualitative structure of docs/01 §1.4.

    The policy rate is a mean-reverting walk on a 25bp grid; other series are built on top
    of it with typical premia. It is **not** Bank of Thailand data.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="ME")
    policy = np.empty(len(idx))
    policy[0] = 0.0225
    for i in range(1, len(idx)):
        drift = 0.08 * (0.02 - policy[i - 1]) + rng.normal(0, 0.0012)
        policy[i] = np.clip(np.round((policy[i - 1] + drift) / 0.0025) * 0.0025, 0.005, 0.05)
    noise = lambda s: rng.normal(0, s, len(idx))  # noqa: E731
    gov_10y = policy + 0.012 + np.cumsum(noise(0.0008)) * 0.3
    cpi = np.empty(len(idx))
    cpi[0] = 0.02
    for i in range(1, len(idx)):
        cpi[i] = 0.02 + 0.9 * (cpi[i - 1] - 0.02) + rng.normal(0, 0.004)
    return pd.DataFrame({
        "policy": policy,
        "interbank": policy - 0.0005 + noise(0.0005),
        "gov_3m": policy - 0.0002 + noise(0.0008),
        "gov_1y": policy + 0.001 + noise(0.001),
        "gov_10y": gov_10y,
        "mlr": policy + 0.045 + noise(0.0005),
        "savings": np.maximum(policy - 0.012, 0.0025),
        "cpi_yoy": cpi,
    }, index=idx.rename("date"))


def load_bot_rates(path: Path = BOT_RATES_FILE) -> tuple[pd.DataFrame, str]:
    """Monthly Thai rates (decimals) from ``data/raw/bot_rates.csv`` — docs/01 §1.4, docs/07 §7.4.

    Expected CSV: a ``date`` column plus any of ``BOT_COLUMNS`` in **percent** (as the BOT
    publishes them). The BOT API needs a registered key, so this is not downloaded here;
    without the file, synthetic data is returned with a banner.
    """
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return df[[c for c in BOT_COLUMNS if c in df.columns]] / 100, "file"
    banner(f"SYNTHETIC THAI RATES (seed 42) — put BOT data in {path.relative_to(ROOT)} to use real data.")
    return synthetic_bot_rates(), "synthetic"
