"""Rational expectations and efficient-market tests — docs/06_efficient_markets.md."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def _clean(r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 10:
        raise ValueError("need at least 10 observations")
    return r


# ---------------------------------------------------------------- §6.1 expectations
def adaptive_expectation(series: pd.Series, lam: float) -> pd.Series:
    """π^e = (1−λ)·Σ λ^j π_{t−j}, i.e. e_t = λ·e_{t−1} + (1−λ)·x_t — docs/06 §6.1.

    Value at t is the forecast for t+1 formed with data up to t.
    """
    if not 0 < lam < 1:
        raise ValueError("lam must be in (0, 1)")
    return series.ewm(alpha=1 - lam, adjust=False).mean()


def expectations_demo(t: int = 300, phi: float = 0.7, mu: float = 0.02, sigma: float = 0.01, lam: float = 0.7,
                      seed: int = 42) -> pd.DataFrame:
    """Adaptive vs rational forecasts of a mean-reverting AR(1) series — docs/06 §6.1.

    x_t = μ + φ(x_{t−1} − μ) + ε_t. The rational (optimal) forecast of x_{t+1} uses the true
    law of motion, μ + φ(x_t − μ); the adaptive forecast is :func:`adaptive_expectation`.
    Rows are aligned so that each forecast sits next to the value it predicts.
    """
    rng = np.random.default_rng(seed)
    x = np.empty(t)
    x[0] = mu
    for i in range(1, t):
        x[i] = mu + phi * (x[i - 1] - mu) + rng.normal(0, sigma)
    s = pd.Series(x, name="actual")
    out = pd.DataFrame({
        "actual": s,
        "adaptive": adaptive_expectation(s, lam).shift(1),
        "rational": (mu + phi * (s - mu)).shift(1),
    }).iloc[1:]
    out["adaptive_error"] = out["actual"] - out["adaptive"]
    out["rational_error"] = out["actual"] - out["rational"]
    return out


def forecast_error_stats(demo: pd.DataFrame) -> pd.DataFrame:
    """Mean, RMSE and lag-1 autocorrelation of each forecast error (rational ⇒ mean ≈ 0, unpredictable)."""
    rows = {}
    for col in ("adaptive", "rational"):
        e = demo[f"{col}_error"].dropna()
        rows[col] = {"mean_error": e.mean(), "rmse": float(np.sqrt((e ** 2).mean())), "error_rho_1": e.autocorr(1)}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- §6.7 weak-form tests
def autocorr_test(r, lags: int = 10) -> pd.DataFrame:
    """ρ_k with its N(0, 1/T) z-stat and the Ljung–Box Q and p-value up to each lag — docs/06 §6.7."""
    x = _clean(r)
    rho = sm.tsa.acf(x, nlags=lags, fft=False)[1:]
    lb = acorr_ljungbox(x, lags=lags)
    return pd.DataFrame({
        "rho": rho,
        "z": rho * np.sqrt(x.size),
        "lb_stat": lb["lb_stat"].to_numpy(),
        "lb_pvalue": lb["lb_pvalue"].to_numpy(),
    }, index=pd.RangeIndex(1, lags + 1, name="lag"))


def runs_test(r) -> dict:
    """Wald–Wolfowitz runs test on the signs of returns (zeros dropped) — docs/06 §6.7."""
    x = _clean(r)
    signs = np.sign(x[x != 0])
    n_pos, n_neg = int((signs > 0).sum()), int((signs < 0).sum())
    n = n_pos + n_neg
    runs = int(1 + np.count_nonzero(np.diff(signs)))
    expected = 2 * n_pos * n_neg / n + 1
    var = 2 * n_pos * n_neg * (2 * n_pos * n_neg - n) / (n ** 2 * (n - 1))
    z = (runs - expected) / np.sqrt(var)
    return {"runs": runs, "expected": float(expected), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def variance_ratio(r, q: int) -> dict:
    """Lo–MacKinlay VR(q) with overlapping q-sums and the homoskedastic z-stat — docs/06 §6.7.

    VR > 1 suggests momentum, VR < 1 mean reversion; VR ≈ 1 under a random walk.
    """
    x = _clean(r)
    if q < 2:
        raise ValueError("q must be >= 2")
    t = x.size
    mu = x.mean()
    var_1 = ((x - mu) ** 2).sum() / (t - 1)
    sums_q = np.convolve(x, np.ones(q), mode="valid")
    m = q * (t - q + 1) * (1 - q / t)  # m includes the factor q, so var_q is already per period
    var_q = ((sums_q - q * mu) ** 2).sum() / m
    vr = var_q / var_1
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * t))
    return {"q": q, "vr": float(vr), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def weak_form_battery(returns: pd.DataFrame, lags: int = 10, q: int = 5) -> pd.DataFrame:
    """Autocorrelation, Ljung–Box, runs and variance-ratio tests for each column — docs/06 §6.9."""
    rows = {}
    for name, r in returns.items():
        ac, runs, vr = autocorr_test(r, lags), runs_test(r), variance_ratio(r, q)
        rows[name] = {
            "T": int(r.notna().sum()),
            "rho_1": ac["rho"].iloc[0],
            "rho_1_z": ac["z"].iloc[0],
            f"LB_Q({lags})": ac["lb_stat"].iloc[-1],
            "LB_p": ac["lb_pvalue"].iloc[-1],
            "runs_z": runs["z"],
            "runs_p": runs["p"],
            f"VR({q})": vr["vr"],
            "VR_z": vr["z"],
            "VR_p": vr["p"],
        }
    out = pd.DataFrame(rows).T
    out["reject_5%"] = (out[["LB_p", "runs_p", "VR_p"]] < 0.05).any(axis=1)
    return out


# ---------------------------------------------------------------- trading-rule backtest
def ma_crossover_backtest(prices: pd.Series, short: int = 20, long: int = 100, cost_bp: float = 10) -> pd.DataFrame:
    """Long when MA(short) > MA(long), else cash (0% return), vs buy-and-hold — docs/06 §6.7.

    The signal at the close of day t sets the position for day t+1 (no look-ahead).
    ``cost_bp`` is charged on each change of position. The test starts once MA(long) exists.
    """
    p = prices.dropna()
    ma_s, ma_l = p.rolling(short).mean(), p.rolling(long).mean()
    signal = (ma_s > ma_l).astype(float)
    out = pd.DataFrame({"price": p, "ma_short": ma_s, "ma_long": ma_l, "signal": signal}).iloc[long - 1:]
    out["position"] = out["signal"].shift(1).fillna(0.0)
    out["ret_bh"] = out["price"].pct_change().fillna(0.0)
    out["trade"] = out["position"].diff().abs().fillna(out["position"].iloc[0])
    out["ret_gross"] = out["position"] * out["ret_bh"]
    out["ret_net"] = out["ret_gross"] - out["trade"] * cost_bp / 10_000
    for col in ("bh", "gross", "net"):
        out[f"eq_{col}"] = (1 + out[f"ret_{col}"]).cumprod()
    return out


def performance_summary(backtest: pd.DataFrame, periods_per_year: float = 252) -> pd.DataFrame:
    """CAGR, volatility, return/vol, max drawdown and trades for buy-and-hold vs the rule (gross, net)."""
    rows = {}
    years = (len(backtest) - 1) / periods_per_year
    for col, label in (("bh", "buy & hold"), ("gross", "MA rule (no cost)"), ("net", "MA rule (after cost)")):
        eq, r = backtest[f"eq_{col}"], backtest[f"ret_{col}"]
        vol = r.std(ddof=1) * np.sqrt(periods_per_year)
        cagr = eq.iloc[-1] ** (1 / years) - 1
        rows[label] = {
            "CAGR": cagr,
            "vol": vol,
            "return/vol": cagr / vol if vol > 0 else np.nan,
            "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "trades": int(backtest["trade"].sum()) if col != "bh" else 0,
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- semi-strong: event study
def event_study(asset_ret: pd.Series | pd.DataFrame, market_ret: pd.Series, event_dates: Iterable,
                est: tuple[int, int] = (-250, -30), win: tuple[int, int] = (-10, 10)) -> dict:
    """Market-model event study — docs/06 §6.7.

    1. Fit R_i = a + b·R_m on trading days ``est`` relative to each event (day 0 = first
       trading day on/after the event date).
    2. AR_t = R_i − (â + b̂·R_m) over ``win``; CAR = Σ AR; CAAR = mean CAR across events.
    3. t(CAAR) = CAAR / (σ̂_AR·√L/√N), L = days accumulated so far.

    ``asset_ret`` is a Series with ``event_dates`` as dates, or a DataFrame with
    ``event_dates`` as ``(column, date)`` pairs. Events without a full window are skipped.
    """
    days = np.arange(win[0], win[1] + 1)
    ars, sigmas, used = {}, [], []
    for event in event_dates:
        col, date = event if isinstance(asset_ret, pd.DataFrame) else (None, event)
        series = asset_ret[col] if col is not None else asset_ret
        df = pd.concat({"y": series, "x": market_ret}, axis=1).dropna()
        pos = int(df.index.searchsorted(pd.Timestamp(date)))
        if pos + est[0] < 0 or pos + win[1] >= len(df):
            continue
        fit_df = df.iloc[pos + est[0]: pos + est[1] + 1]
        b, a = np.polyfit(fit_df["x"], fit_df["y"], 1)
        resid = fit_df["y"] - (a + b * fit_df["x"])
        window = df.iloc[pos + win[0]: pos + win[1] + 1]
        key = (col, df.index[pos]) if col is not None else df.index[pos]
        ars[key] = pd.Series((window["y"] - (a + b * window["x"])).to_numpy(), index=days)
        sigmas.append(resid.std(ddof=2))
        used.append(key)
    if not ars:
        raise ValueError("no event has enough data for the estimation and event windows")
    ar = pd.DataFrame(ars).rename_axis(index="day")
    car = ar.cumsum()
    caar = car.mean(axis=1).rename("CAAR")
    sigma = float(np.sqrt(np.mean(np.square(sigmas))))
    n = ar.shape[1]
    t_caar = (caar / (sigma * np.sqrt(np.arange(1, len(days) + 1)) / np.sqrt(n))).rename("t")
    return {"ar": ar, "car": car, "aar": ar.mean(axis=1).rename("AAR"), "caar": caar, "t_caar": t_caar,
            "sigma_ar": sigma, "n_events": n, "events": used}


# ---------------------------------------------------------------- synthetic data (tests §6.7)
def simulate_returns(kind: str = "iid", t: int = 5000, sigma: float = 0.01, phi: float = 0.3,
                     seed: int = 42) -> np.ndarray:
    """iid N(0, σ²) returns, or an AR(1) r_t = φ·r_{t−1} + ε_t — docs/06 §6.7."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, t)
    if kind == "iid":
        return eps
    if kind == "ar1":
        r = np.empty(t)
        r[0] = eps[0]
        for i in range(1, t):
            r[i] = phi * r[i - 1] + eps[i]
        return r
    raise ValueError("kind must be 'iid' or 'ar1'")


def simulate_event_data(n_days: int = 10_000, n_events: int = 30, jump: float = 0.05, beta: float = 1.1,
                        seed: int = 42) -> tuple[pd.Series, pd.Series, list[pd.Timestamp]]:
    """Market-model returns with a +``jump`` abnormal return injected on each event day — docs/06 §6.7.

    Events are spaced more than 260 days apart, so no estimation window contains another event.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    market = pd.Series(rng.normal(0.0004, 0.01, n_days), index=idx, name="market")
    asset = pd.Series(0.0001 + beta * market.to_numpy() + rng.normal(0.0, 0.012, n_days), index=idx, name="asset")
    positions = np.linspace(300, n_days - 50, n_events).astype(int)
    asset.iloc[positions] += jump
    return asset, market, list(idx[positions])
