"""Return measures and portfolio risk — docs/05 §5.1, §5.3."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .bonds import holding_period_return  # noqa: F401  (R = (P1 − P0 + D1)/P0, docs/05 §5.1)
from .tvm import annualize


def simple_returns(prices: pd.DataFrame | pd.Series):
    """R_t = P_t/P_{t−1} − 1 — docs/05 §5.1."""
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series):
    """r_t = ln(P_t/P_{t−1}); additive over time — docs/05 §5.1."""
    return np.log(prices).diff().iloc[1:]


def period_returns(prices: pd.DataFrame | pd.Series, rule: str = "ME", log: bool = False):
    """Returns over calendar periods (``"W-FRI"``, ``"ME"``…) from the last price in each period."""
    last = prices.resample(rule).last()
    return log_returns(last) if log else simple_returns(last)


def arithmetic_mean(r: ArrayLike) -> float:
    """(1/T)·Σ R_t — docs/05 §5.1."""
    return float(np.nanmean(np.asarray(r, dtype=float)))


def geometric_mean(r: ArrayLike) -> float:
    """[Π(1 + R_t)]^(1/T) − 1 — docs/05 §5.1."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        raise ValueError("returns must not be empty")
    return float(np.exp(np.log1p(r).mean()) - 1)


def per_period_rate(annual_rate, periods_per_year: float):
    """Convert an annual rate to a per-period rate: (1 + R)^(1/c) − 1 (e.g. rf for a monthly regression)."""
    return annualize(annual_rate, 1 / periods_per_year)


def excess_returns(r, rf=0.0):
    """R_t − r_{f,t} — docs/05 §5.1."""
    return r - rf


def leveraged_return(rp: float, vb_ve: float, rd: float) -> float:
    """R_L = R_P + (V_B/V_E)(R_P − r_D) — docs/05 §5.1."""
    return float(rp + vb_ve * (rp - rd))


def summary_stats(returns: pd.DataFrame, periods_per_year: float) -> pd.DataFrame:
    """Per-period and annualized arithmetic mean, geometric mean and volatility for each column."""
    rows = {}
    for name, r in returns.items():
        r = r.dropna()
        arith, geo, std = arithmetic_mean(r), geometric_mean(r), float(r.std(ddof=1))
        rows[name] = {
            "n": len(r),
            "mean": arith,
            "geo_mean": geo,
            "std": std,
            "ann_mean": float(annualize(arith, periods_per_year)),
            "ann_geo_mean": float(annualize(geo, periods_per_year)),
            "ann_vol": std * np.sqrt(periods_per_year),
        }
    return pd.DataFrame(rows).T


def portfolio_variance(w: ArrayLike, cov: ArrayLike) -> float:
    """σ_p² = wᵀ Σ w — docs/05 §5.3."""
    w = np.asarray(w, dtype=float)
    return float(w @ np.asarray(cov, dtype=float) @ w)


def equal_weight_variance(avg_var: float, avg_cov: float, n) -> np.ndarray | float:
    """σ_p² = σ̄²/N + (N−1)/N·cov̄ for an equal-weighted portfolio — docs/05 §5.3."""
    n = np.asarray(n, dtype=float)
    out = avg_var / n + (n - 1) / n * avg_cov
    return float(out) if out.ndim == 0 else out


def diversification_curve(returns: pd.DataFrame, n_max: int | None = None, n_sims: int = 500,
                          seed: int = 42) -> pd.DataFrame:
    """Average std dev of random equal-weight portfolios of N stocks, N = 1..n_max — docs/05 §5.3.

    Also returns the closed-form curve from the average variance and covariance.
    """
    returns = returns.dropna()
    cov = returns.cov().to_numpy()
    k = cov.shape[0]
    n_max = min(n_max or k, k)
    rng = np.random.default_rng(seed)
    off_diag = cov[~np.eye(k, dtype=bool)]
    rows = []
    for n in range(1, n_max + 1):
        stds = []
        for _ in range(n_sims):
            idx = rng.choice(k, size=n, replace=False)
            w = np.full(n, 1 / n)
            stds.append(np.sqrt(portfolio_variance(w, cov[np.ix_(idx, idx)])))
        rows.append({
            "n_stocks": n,
            "simulated_std": float(np.mean(stds)),
            "formula_std": float(np.sqrt(equal_weight_variance(np.diag(cov).mean(), off_diag.mean(), n))),
        })
    out = pd.DataFrame(rows).set_index("n_stocks")
    out.attrs["floor_std"] = float(np.sqrt(off_diag.mean())) if off_diag.mean() > 0 else 0.0
    return out
