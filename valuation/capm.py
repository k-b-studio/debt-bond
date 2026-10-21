"""Capital Asset Pricing Model and beta — docs/05_risk_return_capm.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def capm_required_return(rf: float, beta, rm: float | None = None, mrp: float | None = None):
    """r_a = r_f + β·(r_m − r_f); give ``rm`` or ``mrp`` — docs/05 §5.4."""
    if (rm is None) == (mrp is None):
        raise ValueError("give exactly one of rm or mrp")
    mrp = rm - rf if mrp is None else mrp
    out = rf + np.asarray(beta, dtype=float) * mrp
    return float(out) if out.ndim == 0 else out


def _align(asset_ret, market_ret, rf=0.0) -> tuple[pd.Series, pd.Series]:
    """Excess returns on the common index (positional if either input is not a Series), NaNs dropped."""
    if not (isinstance(asset_ret, pd.Series) and isinstance(market_ret, pd.Series)):
        asset_ret = pd.Series(np.asarray(asset_ret, dtype=float))
        market_ret = pd.Series(np.asarray(market_ret, dtype=float))
    df = pd.concat({"y": asset_ret, "x": market_ret}, axis=1, join="inner")
    df["rf"] = rf.reindex(df.index) if isinstance(rf, pd.Series) else rf
    df = df.dropna()
    if len(df) < 3:
        raise ValueError("need at least 3 overlapping observations")
    return df["y"] - df["rf"], df["x"] - df["rf"]


def estimate_beta(asset_ret, market_ret, rf=0.0) -> dict:
    """OLS of R_i − r_f = α + β(R_m − r_f) + ε — docs/05 §5.5.

    Returns alpha, beta, their standard errors and t-stats, R², residual std and n.
    ``rf`` is a per-period rate (scalar or Series aligned to the returns).
    """
    y, x = _align(asset_ret, market_ret, rf)
    fit = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit()
    return {
        "alpha": float(fit.params[0]),
        "beta": float(fit.params[1]),
        "se_alpha": float(fit.bse[0]),
        "se_beta": float(fit.bse[1]),
        "t_alpha": float(fit.tvalues[0]),
        "t_beta": float(fit.tvalues[1]),
        "r2": float(fit.rsquared),
        "resid_std": float(np.sqrt(fit.mse_resid)),
        "n": int(fit.nobs),
    }


def rolling_beta(asset_ret: pd.Series, market_ret: pd.Series, window: int = 36) -> pd.Series:
    """Rolling β = Cov(R_i, R_m)/Var(R_m) over ``window`` periods — docs/05 §5.5."""
    df = pd.concat({"y": asset_ret, "x": market_ret}, axis=1).dropna()
    return (df["y"].rolling(window).cov(df["x"]) / df["x"].rolling(window).var()).rename("rolling_beta")


def blume_beta(beta):
    """Blume-adjusted beta ⅔·β + ⅓ — docs/05 §5.5."""
    return 2 / 3 * beta + 1 / 3


def variance_decomposition(asset_ret, market_ret, rf=0.0) -> dict:
    """σ_i² = β²σ_m² (systematic) + σ_ε² (unsystematic) — docs/05 §5.3."""
    y, x = _align(asset_ret, market_ret, rf)
    beta = float(np.cov(y, x)[0, 1] / np.var(x, ddof=1))
    total = float(np.var(y, ddof=1))
    systematic = beta ** 2 * float(np.var(x, ddof=1))
    return {"total": total, "systematic": systematic, "unsystematic": total - systematic,
            "systematic_share": systematic / total}


def sml_alpha(realized_mean, rf: float, beta, rm: float):
    """Realized mean return minus the CAPM required return (Jensen's alpha) — docs/05 §5.5."""
    return realized_mean - capm_required_return(rf, beta, rm=rm)


def treynor(mean_ret, rf: float, beta):
    """(r_a − r_f)/β; equals r_m − r_f for every asset on the SML — docs/05 §5.4."""
    return (mean_ret - rf) / beta


def historical_mrp(market_ret: pd.Series, rf=0.0, periods_per_year: float = 12) -> float:
    """Annualized arithmetic mean of R_m − r_f (per-period ``rf``) — docs/05 §5.6 (a)."""
    excess = (market_ret - rf).dropna()
    return float(excess.mean() * periods_per_year)


def implied_mrp(d1: float, price: float, g: float, rf: float) -> float:
    """MRP = D1/P_index + g − r_f (Gordon on the index) — docs/05 §5.6 (b)."""
    return float(d1 / price + g - rf)
