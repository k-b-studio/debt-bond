"""Time value of money — docs/02_time_value_of_money.md.

Conventions (docs/07 §7.2): rates are decimals, ``t`` is in years, ``m`` is the
compounding frequency per year, cash flows fall at the end of each period unless
``due=True``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import brentq

TOL = 1e-12


# ---------------------------------------------------------------- helpers
def _scalar_or_array(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def _check_rate(rate, m=1) -> None:
    if np.any(1 + np.asarray(rate, dtype=float) / m <= 0):
        raise ValueError("1 + rate/m must be > 0")


def _check_time(t) -> None:
    if np.any(np.asarray(t, dtype=float) < 0):
        raise ValueError("time / number of periods must be >= 0")


def _check_cash_flows(cfs) -> np.ndarray:
    cfs = np.asarray(cfs, dtype=float).ravel()
    if cfs.size == 0:
        raise ValueError("cash flows must not be empty")
    return cfs


def _bisect(f, lo: float, hi: float, tol: float = TOL) -> float:
    """Plain bisection (the original pv.ipynb irr() style), used if brentq fails."""
    f_lo = f(lo)
    for _ in range(500):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def find_root(f, lo: float, hi: float, tol: float = TOL, max_hi: float = 1e6) -> float:
    """Root of ``f`` on ``[lo, hi]`` with brentq — docs/07 §7.2.

    If ``f(lo)`` and ``f(hi)`` have the same sign, ``hi`` is widened (doubled) up to
    ``max_hi``. If brentq fails to converge, fall back to bisection.
    """
    f_lo, f_hi = f(lo), f(hi)
    while np.sign(f_lo) == np.sign(f_hi) and hi < max_hi:
        hi = hi * 2 if hi > 0 else 1.0
        f_hi = f(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        raise ValueError(f"no root bracketed in [{lo}, {hi}]")
    try:
        return float(brentq(f, lo, hi, xtol=tol, maxiter=500))
    except (RuntimeError, ValueError):
        return float(_bisect(f, lo, hi, tol))


# ---------------------------------------------------------------- §2.2 single cash flow
def discount_factor(rate, t, m: int = 1):
    """DF = (1 + rate/m)^(−m·t) — docs/02 §2.2. ``rate`` and ``t`` may be arrays."""
    _check_rate(rate, m)
    _check_time(t)
    return _scalar_or_array((1 + np.asarray(rate, dtype=float) / m) ** (-m * np.asarray(t, dtype=float)))


def fv(pv: float, rate: float, t: float, m: int = 1) -> float:
    """FV = PV·(1 + rate/m)^(m·t) — docs/02 §2.2, Mishkin eq. (1)."""
    return float(pv / discount_factor(rate, t, m))


def pv(fv: float, rate: float, t: float, m: int = 1) -> float:
    """PV = FV / (1 + rate/m)^(m·t) — docs/02 §2.2, Mishkin eq. (1)."""
    return float(fv * discount_factor(rate, t, m))


# ---------------------------------------------------------------- §2.3 streams
def pv_stream(cfs: ArrayLike, rate, times: ArrayLike | None = None, m: int = 1) -> float:
    """PV of a cash-flow stream (cash-flow additivity) — docs/02 §2.3.

    ``times`` are in years; the default is one cash flow per period, ``1/m, 2/m, …``.
    ``rate`` may be a vector (spot curve) so that each CF is discounted at its own rate.
    """
    cfs = _check_cash_flows(cfs)
    times = np.arange(1, cfs.size + 1) / m if times is None else np.asarray(times, dtype=float)
    if times.shape != cfs.shape:
        raise ValueError("cfs and times must have the same length")
    return float(cfs @ np.atleast_1d(discount_factor(rate, times, m)))


def npv(rate: float, cfs: ArrayLike, t0_included: bool = True) -> float:
    """NPV with one CF per period; ``cfs[0]`` is at t = 0 (undiscounted) if ``t0_included``."""
    cfs = _check_cash_flows(cfs)
    start = 0 if t0_included else 1
    return float(cfs @ np.atleast_1d(discount_factor(rate, np.arange(start, start + cfs.size))))


def irr(cfs: ArrayLike, lo: float = -0.99, hi: float = 1.0) -> float:
    """Internal rate of return: the root of ``npv`` — docs/02 §2.3."""
    cfs = _check_cash_flows(cfs)
    return find_root(lambda r: npv(r, cfs), lo, hi)


# ---------------------------------------------------------------- §2.4 frequency, EAR, logs
def ear(rate: float, m: float) -> float:
    """Effective annual rate (1 + R/m)^m − 1; ``m = np.inf`` gives continuous e^R − 1 — docs/02 §2.4."""
    if math.isinf(m):
        return float(math.expm1(rate))
    _check_rate(rate, m)
    return float((1 + rate / m) ** m - 1)


def continuous_fv(pv: float, rate: float, t: float) -> float:
    """FV = PV·e^(R·t) — docs/02 §2.4."""
    return float(pv * math.exp(rate * t))


def log_return(p0, p1):
    """Continuously compounded return ln(P1/P0) — docs/02 §2.4."""
    return _scalar_or_array(np.log(np.asarray(p1, dtype=float) / np.asarray(p0, dtype=float)))


def annualize(r_period, c: float):
    """(1 + R_period)^c − 1, with c = periods per year (e.g. 52, 365/days, 2/3) — docs/02 §2.4."""
    return _scalar_or_array((1 + np.asarray(r_period, dtype=float)) ** c - 1)


# ---------------------------------------------------------------- §2.5 annuities
def annuity_pv(a: float, r: float, n: int, due: bool = False) -> float:
    """PV of a level annuity A·[1 − (1+r)^−n]/r; annuity due × (1+r) — docs/02 §2.5."""
    _check_rate(r)
    _check_time(n)
    value = a * n if r == 0 else a * (1 - (1 + r) ** -n) / r
    return float(value * (1 + r) if due else value)


def annuity_payment(pv: float, r: float, n: int) -> float:
    """Level payment A = r·PV / [1 − (1+r)^−n] — docs/02 §2.5."""
    _check_rate(r)
    if n <= 0:
        raise ValueError("n must be > 0")
    return float(pv / n if r == 0 else r * pv / (1 - (1 + r) ** -n))


def annuity_rate(pv: float, a: float, n: int, due: bool = False) -> float:
    """Periodic rate that makes ``annuity_pv(a, r, n) == pv`` (root-find) — docs/02 §2.5."""
    return find_root(lambda r: annuity_pv(a, r, n, due) - pv, -0.99, 1.0)


def amortization_schedule(principal: float, r: float, n: int) -> pd.DataFrame:
    """Fixed-payment loan schedule: period, payment, interest, principal, balance — docs/02 §2.8."""
    payment = annuity_payment(principal, r, n)
    rows, balance = [], float(principal)
    for k in range(1, n + 1):
        interest = balance * r
        repaid = payment - interest
        balance -= repaid
        rows.append((k, payment, interest, repaid, balance))
    out = pd.DataFrame(rows, columns=["period", "payment", "interest", "principal", "balance"])
    out.loc[out.index[-1], "balance"] = 0.0 if abs(balance) < 1e-6 else balance
    return out


# ---------------------------------------------------------------- §2.6 perpetuities
def perpetuity_pv(c: float, rate: float) -> float:
    """PV = C / i — docs/02 §2.6."""
    if rate <= 0:
        raise ValueError("rate must be > 0")
    return float(c / rate)


def perpetuity_rate(c: float, price: float) -> float:
    """Yield of a perpetuity (consol) i = C / P — docs/02 §2.6, docs/03 §3.2."""
    if price <= 0:
        raise ValueError("price must be > 0")
    return float(c / price)


def growing_perpetuity_pv(c1: float, rate: float, g: float) -> float:
    """PV = C1 / (i − g), i > g — docs/02 §2.6 (Gordon model in docs/04)."""
    if rate <= g:
        raise ValueError("rate must be > g")
    return float(c1 / (rate - g))


# ---------------------------------------------------------------- §2.7 forwards
def forward_rate(z_a: float, a: float, z_b: float, b: float) -> float:
    """Implied forward f from (1+z_B)^B = (1+z_A)^A (1+f)^(B−A) — docs/02 §2.7."""
    if b <= a:
        raise ValueError("b must be > a")
    return float(((1 + z_b) ** b / (1 + z_a) ** a) ** (1 / (b - a)) - 1)
