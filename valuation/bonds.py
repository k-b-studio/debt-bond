"""Debt instruments, yields and interest-rate risk — docs/03_debt_valuation.md.

Refactor of ``pv.ipynb``: the argument names ``par, coupon_rate, months, freq, ytm,
tax_rate`` are kept. ``freq`` = coupons per year = compounding frequency m.
Bonds are priced on a coupon date (no accrued interest).

Coupon tax (docs/03 §3.9): the after-tax cash flow is ``CF_k − tax_rate · C/m``, so tax
applies to the coupon only and the principal is untaxed (Thai 15% withholding tax).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .tvm import _check_rate, _scalar_or_array, find_root

TOL = 1e-12


# ---------------------------------------------------------------- pv.ipynb building blocks
def periods(months: float, freq: int) -> np.ndarray:
    """Coupon period index 1..n with n = round(months·freq/12) — pv.ipynb ``periods()``."""
    if months < 0:
        raise ValueError("months must be >= 0")
    n = int(round(months * freq / 12))
    if n < 1:
        raise ValueError("bond must have at least one coupon period")
    return np.arange(1, n + 1)


def cash_flows(par: float, coupon_rate: float, months: float, freq: int) -> np.ndarray:
    """Coupon C/m every period, plus par at maturity — pv.ipynb ``cash_flows()``."""
    flows = np.full(len(periods(months, freq)), par * coupon_rate / freq, dtype=float)
    flows[-1] += par
    return flows


def after_tax_cash_flows(par, coupon_rate, months, freq, tax_rate) -> np.ndarray:
    """Cash flows less tax on the coupon only — docs/03 §3.9, pv.ipynb ``after_tax_cash_flows()``."""
    return cash_flows(par, coupon_rate, months, freq) - par * coupon_rate / freq * tax_rate


def discount_factors(ytm, months, freq) -> np.ndarray:
    """(1 + ytm/m)^(−k) for k = 1..n — pv.ipynb ``discount_factors()``."""
    _check_rate(ytm, freq)
    return (1 + ytm / freq) ** -periods(months, freq)


def bond_cash_flows(par, coupon_rate, months, freq=1, tax_rate=0.0) -> tuple[np.ndarray, np.ndarray]:
    """(times in years, cash flows) of a coupon bond — docs/03 §3.2."""
    return periods(months, freq) / freq, after_tax_cash_flows(par, coupon_rate, months, freq, tax_rate)


# ---------------------------------------------------------------- §3.2 price and YTM
def bond_price(par, coupon_rate, months, ytm, freq=1, tax_rate=0.0):
    """P = Σ (C/m)/(1+y/m)^k + F/(1+y/m)^n — docs/03 §3.2.

    ``ytm`` may be an array, in which case an array of prices is returned.
    """
    _check_rate(ytm, freq)
    flows = after_tax_cash_flows(par, coupon_rate, months, freq, tax_rate)
    k = periods(months, freq)
    factors = (1 + np.asarray(ytm, dtype=float)[..., None] / freq) ** -k
    return _scalar_or_array(factors @ flows)


def present_value(par, coupon_rate, months, ytm, freq=1, tax_rate=0.0) -> float:
    """pv.ipynb name for :func:`bond_price`."""
    return bond_price(par, coupon_rate, months, ytm, freq, tax_rate)


def net_present_value(price, par, coupon_rate, months, rate, freq=1, tax_rate=0.0) -> float:
    """PV at ``rate`` minus the price paid — pv.ipynb ``net_present_value()``."""
    return bond_price(par, coupon_rate, months, rate, freq, tax_rate) - price


def bond_ytm(price, par, coupon_rate, months, freq=1, tax_rate=0.0, tol=TOL) -> float:
    """Yield to maturity: the IRR of the (after-tax) cash flows — docs/03 §3.2.

    brentq on ``bond_price(y) − price`` with bracket (−freq + 1e-9, 1.0), widened if needed,
    so that 1 + y/m > 0 always holds. Negative yields are allowed.
    """
    if price <= 0:
        raise ValueError("price must be > 0")
    return find_root(lambda y: net_present_value(price, par, coupon_rate, months, y, freq, tax_rate),
                     -freq + 1e-9, 1.0, tol)


def irr(price, par, coupon_rate, months, freq=1, tax_rate=0.0, tol=TOL) -> float:
    """pv.ipynb name for :func:`bond_ytm`."""
    return bond_ytm(price, par, coupon_rate, months, freq, tax_rate, tol)


def bond_schedule(par, coupon_rate, months, ytm, freq=1, tax_rate=0.0) -> pd.DataFrame:
    """Per-period cash flows and PVs — the ``schedule`` table from pv.ipynb."""
    out = pd.DataFrame({
        "period": periods(months, freq),
        "time": periods(months, freq) / freq,
        "cash_flow": cash_flows(par, coupon_rate, months, freq),
        "after_tax_cash_flow": after_tax_cash_flows(par, coupon_rate, months, freq, tax_rate),
        "discount_factor": discount_factors(ytm, months, freq),
    })
    out["present_value"] = out["cash_flow"] * out["discount_factor"]
    out["after_tax_present_value"] = out["after_tax_cash_flow"] * out["discount_factor"]
    return out


def price_yield_curve(par, coupon_rate, months, ytms, freq=1, tax_rate=0.0) -> pd.DataFrame:
    """Price before and after coupon tax over a grid of yields — the ``curve`` table from pv.ipynb."""
    ytms = np.asarray(ytms, dtype=float)
    return pd.DataFrame({
        "ytm": ytms,
        "price": bond_price(par, coupon_rate, months, ytms, freq),
        "price_after_tax": bond_price(par, coupon_rate, months, ytms, freq, tax_rate),
    })


# ---------------------------------------------------------------- discount bonds, consols
def zero_price(face: float, ytm: float, t: float, m: int = 1) -> float:
    """Discount (zero-coupon) bond P = F/(1 + y/m)^(m·t) — docs/03 §3.1."""
    _check_rate(ytm, m)
    if t < 0:
        raise ValueError("t must be >= 0")
    return float(face * (1 + ytm / m) ** (-m * t))


def zero_ytm(price: float, face: float, t: float) -> float:
    """Closed-form annual YTM of a zero: (F/P)^(1/t) − 1 — docs/03 §3.2, Mishkin eq. (6)."""
    if price <= 0 or t <= 0:
        raise ValueError("price and t must be > 0")
    return float((face / price) ** (1 / t) - 1)


def current_yield(coupon: float, price: float) -> float:
    """i_c = C / P (annual coupon over price) — docs/03 §3.4."""
    if price <= 0:
        raise ValueError("price must be > 0")
    return float(coupon / price)


consol_yield = current_yield  # a perpetuity's YTM equals its current yield — docs/03 §3.2


# ---------------------------------------------------------------- §3.2 / §3.5 tables
def table1(par=1000.0, coupon_rate=0.10, years=10, prices=(1200, 1100, 1000, 900, 800), freq=1) -> pd.DataFrame:
    """Mishkin Table 1: YTM of a 10-year 10% coupon bond at several prices — docs/03 §3.2."""
    return pd.DataFrame({
        "price": list(prices),
        "ytm": [bond_ytm(p, par, coupon_rate, years * 12, freq) for p in prices],
    })


def holding_period_return(p0: float, p1: float, income: float = 0.0) -> float:
    """R = (C + P_{t+1} − P_t)/P_t — docs/03 §3.5, Mishkin eq. (7)."""
    if p0 == 0:
        raise ValueError("p0 must be non-zero")
    return float((income + p1 - p0) / p0)


def horizon_return(par, coupon_rate, years, y0, y1, freq=1, horizon=1.0) -> dict:
    """Buy at yield ``y0``, sell after ``horizon`` years at yield ``y1`` — docs/03 §3.5.

    Coupons received during the horizon are summed (not reinvested). If the bond matures
    within the horizon it is redeemed at par.
    """
    months, held = round(years * 12), round(horizon * 12)
    p0 = bond_price(par, coupon_rate, months, y0, freq)
    left = months - held
    p1 = float(par) if left <= 0 else bond_price(par, coupon_rate, left, y1, freq)
    income = par * coupon_rate * min(held, months) / 12
    return {
        "initial_price": p0,
        "next_price": p1,
        "current_yield": income / p0,
        "capital_gain": (p1 - p0) / p0,
        "return": holding_period_return(p0, p1, income),
    }


def table2(coupon_rate=0.10, y0=0.10, y1=0.20, maturities=(30, 20, 10, 5, 2, 1), par=1000.0) -> pd.DataFrame:
    """Mishkin Table 2: one-year returns when rates jump from ``y0`` to ``y1`` — docs/03 §3.5."""
    rows = [{"years_to_maturity": n, **horizon_return(par, coupon_rate, n, y0, y1)} for n in maturities]
    return pd.DataFrame(rows).set_index("years_to_maturity")


def shock_returns(shocks_bp: Sequence[float], maturities: Sequence[float], coupon_rate=0.10, y0=0.10,
                  par=1000.0, freq=1) -> pd.DataFrame:
    """One-year returns for parallel yield shocks (rows: maturity, columns: shock in bp) — docs/03 §3.11."""
    return pd.DataFrame(
        {s: [horizon_return(par, coupon_rate, n, y0, y0 + s / 10_000, freq)["return"] for n in maturities]
         for s in shocks_bp},
        index=pd.Index(maturities, name="years_to_maturity"),
    ).rename_axis(columns="shock_bp")


# ---------------------------------------------------------------- §3.6 reinvestment risk
def rollover_value(principal: float, rates: Sequence[float]) -> float:
    """Value after rolling one-period bonds at ``rates`` — docs/03 §3.6, Mishkin fn. 6."""
    return float(principal * np.prod(1 + np.asarray(rates, dtype=float)))


# ---------------------------------------------------------------- §3.7 duration and convexity
def duration(par, coupon_rate, months, ytm, freq=1, tax_rate=0.0) -> dict:
    """Macaulay and modified duration, convexity and DV01 — docs/03 §3.7.

    D_mac = Σ t_k·PV_k / P,  D_mod = D_mac/(1+y/m),
    Convexity = Σ CF_k·k(k+1)/(1+y/m)^k / [P·(1+y/m)²·m²],  DV01 = D_mod·P·0.0001.
    """
    _check_rate(ytm, freq)
    k = periods(months, freq)
    flows = after_tax_cash_flows(par, coupon_rate, months, freq, tax_rate)
    base = 1 + ytm / freq
    pv_k = flows * base ** -k
    price = pv_k.sum()
    macaulay = (k / freq * pv_k).sum() / price
    modified = macaulay / base
    convexity = (pv_k * k * (k + 1)).sum() / (price * base ** 2 * freq ** 2)
    return {
        "price": float(price),
        "macaulay": float(macaulay),
        "modified": float(modified),
        "convexity": float(convexity),
        "dv01": float(modified * price * 1e-4),
    }


def effective_duration(par, coupon_rate, months, ytm, freq=1, tax_rate=0.0, h=1e-4) -> dict:
    """Finite-difference duration (P(y−h) − P(y+h))/(2hP) and convexity — check on docs/03 §3.7."""
    p0, p_dn, p_up = (bond_price(par, coupon_rate, months, y, freq, tax_rate) for y in (ytm, ytm - h, ytm + h))
    return {
        "duration": float((p_dn - p_up) / (2 * h * p0)),
        "convexity": float((p_dn + p_up - 2 * p0) / (h ** 2 * p0)),
    }


def price_change_approx(modified: float, convexity: float, dy) -> float:
    """ΔP/P ≈ −D_mod·Δy + ½·Convexity·(Δy)² — docs/03 §3.7."""
    dy = np.asarray(dy, dtype=float)
    return _scalar_or_array(-modified * dy + 0.5 * convexity * dy ** 2)


# ---------------------------------------------------------------- §3.8 real rates
def real_rate(nominal, infl, exact: bool = True):
    """Fisher: exact (1+i)/(1+π) − 1, or approx. i − π — docs/03 §3.8, Mishkin eq. (9)."""
    nominal, infl = np.asarray(nominal, dtype=float), np.asarray(infl, dtype=float)
    return _scalar_or_array((1 + nominal) / (1 + infl) - 1 if exact else nominal - infl)


def after_tax_real(nominal: float, tax: float, infl: float) -> float:
    """After-tax real rate i(1 − τ) − π^e — docs/03 §3.8, Mishkin eq. (10)."""
    return float(nominal * (1 - tax) - infl)
