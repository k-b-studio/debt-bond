"""Common stock valuation (dividend discount models) — docs/04_equity_valuation.md.

The discount rate ``ke`` is the required return on equity, e.g. from CAPM (docs/05).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .tvm import pv_stream


def _check_ke_g(ke: float, g: float) -> None:
    if ke <= g:
        raise ValueError(f"ke ({ke}) must be > g ({g})")


def one_period_price(d1: float, p1: float, ke: float) -> float:
    """P0 = D1/(1+ke) + P1/(1+ke) — docs/04 §4.2, Mishkin eq. (1)."""
    return float((d1 + p1) / (1 + ke))


def ddm_explicit(dividends: ArrayLike, ke: float, terminal_price: float = 0.0) -> float:
    """P0 = Σ D_t/(1+ke)^t + P_n/(1+ke)^n, t = 1..n — docs/04 §4.3, Mishkin eq. (2).

    ``n = len(dividends)``; pass zeros to value a terminal price alone, e.g. ``[0] * 75``.
    """
    flows = np.atleast_1d(np.asarray(dividends, dtype=float)).copy()
    if flows.size == 0:
        raise ValueError("dividends must not be empty")
    flows[-1] += terminal_price
    return pv_stream(flows, ke)


def growing_dividends(d0: float, g: float, n: int) -> np.ndarray:
    """D_t = D0·(1+g)^t for t = 1..n."""
    return d0 * (1 + g) ** np.arange(1, n + 1)


def gordon_price(ke: float, g: float, d1: float | None = None, d0: float | None = None) -> float:
    """Gordon growth P0 = D1/(ke − g) = D0(1+g)/(ke − g), ke > g — docs/04 §4.4, Mishkin eq. (5).

    Give exactly one of ``d1`` or ``d0``.
    """
    if (d1 is None) == (d0 is None):
        raise ValueError("give exactly one of d1 or d0")
    _check_ke_g(ke, g)
    d1 = d0 * (1 + g) if d1 is None else d1
    return float(d1 / (ke - g))


def gordon_implied_return(price: float, d1: float, g: float) -> float:
    """ke = D1/P0 + g (dividend yield + growth) — docs/04 §4.4."""
    return float(d1 / price + g)


def gordon_implied_growth(price: float, d1: float, ke: float) -> float:
    """g = ke − D1/P0 — docs/04 §4.4."""
    return float(ke - d1 / price)


def gordon_implied_growth_from_d0(price: float, d0: float, ke: float) -> float:
    """g solving P0 = D0(1+g)/(ke − g), i.e. g = (ke·P0 − D0)/(P0 + D0) — docs/04 §4.4, §4.7 (PTT exercise)."""
    return float((ke * price - d0) / (price + d0))


def gordon_sensitivity(ke: float, g: float, d1: float) -> dict:
    """∂P0/∂g = D1/(ke−g)² and ∂P0/∂ke = −D1/(ke−g)² — docs/04 §4.7."""
    _check_ke_g(ke, g)
    slope = d1 / (ke - g) ** 2
    return {"dP_dg": float(slope), "dP_dke": float(-slope)}


def multi_stage_ddm(d0: float, growth_path: Sequence[float], g_terminal: float, ke: float,
                    detail: bool = False) -> float | dict:
    """Explicit dividends along ``growth_path`` then a Gordon terminal value — docs/04 §4.5.

    P0 = Σ_{t=1..n} D_t/(1+ke)^t + [D_n(1+g_L)/(ke − g_L)]/(1+ke)^n.
    With ``detail=True`` return the stage-1 PV, terminal value and its PV as well.
    """
    _check_ke_g(ke, g_terminal)
    divs = d0 * np.cumprod(1 + np.asarray(growth_path, dtype=float))
    n = divs.size
    terminal = (divs[-1] if n else d0) * (1 + g_terminal) / (ke - g_terminal)
    pv_stage1 = pv_stream(divs, ke) if n else 0.0
    pv_terminal = terminal / (1 + ke) ** n
    price = float(pv_stage1 + pv_terminal)
    if not detail:
        return price
    return {"price": price, "pv_stage1": float(pv_stage1), "terminal_value": float(terminal),
            "pv_terminal": float(pv_terminal), "dividends": divs}


def two_stage_ddm(d0: float, g_short: float, n: int, g_long: float, ke: float, detail: bool = False) -> float | dict:
    """Two-stage DDM: ``g_short`` for ``n`` years, then ``g_long`` forever — docs/04 §4.5, CFA LM2 eq. 15–17."""
    return multi_stage_ddm(d0, [g_short] * n, g_long, ke, detail)


def justified_forward_pe(payout: float, ke: float, g: float) -> float:
    """P0/E1 = payout/(ke − g) — docs/04 §4.6."""
    _check_ke_g(ke, g)
    return float(payout / (ke - g))


def pe_implied_return(pe: float, payout: float, g: float) -> float:
    """ke = payout/(P/E) + g — docs/04 §4.6."""
    return float(payout / pe + g)


def pe_implied_growth(pe: float, payout: float, ke: float) -> float:
    """g = ke − payout/(P/E) — docs/04 §4.6."""
    return float(ke - payout / pe)
