"""Credit spreads and the rate build-up — docs/01 §1.5, docs/03 §3.10.

ThaiBMA average corporate spreads (bp over government yields) by rating and maturity
bucket. Spreads are in bp in the files and are converted to decimals here, at the
loader boundary (docs/07 §7.2).

Bucket boundaries: T ≤ 3 → ``"<=3"``, 3 < T ≤ 5 → ``"3-5"``, T > 5 → ``">5"``
(the file labels are "<= 3 Yrs.", "3 - 5 Yrs.", "> 5 Yrs.").
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .bonds import bond_price

BUCKETS = ("<=3", "3-5", ">5")
RATINGS = ("AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-")
_BUCKET_LABELS = {"<= 3 Yrs.": "<=3", "3 - 5 Yrs.": "3-5", "> 5 Yrs.": ">5"}
_METRIC_LABELS = {"Averaged Spread (bp)": "spread", "Standard Deviation (bp)": "std", "Averaged Maturity": "maturity"}


def read_spread_file(path: str | Path) -> dict[str, pd.DataFrame]:
    """All three tables in a ThaiBMA spread file: ``spread`` and ``std`` (bp), ``maturity`` (years).

    The sheet has a 2-row header: row 0 = metric group, row 1 = maturity bucket.
    """
    raw = pd.read_excel(path, header=[0, 1], index_col=0)
    raw.index = raw.index.astype(str).str.strip()
    raw.index.name = "rating"
    metrics = pd.Series(raw.columns.get_level_values(0)).replace(r"^Unnamed.*", np.nan, regex=True).ffill()
    raw.columns = pd.MultiIndex.from_arrays([
        metrics.map(lambda s: _METRIC_LABELS[str(s).strip()]),
        [_BUCKET_LABELS[str(b).strip()] for b in raw.columns.get_level_values(1)],
    ])
    return {name: raw[name][list(BUCKETS)].astype(float) for name in _METRIC_LABELS.values()}


def load_spreads(path: str | Path, metric: str = "spread") -> pd.DataFrame:
    """Spread table indexed by rating with columns ``["<=3", "3-5", ">5"]``, values in bp — docs/01 §1.5."""
    return read_spread_file(path)[metric]


def maturity_bucket(maturity_years: float) -> str:
    """Bucket label for a maturity in years (≤3, (3, 5], >5)."""
    if maturity_years < 0:
        raise ValueError("maturity must be >= 0")
    return "<=3" if maturity_years <= 3 else "3-5" if maturity_years <= 5 else ">5"


def spread_for(rating: str, maturity_years: float, table: pd.DataFrame) -> float:
    """Spread for a rating and maturity as a **decimal** — docs/01 §1.5.

    If the cell is NaN (e.g. BBB > 5Y), fall back to the nearest bucket with data and warn.
    """
    if rating not in table.index:
        raise KeyError(f"unknown rating {rating!r}; expected one of {list(table.index)}")
    bucket = maturity_bucket(maturity_years)
    value = table.at[rating, bucket]
    if pd.isna(value):
        pos = BUCKETS.index(bucket)
        for alt in sorted(BUCKETS, key=lambda b: (abs(BUCKETS.index(b) - pos), -BUCKETS.index(b))):
            if pd.notna(table.at[rating, alt]):
                warnings.warn(f"no {rating} spread for bucket {bucket}; using {alt}", stacklevel=2)
                value = table.at[rating, alt]
                break
        else:
            raise ValueError(f"no spread data for rating {rating}")
    return float(value) / 10_000


def gov_yield(curve: Mapping[float, float], t: float) -> float:
    """Government yield at maturity ``t`` by linear interpolation of ``{tenor: yield}``, flat outside."""
    tenors = np.array(sorted(curve), dtype=float)
    return float(np.interp(t, tenors, [curve[k] for k in sorted(curve)]))


def corporate_yield(gov_curve: Mapping[float, float], rating: str, maturity_years: float, table: pd.DataFrame) -> float:
    """y_corp(T, rating) = y_gov(T) + spread(T, rating) — docs/01 §1.5."""
    return gov_yield(gov_curve, maturity_years) + spread_for(rating, maturity_years, table)


def price_by_rating(par: float, coupon_rate: float, months: float, gov_curve: Mapping[float, float],
                    table: pd.DataFrame, freq: int = 1, ratings=None) -> pd.DataFrame:
    """Price one bond at each rating's corporate yield — docs/03 §3.10."""
    t = months / 12
    rows = []
    for rating in ratings or table.index:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            s = spread_for(rating, t, table)
        y = gov_yield(gov_curve, t) + s
        rows.append({
            "rating": rating,
            "gov_yield": gov_yield(gov_curve, t),
            "spread_bp": s * 10_000,
            "yield": y,
            "price": bond_price(par, coupon_rate, months, y, freq),
            "note": "; ".join(str(w.message) for w in caught),
        })
    return pd.DataFrame(rows).set_index("rating")


def rate_build_up(real_rf: float, inflation: float, drp: float = 0.0, lp: float = 0.0, mp: float = 0.0) -> pd.Series:
    """r = r_f^real + π + DRP + LP + MP (additive build-up) — docs/01 §1.5."""
    parts = pd.Series({"real risk-free": real_rf, "inflation premium": inflation,
                       "default risk premium": drp, "liquidity premium": lp, "maturity premium": mp})
    return pd.concat([parts, pd.Series({"total": parts.sum()})])
