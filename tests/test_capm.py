"""docs/07 §7.5 — returns and CAPM tests (docs/05)."""
import numpy as np
import pandas as pd
import pytest

from valuation import capm, returns

RATE = 1e-4


def test_capm_required_return():
    assert capm.capm_required_return(0.025, 1.2, rm=0.08) == pytest.approx(0.091, abs=RATE)
    assert capm.capm_required_return(0.025, 1.2, mrp=0.055) == pytest.approx(0.091, abs=RATE)


def test_beta_meaning():
    """β 1.5 and a +10% market move → +15% (zero rf, on the SML)."""
    assert capm.capm_required_return(0.0, 1.5, rm=0.10) == pytest.approx(0.15)


def test_leveraged_return():
    assert returns.leveraged_return(0.08, 3 / 7, 0.05) == pytest.approx(0.0929, abs=RATE)


def test_geometric_mean():
    assert returns.geometric_mean([0.10, -0.10]) == pytest.approx(-0.00501, abs=1e-5)


@pytest.fixture(scope="module")
def synthetic():
    """docs/05 §5.5: R_m ~ N(0.01, 0.04²) (0.04 read as the std dev), ε ~ N(0, 0.02²),
    R_i = 0.002 + 1.3·R_m + ε, n = 5,000, seed 42."""
    rng = np.random.default_rng(42)
    rm = rng.normal(0.01, 0.04, 5_000)
    ri = 0.002 + 1.3 * rm + rng.normal(0.0, 0.02, 5_000)
    return ri, rm


def test_estimate_beta_synthetic(synthetic):
    ri, rm = synthetic
    fit = capm.estimate_beta(ri, rm)
    assert fit["beta"] == pytest.approx(1.3, abs=0.05)
    assert fit["beta"] == pytest.approx(np.cov(ri, rm)[0, 1] / np.var(rm, ddof=1), abs=1e-10)
    assert fit["t_beta"] > 10
    assert 0 < fit["r2"] < 1


def test_estimate_beta_with_series_and_rf(synthetic):
    ri, rm = synthetic
    idx = pd.date_range("2000-01-31", periods=len(ri), freq="ME")
    fit = capm.estimate_beta(pd.Series(ri, idx), pd.Series(rm, idx), rf=0.001)
    assert fit["beta"] == pytest.approx(capm.estimate_beta(ri, rm)["beta"], abs=1e-10)


def test_variance_decomposition_adds_up(synthetic):
    ri, rm = synthetic
    v = capm.variance_decomposition(ri, rm)
    assert v["systematic"] + v["unsystematic"] == pytest.approx(v["total"])
    assert v["systematic_share"] == pytest.approx(capm.estimate_beta(ri, rm)["r2"], abs=1e-10)


def test_sml_alpha_and_treynor():
    assert capm.sml_alpha(0.12, 0.025, 1.2, 0.08) == pytest.approx(0.12 - 0.091)
    assert capm.treynor(0.091, 0.025, 1.2) == pytest.approx(0.08 - 0.025)


def test_rolling_beta(synthetic):
    ri, rm = synthetic
    idx = pd.date_range("2000-01-31", periods=len(ri), freq="ME")
    rb = capm.rolling_beta(pd.Series(ri, idx), pd.Series(rm, idx), window=36).dropna()
    assert rb.mean() == pytest.approx(1.3, abs=0.05)


def test_diversification_curve_falls_toward_floor():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.04, 600)
    df = pd.DataFrame({f"s{i}": m + rng.normal(0, 0.08, 600) for i in range(20)})
    curve = returns.diversification_curve(df, n_sims=200)
    assert curve["simulated_std"].is_monotonic_decreasing
    assert curve["simulated_std"].iloc[-1] == pytest.approx(curve["formula_std"].iloc[-1], rel=1e-6)
    assert curve["formula_std"].iloc[-1] > curve.attrs["floor_std"]
