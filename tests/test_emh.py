"""docs/06 §6.7 — synthetic expectations for the EMH test toolkit (seed 42)."""
import numpy as np
import pandas as pd
import pytest

from valuation import emh


@pytest.fixture(scope="module")
def iid():
    return emh.simulate_returns("iid", t=5_000, sigma=0.01, seed=42)


@pytest.fixture(scope="module")
def ar1():
    return emh.simulate_returns("ar1", t=5_000, sigma=0.01, phi=0.3, seed=42)


def test_iid_looks_random(iid):
    assert emh.autocorr_test(iid, lags=10)["lb_pvalue"].iloc[-1] > 0.05
    assert abs(emh.runs_test(iid)["z"]) < 1.96
    assert emh.variance_ratio(iid, 5)["vr"] == pytest.approx(1.0, abs=0.1)


def test_ar1_is_predictable(ar1):
    ac = emh.autocorr_test(ar1, lags=10)
    assert ac["rho"].iloc[0] == pytest.approx(0.3, abs=0.05)
    assert ac["lb_pvalue"].iloc[-1] < 0.01
    assert emh.variance_ratio(ar1, 5)["vr"] > 1


def test_runs_test_matches_formula():
    r = np.array([1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1], dtype=float)
    out = emh.runs_test(r)
    n_pos, n_neg, n = 7, 5, 12
    assert out["runs"] == 7
    assert out["expected"] == pytest.approx(2 * n_pos * n_neg / n + 1)


def test_weak_form_battery_flags_ar1(iid, ar1):
    table = emh.weak_form_battery(pd.DataFrame({"iid": iid, "ar1": ar1}))
    assert not table.loc["iid", "reject_5%"]
    assert table.loc["ar1", "reject_5%"]


def test_event_study_detects_injected_jump():
    asset, market, events = emh.simulate_event_data(jump=0.05, seed=42)
    res = emh.event_study(asset, market, events)
    caar = res["caar"]
    assert res["n_events"] == len(events)
    assert caar.loc[-1] == pytest.approx(0.0, abs=0.015)
    assert caar.loc[0] - caar.loc[-1] == pytest.approx(0.05, abs=0.01)
    assert caar.loc[10] - caar.loc[0] == pytest.approx(0.0, abs=0.015)   # flat after day 0
    assert res["t_caar"].loc[0] > 3


def test_ma_backtest_no_lookahead_and_costs():
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800))), index=pd.bdate_range("2020-01-01", periods=800))
    bt = emh.ma_crossover_backtest(prices, short=20, long=100, cost_bp=10)
    assert (bt["position"].iloc[1:].to_numpy() == bt["signal"].iloc[:-1].to_numpy()).all()
    assert bt["eq_net"].iloc[-1] <= bt["eq_gross"].iloc[-1]
    summary = emh.performance_summary(bt)
    assert summary.loc["MA rule (no cost)", "trades"] == bt["trade"].sum()


def test_adaptive_expectation_recursion():
    s = pd.Series([1.0, 2.0, 3.0])
    e = emh.adaptive_expectation(s, lam=0.5)
    assert e.tolist() == pytest.approx([1.0, 1.5, 2.25])


def test_rational_forecast_errors_are_unpredictable():
    stats = emh.forecast_error_stats(emh.expectations_demo(t=2_000, phi=0.7, lam=0.7, seed=42))
    assert abs(stats.loc["rational", "error_rho_1"]) < 0.1
    assert stats.loc["adaptive", "error_rho_1"] > 0.3
    assert stats.loc["rational", "rmse"] < stats.loc["adaptive", "rmse"]
