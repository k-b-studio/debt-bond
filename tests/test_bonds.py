"""docs/07 §7.5 — bond tests, plus the pv.ipynb snapshot (regression) test."""
import numpy as np
import pytest

from valuation import bonds, tvm

CUR = 0.01
RATE = 1e-4


def test_bond_price_discount_bond():
    assert bonds.bond_price(1000, 0.10, 8 * 12, 0.1225) == pytest.approx(889.20, abs=CUR)


def test_table1_ytm():
    expected = [0.0713, 0.0848, 0.1000, 0.1175, 0.1381]
    assert bonds.table1()["ytm"].to_numpy() == pytest.approx(expected, abs=RATE)


def test_par_bond_price_equals_face():
    assert bonds.bond_price(100, 0.02, 7 * 12, 0.02) == pytest.approx(100.00, abs=CUR)


def test_bond_ytm_cfa_greek():
    assert bonds.bond_ytm(93.091, 100, 0.02, 5 * 12) == pytest.approx(0.03532, abs=RATE)


def test_bond_price_semiannual():
    assert bonds.bond_price(100, 0.067, 20 * 12, 0.077, freq=2) == pytest.approx(89.88, abs=CUR)


def test_zero_price_annual_vs_semiannual():
    assert bonds.zero_price(100, 0.067, 20) == pytest.approx(27.33, abs=CUR)
    assert bonds.zero_price(100, 0.067, 20, m=2) == pytest.approx(26.77, abs=CUR)


@pytest.mark.parametrize("price, face, t, expected", [
    (900, 1000, 1, 0.1111),
    (950, 1000, 1, 0.0526),
    (22.68224, 100, 20, 0.0770),
])
def test_zero_ytm(price, face, t, expected):
    assert bonds.zero_ytm(price, face, t) == pytest.approx(expected, abs=RATE)


def test_current_yield():
    price = bonds.bond_price(1000, 0.10, 20 * 12, 0.12)
    assert price == pytest.approx(850.61, abs=CUR)
    assert bonds.current_yield(100, price) == pytest.approx(0.1176, abs=RATE)


def test_consol_yield_kb_perpetual():
    assert bonds.consol_yield(3.30, 97.03) == pytest.approx(0.0340, abs=RATE)


def test_price_yield_facts():
    """docs/03 §3.3: par ⇔ YTM = coupon; price falls as yield rises; discount ⇒ YTM > coupon."""
    ytms = np.linspace(0.02, 0.20, 50)
    prices = bonds.bond_price(1000, 0.10, 120, ytms)
    assert np.all(np.diff(prices) < 0)
    assert bonds.bond_ytm(1000, 1000, 0.10, 120) == pytest.approx(0.10, abs=1e-10)
    assert bonds.bond_ytm(900, 1000, 0.10, 120) > 0.10 > bonds.bond_ytm(1100, 1000, 0.10, 120)


def test_table2():
    t = bonds.table2()
    assert t["next_price"].round().tolist() == [503, 516, 597, 741, 917, 1000]
    assert (t["return"] * 100).round(1).tolist() == [-39.7, -38.4, -30.3, -15.9, 1.7, 10.0]
    assert (t["capital_gain"] * 100).round(1).tolist() == [-49.7, -48.4, -40.3, -25.9, -8.3, 0.0]


def test_holding_period_return():
    assert bonds.holding_period_return(1000, 1200, 100) == pytest.approx(0.30, abs=RATE)


def test_rollover_value():
    v = bonds.rollover_value(1000, [0.10, 0.20])
    assert v == pytest.approx(1_320, abs=CUR)
    assert tvm.annualize(v / 1000 - 1, 1 / 2) == pytest.approx(0.1489, abs=RATE)
    v_low = bonds.rollover_value(1000, [0.10, 0.05])
    assert v_low == pytest.approx(1_155, abs=CUR)
    assert tvm.annualize(v_low / 1000 - 1, 1 / 2) == pytest.approx(0.0747, abs=RATE)


def test_duration_convexity():
    d = bonds.duration(1000, 0.10, 120, 0.10)
    assert d["macaulay"] == pytest.approx(6.7590, abs=RATE)
    assert d["modified"] == pytest.approx(6.1446, abs=RATE)
    assert d["convexity"] == pytest.approx(52.79, abs=CUR)
    assert d["dv01"] == pytest.approx(d["modified"] * d["price"] * 1e-4)
    fd = bonds.effective_duration(1000, 0.10, 120, 0.10)
    assert fd["duration"] == pytest.approx(d["modified"], abs=1e-5)
    assert fd["convexity"] == pytest.approx(d["convexity"], abs=1e-2)


def test_duration_convexity_approximation_plus_100bp():
    d = bonds.duration(1000, 0.10, 120, 0.10)
    assert bonds.price_change_approx(d["modified"], d["convexity"], 0.01) == pytest.approx(-0.05881, abs=1e-5)
    assert bonds.bond_price(1000, 0.10, 120, 0.11) / 1000 - 1 == pytest.approx(-0.05889, abs=1e-5)


def test_real_rates():
    assert bonds.real_rate(0.05, 0.03, exact=False) == pytest.approx(0.02, abs=RATE)
    assert bonds.real_rate(0.08, 0.10, exact=False) == pytest.approx(-0.02, abs=RATE)
    assert bonds.after_tax_real(0.10, 0.30, 0.05) == pytest.approx(0.02, abs=RATE)
    assert bonds.real_rate(0.024, 0.02, exact=True) == pytest.approx(0.0039, abs=RATE)


def test_negative_yield_allowed():
    price = bonds.bond_price(100, 0.0, 60, -0.005)
    assert price > 100
    assert bonds.bond_ytm(price, 100, 0.0, 60) == pytest.approx(-0.005, abs=1e-10)


def test_invalid_yield_raises():
    with pytest.raises(ValueError):
        bonds.bond_price(100, 0.05, 60, -2.0, freq=1)


# ---------------------------------------------------------------- pv.ipynb snapshot
# Original functions, verbatim from pv.ipynb (commit bcb022d), used as the reference.
def _orig_periods(months, freq):
    return np.arange(1, int(round(months * freq / 12)) + 1)


def _orig_cash_flows(par, coupon_rate, months, freq):
    coupon = par * coupon_rate / freq
    flows = np.full(len(_orig_periods(months, freq)), coupon)
    flows[-1] += par
    return flows


def _orig_after_tax_cash_flows(par, coupon_rate, months, freq, tax_rate):
    coupon = par * coupon_rate / freq
    return _orig_cash_flows(par, coupon_rate, months, freq) - coupon * tax_rate


def _orig_discount_factors(ytm, months, freq):
    return (1 + ytm / freq) ** -_orig_periods(months, freq)


def _orig_present_value(par, coupon_rate, months, ytm, freq=1, tax_rate=0.0):
    flows = _orig_after_tax_cash_flows(par, coupon_rate, months, freq, tax_rate)
    factors = _orig_discount_factors(ytm, months, freq)
    return float(flows @ factors)


def _orig_irr(price, par, coupon_rate, months, freq=1, tax_rate=0.0, tol=1e-12):
    low, high = -freq + 1e-9, 1.0
    while high - low > tol:
        mid = (low + high) / 2
        if _orig_present_value(par, coupon_rate, months, mid, freq, tax_rate) - price > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


PV_NB = dict(par=1_000_000, coupon_rate=0.03, months=36, freq=4)  # docs/07 §7.5 inputs
YTM, TAX = 0.05, 0.15


def test_pv_notebook_snapshot_values():
    assert bonds.present_value(**PV_NB, ytm=YTM) == pytest.approx(944_603.44, abs=CUR)
    assert bonds.present_value(**PV_NB, ytm=YTM, tax_rate=TAX) == pytest.approx(932_139.21, abs=CUR)


def test_pv_notebook_matches_original_functions():
    for y in np.arange(0.02, 0.0801, 0.005):
        for tax in (0.0, TAX):
            assert bonds.present_value(**PV_NB, ytm=y, tax_rate=tax) == pytest.approx(
                _orig_present_value(**PV_NB, ytm=y, tax_rate=tax), abs=1e-6)
    schedule = bonds.bond_schedule(**PV_NB, ytm=YTM, tax_rate=TAX)
    np.testing.assert_allclose(schedule["cash_flow"], _orig_cash_flows(**PV_NB))
    np.testing.assert_allclose(schedule["after_tax_cash_flow"], _orig_after_tax_cash_flows(**PV_NB, tax_rate=TAX))
    np.testing.assert_allclose(schedule["discount_factor"], _orig_discount_factors(YTM, 36, 4))


@pytest.mark.parametrize("price", [850_000, 950_000, 991_000, 1_000_000, 1_150_000])
@pytest.mark.parametrize("tax", [0.0, TAX])
def test_pv_notebook_irr_matches_original(price, tax):
    assert bonds.irr(price, **PV_NB, tax_rate=tax) == pytest.approx(_orig_irr(price, **PV_NB, tax_rate=tax), abs=1e-9)
