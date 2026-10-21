"""docs/07 §7.5 — TVM tests, plus docs/02 extras."""
import math

import pytest

from valuation import tvm

CUR = 0.01
RATE = 1e-4


@pytest.mark.parametrize("args, kwargs, expected", [
    ((100, 0.10, 1), {}, 110.00),                        # slide 6
    ((100, 0.10, 2), {}, 121.00),                        # slide 8
])
def test_fv(args, kwargs, expected):
    assert tvm.fv(*args, **kwargs) == pytest.approx(expected, abs=CUR)


@pytest.mark.parametrize("args, kwargs, expected", [
    ((250, 0.15, 2), {}, 189.04),                        # Mishkin simple PV
    ((100, 0.067, 20), {}, 27.33),                       # CFA LM2 Ex 1
    ((100, -0.0005, 10), {}, 100.50),                    # negative rate
    ((5_000_000, 0.06, 10), {"m": 12}, 2_748_163.67),    # monthly compounding
])
def test_pv(args, kwargs, expected):
    assert tvm.pv(*args, **kwargs) == pytest.approx(expected, abs=CUR)


def test_pv_returns_python_float():
    assert type(tvm.pv(100, 0.1, 1)) is float


def test_annuity_due_lottery():
    assert tvm.annuity_pv(1e6, 0.10, 20, due=True) == pytest.approx(9_364_920.09, abs=CUR)


def test_annuity_due_equals_stream_from_t0():
    assert tvm.annuity_pv(1e6, 0.10, 20, due=True) == pytest.approx(tvm.npv(0.10, [1e6] * 20), abs=CUR)


@pytest.mark.parametrize("args, expected", [
    ((100_000, 0.07, 20), 9_439.29),
    ((800_000, 0.0525 / 12, 360), 4_417.63),
])
def test_annuity_payment(args, expected):
    assert tvm.annuity_payment(*args) == pytest.approx(expected, abs=CUR)


def test_annuity_rate():
    assert tvm.annuity_rate(1_000, 85.81, 25) == pytest.approx(0.0700, abs=RATE)


def test_annuity_zero_rate():
    assert tvm.annuity_pv(100, 0.0, 10) == 1000
    assert tvm.annuity_payment(1000, 0.0, 10) == 100


def test_mortgage_schedule():
    s = tvm.amortization_schedule(800_000, 0.0525 / 12, 360)
    assert s.loc[0, "interest"] == pytest.approx(3_500.00, abs=CUR)
    assert s.loc[1, "interest"] == pytest.approx(3_495.99, abs=CUR)
    assert s["balance"].iloc[-1] == pytest.approx(0.0, abs=CUR)


def test_perpetuity():
    assert tvm.perpetuity_pv(100, 0.10) == pytest.approx(1_000, abs=CUR)
    assert tvm.perpetuity_pv(100, 0.20) == pytest.approx(500, abs=CUR)
    assert tvm.perpetuity_rate(100, 2_000) == pytest.approx(0.05, abs=RATE)


def test_growing_perpetuity_requires_rate_above_g():
    with pytest.raises(ValueError):
        tvm.growing_perpetuity_pv(1, 0.05, 0.05)


@pytest.mark.parametrize("r, c, expected", [
    (0.002, 52, 0.1095),
    (0.20, 2 / 3, 0.1292),
    (0.062, 365 / 100, 0.2455),
])
def test_annualize(r, c, expected):
    assert tvm.annualize(r, c) == pytest.approx(expected, abs=RATE)


def test_log_return():
    assert tvm.log_return(30, 34.50) == pytest.approx(0.139762, abs=1e-6)


def test_log_returns_add_up():
    assert tvm.log_return(30, 31) + tvm.log_return(31, 34.5) == pytest.approx(tvm.log_return(30, 34.5), abs=1e-12)


def test_forward_rate():
    assert tvm.forward_rate(0.02, 1, 0.03, 2) == pytest.approx(0.040098, abs=1e-6)


def test_ear():
    assert tvm.ear(0.12, 12) == pytest.approx(0.126825, abs=1e-6)
    assert tvm.ear(0.12, math.inf) == pytest.approx(math.exp(0.12) - 1, abs=1e-12)


def test_pv_stream_with_spot_curve():
    cfs, z = [5, 105], [0.02, 0.03]
    assert tvm.pv_stream(cfs, z) == pytest.approx(5 / 1.02 + 105 / 1.03 ** 2, abs=1e-10)


def test_irr():
    assert tvm.npv(tvm.irr([-100, 60, 60]), [-100, 60, 60]) == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("call", [
    lambda: tvm.pv(100, -1.5, 1),             # 1 + rate/m <= 0
    lambda: tvm.pv(100, 0.05, -1),            # negative periods
    lambda: tvm.pv_stream([], 0.05),          # empty cash flows
])
def test_input_validation(call):
    with pytest.raises(ValueError):
        call()
