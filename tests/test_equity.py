"""docs/07 §7.5 — equity tests (docs/04)."""
import pytest

from valuation import equity

CUR = 0.01
RATE = 1e-4


def test_one_period_price():
    assert equity.one_period_price(0.16, 60, 0.12) == pytest.approx(53.71, abs=CUR)


def test_ddm_explicit_far_terminal_value():
    assert equity.ddm_explicit([0] * 75, 0.12, terminal_price=50) == pytest.approx(0.0102, abs=RATE)


@pytest.mark.parametrize("ke, expected", [(0.15, 16.67), (0.12, 22.22), (0.10, 28.57)])
def test_gordon_mishkin(ke, expected):
    assert equity.gordon_price(ke, 0.03, d1=2) == pytest.approx(expected, abs=CUR)


def test_gordon_shipline():
    assert equity.gordon_price(0.15, 0.06, d0=1.50) == pytest.approx(17.67, abs=CUR)
    assert equity.gordon_price(0.15, 0.0, d1=1.50) == pytest.approx(10.00, abs=CUR)


@pytest.mark.parametrize("g_long, expected", [(0.0, 11.66), (0.02, 13.05), (0.04, 14.94), (0.06, 17.67)])
def test_two_stage_ddm(g_long, expected):
    assert equity.two_stage_ddm(1.5, 0.06, 3, g_long, 0.15) == pytest.approx(expected, abs=CUR)


def test_two_stage_breakdown():
    d = equity.two_stage_ddm(1.5, 0.06, 3, 0.02, 0.15, detail=True)
    assert d["pv_stage1"] == pytest.approx(3.832, abs=1e-3)
    assert d["pv_terminal"] == pytest.approx(9.22, abs=CUR)


def test_ddm_explicit_converges_to_gordon():
    divs = equity.growing_dividends(1.0, 0.03, 2000)
    assert equity.ddm_explicit(divs, 0.10) == pytest.approx(equity.gordon_price(0.10, 0.03, d0=1.0), abs=1e-6)


def test_implied_return_and_growth():
    assert equity.gordon_implied_return(63, 1.76, 0.04) == pytest.approx(0.0679, abs=RATE)
    assert equity.gordon_implied_growth(63, 1.76, 0.07) == pytest.approx(0.0421, abs=RATE)


def test_pe_link():
    assert equity.justified_forward_pe(0.70, 0.09, 0.045) == pytest.approx(15.56, abs=CUR)
    assert equity.pe_implied_return(28, 0.70, 0.04) == pytest.approx(0.0650, abs=RATE)
    assert equity.pe_implied_growth(28, 0.70, 0.065) == pytest.approx(0.0400, abs=RATE)
    assert equity.pe_implied_growth(19, 0.60, 0.08) == pytest.approx(0.0484, abs=RATE)


@pytest.mark.parametrize("ke, g", [(0.05, 0.05), (0.04, 0.06)])
def test_gordon_raises_when_ke_not_above_g(ke, g):
    with pytest.raises(ValueError):
        equity.gordon_price(ke, g, d1=1.0)


def test_gordon_needs_exactly_one_dividend():
    with pytest.raises(ValueError):
        equity.gordon_price(0.1, 0.02)
    with pytest.raises(ValueError):
        equity.gordon_price(0.1, 0.02, d1=1, d0=1)


def test_implied_growth_from_d0_round_trip():
    price = equity.gordon_price(0.09, 0.025, d0=2.0)
    assert equity.gordon_implied_growth_from_d0(price, 2.0, 0.09) == pytest.approx(0.025, abs=1e-12)
