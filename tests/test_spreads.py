"""docs/01 §1.5 and docs/03 §3.10 — ThaiBMA spread loader."""
import pytest

from valuation import bonds, data, spreads


@pytest.fixture(scope="module")
def tables():
    return data.load_spread_tables("ttm")


def test_layout(tables):
    for name in ("spread", "std", "maturity"):
        assert list(tables[name].columns) == ["<=3", "3-5", ">5"]
        assert list(tables[name].index) == list(spreads.RATINGS)
    assert tables["spread"].at["AAA", "<=3"] == pytest.approx(29.970415, abs=1e-6)
    assert tables["spread"].at["BBB-", ">5"] == pytest.approx(513.084925, abs=1e-6)


def test_load_spreads_duration_file():
    t = spreads.load_spreads(data.SPREAD_DURATION_FILE)
    assert t.at["AAA", "<=3"] == pytest.approx(32.889836, abs=1e-6)


@pytest.mark.parametrize("t, bucket", [(0.5, "<=3"), (3.0, "<=3"), (3.01, "3-5"), (5.0, "3-5"), (5.5, ">5")])
def test_maturity_bucket(t, bucket):
    assert spreads.maturity_bucket(t) == bucket


def test_spread_for_returns_decimal(tables):
    assert spreads.spread_for("A", 4, tables["spread"]) == pytest.approx(82.239676 / 10_000, abs=1e-10)


def test_spread_for_nan_falls_back_with_warning(tables):
    with pytest.warns(UserWarning, match="BBB"):
        s = spreads.spread_for("BBB", 7, tables["spread"])
    assert s == pytest.approx(tables["spread"].at["BBB", "3-5"] / 10_000)


def test_corporate_yield_and_price(tables):
    curve = {1: 0.015, 5: 0.020, 10: 0.025}
    assert spreads.gov_yield(curve, 3) == pytest.approx(0.0175)
    y = spreads.corporate_yield(curve, "AA", 5, tables["spread"])
    assert y == pytest.approx(0.020 + tables["spread"].at["AA", "3-5"] / 10_000)
    table = spreads.price_by_rating(100, 0.03, 60, curve, tables["spread"])
    assert table.at["AA", "price"] == pytest.approx(bonds.bond_price(100, 0.03, 60, y))
