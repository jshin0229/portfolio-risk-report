import numpy as np
import pandas as pd
import pytest
from riskreport import Portfolio
from riskreport.cli import parse_holdings


def make(seed=0, n=800):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    m = rng.normal(.0004, .01, n)
    a = 100 * np.cumprod(1 + m + rng.normal(0, .005, n))
    b = 100 * np.cumprod(1 + rng.normal(.0002, .004, n))
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    bench = pd.Series(100 * np.cumprod(1 + m), index=idx)
    return prices, bench


def test_weights_are_normalized():
    prices, bench = make()
    p = Portfolio(prices, pd.Series({"A": 3, "B": 1}), bench)
    assert p.weights.sum() == pytest.approx(1) and p.weights["A"] == pytest.approx(.75)


def test_risk_contributions_sum_to_one():
    prices, bench = make()
    rc = Portfolio(prices, pd.Series({"A": .5, "B": .5}), bench).risk_contributions()
    assert rc["Risk contribution"].sum() == pytest.approx(1)


def test_benchmark_beta_is_one():
    prices, bench = make()
    prices["SPY"] = bench
    p = Portfolio(prices, pd.Series({"SPY": 1.0}), bench)
    assert p.summary()["Beta vs benchmark"] == pytest.approx(1, abs=1e-9)


def test_cvar_at_least_var():
    prices, bench = make()
    s = Portfolio(prices, pd.Series({"A": .6, "B": .4}), bench).summary()
    assert s["1-day CVaR (95%)"] >= s["1-day VaR (95%)"] > 0


def test_missing_ticker_raises():
    prices, bench = make()
    with pytest.raises(ValueError):
        Portfolio(prices, pd.Series({"ZZZ": 1.0}), bench)


def test_monte_carlo_percentiles_ordered():
    prices, bench = make()
    mc = Portfolio(prices, pd.Series({"A": .5, "B": .5}), bench).monte_carlo(sims=500)
    last = mc.iloc[-1]
    assert last.p5 <= last.p25 <= last.p50 <= last.p75 <= last.p95


def test_parse_holdings():
    w = parse_holdings(["aapl:0.4", "msft:0.6"])
    assert list(w.index) == ["AAPL", "MSFT"] and w.sum() == pytest.approx(1)
