"""Portfolio risk analytics on daily returns.

Everything here is plain pandas/numpy so each number in the report can be traced
back to a formula.
"""
from dataclasses import dataclass
from functools import cached_property
import numpy as np
import pandas as pd

TD = 252
STRESS_PERIODS = {
    "2008 financial crisis": ("2007-10-09", "2009-03-09"),
    "2020 COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate hikes": ("2022-01-03", "2022-10-12"),
}


@dataclass
class Portfolio:
    prices: pd.DataFrame          # daily prices, one column per holding
    weights: pd.Series            # target weights, rebalanced daily (sum to 1)
    benchmark: pd.Series          # daily benchmark prices (e.g. SPY)
    rf: float = 0.0               # annual risk-free rate for Sharpe

    def __post_init__(self):
        self.weights = self.weights / self.weights.sum()
        missing = set(self.weights.index) - set(self.prices.columns)
        if missing:
            raise ValueError(f"No prices for: {sorted(missing)}")

    # ---------- returns ----------
    @cached_property
    def asset_returns(self) -> pd.DataFrame:
        return self.prices[self.weights.index].pct_change().dropna()

    @cached_property
    def returns(self) -> pd.Series:
        return (self.asset_returns @ self.weights).rename("Portfolio")

    @cached_property
    def bench_returns(self) -> pd.Series:
        return self.benchmark.pct_change().reindex(self.returns.index).fillna(0).rename("Benchmark")

    # ---------- headline stats ----------
    @staticmethod
    def drawdown(r: pd.Series) -> pd.Series:
        eq = (1 + r).cumprod()
        return eq / eq.cummax() - 1

    def summary(self, r: pd.Series | None = None) -> dict:
        r = self.returns if r is None else r
        years = len(r) / TD
        vol = r.std() * np.sqrt(TD)
        cagr = (1 + r).prod() ** (1 / years) - 1
        var95 = -np.percentile(r, 5)
        cvar95 = -r[r <= -var95].mean()
        b = self.bench_returns.loc[r.index]
        beta = np.cov(r, b)[0, 1] / b.var()
        return {
            "Annual return (CAGR)": cagr,
            "Annual volatility": vol,
            "Sharpe ratio": (r.mean() * TD - self.rf) / vol if vol else 0.0,
            "Max drawdown": self.drawdown(r).min(),
            "Beta vs benchmark": beta,
            "1-day VaR (95%)": var95,
            "1-day CVaR (95%)": cvar95,
            "Correlation to benchmark": r.corr(b),
        }

    # ---------- risk decomposition ----------
    @cached_property
    def covariance(self) -> pd.DataFrame:
        return self.asset_returns.cov() * TD

    def risk_contributions(self) -> pd.DataFrame:
        """Share of total portfolio variance from each holding (sums to 100%)."""
        w = self.weights
        cov = self.covariance
        port_var = float(w @ cov @ w)
        marginal = cov @ w
        contrib = w * marginal / port_var
        vol = np.sqrt(np.diag(cov))
        return pd.DataFrame({"Weight": w, "Volatility": vol, "Risk contribution": contrib},
                            index=w.index).sort_values("Risk contribution", ascending=False)

    def correlation(self) -> pd.DataFrame:
        return self.asset_returns.corr()

    # ---------- scenarios ----------
    def stress_tests(self) -> pd.DataFrame:
        rows = {}
        for name, (a, b) in STRESS_PERIODS.items():
            r = self.returns.loc[a:b]
            if len(r) < 5 or r.index[0] > pd.Timestamp(a) + pd.Timedelta(days=7):
                continue  # portfolio didn't exist yet
            rows[name] = {"Portfolio": (1 + r).prod() - 1,
                          "Benchmark": (1 + self.bench_returns.loc[a:b]).prod() - 1}
        return pd.DataFrame(rows).T

    def monte_carlo(self, years=1, sims=5000, seed=7) -> pd.DataFrame:
        """Bootstrap daily returns to simulate a range of outcomes for $10,000."""
        rng = np.random.default_rng(seed)
        r = self.returns.to_numpy()
        n = int(TD * years)
        paths = 10_000 * np.cumprod(1 + rng.choice(r, size=(sims, n)), axis=1)
        pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0).T
        return pd.DataFrame(pct, columns=["p5", "p25", "p50", "p75", "p95"])
