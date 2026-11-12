"""python -m riskreport AAPL:0.25 MSFT:0.25 TLT:0.3 GLD:0.2 --out report.html"""
import argparse
from pathlib import Path
import pandas as pd

from .analytics import Portfolio
from .report import render


def parse_holdings(items):
    w = {}
    for it in items:
        t, _, x = it.partition(":")
        w[t.upper()] = float(x) if x else 1.0
    return pd.Series(w)


def download(tickers, start):
    import yfinance as yf
    df = yf.download(sorted(set(tickers)), start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df.dropna()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate a one-page HTML risk report for a portfolio.")
    ap.add_argument("holdings", nargs="+", help="TICKER:WEIGHT pairs (weights are normalized)")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--title", default="Portfolio Risk Report")
    ap.add_argument("--out", default="report.html")
    a = ap.parse_args(argv)

    w = parse_holdings(a.holdings)
    prices = download(list(w.index) + [a.benchmark], a.start)
    p = Portfolio(prices[list(w.index)], w, prices[a.benchmark])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(render(p, a.title, a.benchmark), encoding="utf-8")
    s = p.summary()
    print(f"Wrote {a.out}  |  CAGR {s['Annual return (CAGR)']:.1%}  vol {s['Annual volatility']:.1%}  "
          f"Sharpe {s['Sharpe ratio']:.2f}  max DD {s['Max drawdown']:.1%}")


if __name__ == "__main__":
    main()
