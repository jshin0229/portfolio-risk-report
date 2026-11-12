"""Render a Portfolio into a single self-contained HTML file (charts embedded)."""
import base64
import html
import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from .analytics import Portfolio

ACCENT, GREY = "#2563eb", "#9ca3af"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 10})


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _growth(p: Portfolio):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot((1 + p.returns).cumprod() * 10_000, color=ACCENT, lw=1.6, label="Portfolio")
    ax.plot((1 + p.bench_returns).cumprod() * 10_000, color=GREY, lw=1.2, label="Benchmark")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.set_title("Growth of $10,000"); ax.legend(frameon=False)
    return _img(fig)


def _drawdown(p: Portfolio):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.fill_between(p.returns.index, p.drawdown(p.returns), color=ACCENT, alpha=.35, lw=0, label="Portfolio")
    ax.plot(p.drawdown(p.bench_returns), color=GREY, lw=1, label="Benchmark")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.set_title("Drawdown from peak")
    ax.legend(frameon=False, loc="lower left")
    return _img(fig)


def _risk_bars(p: Portfolio):
    rc = p.risk_contributions()[::-1]
    fig, ax = plt.subplots(figsize=(6, .45 * len(rc) + 1))
    y = np.arange(len(rc))
    ax.barh(y + .2, rc["Weight"], .4, color=GREY, label="Capital weight")
    ax.barh(y - .2, rc["Risk contribution"], .4, color=ACCENT, label="Risk contribution")
    ax.set_yticks(y, rc.index); ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Where the risk comes from")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2)
    return _img(fig)


def _corr(p: Portfolio):
    c = p.correlation()
    fig, ax = plt.subplots(figsize=(.6 * len(c) + 2, .55 * len(c) + 1.4))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right"); ax.set_yticks(range(len(c)), c.index)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(c.iat[i, j]) > .6 else "black")
    ax.grid(False); ax.set_title("Correlation of daily returns"); fig.colorbar(im, shrink=.8)
    return _img(fig)


def _monte_carlo(p: Portfolio):
    mc = p.monte_carlo()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(mc))
    ax.fill_between(x, mc.p5, mc.p95, color=ACCENT, alpha=.15, lw=0, label="5th–95th percentile")
    ax.fill_between(x, mc.p25, mc.p75, color=ACCENT, alpha=.3, lw=0, label="25th–75th percentile")
    ax.plot(x, mc.p50, color=ACCENT, lw=1.6, label="Median")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}")); ax.set_xlabel("Trading days ahead")
    ax.set_title("Range of 1-year outcomes for $10,000 (5,000 bootstrapped simulations)"); ax.legend(frameon=False, fontsize=8)
    return _img(fig), mc.iloc[-1]


def _pct(x): return f"{x:.1%}"


def render(p: Portfolio, title: str = "Portfolio Risk Report", bench_name: str = "SPY") -> str:
    s, sb = p.summary(), p.summary(p.bench_returns)
    fmt = {"Sharpe ratio": "{:.2f}", "Beta vs benchmark": "{:.2f}", "Correlation to benchmark": "{:.2f}"}
    stat_rows = "".join(
        f"<tr><td>{k}</td><td>{fmt.get(k, '{:.1%}').format(v)}</td><td>{fmt.get(k, '{:.1%}').format(sb[k])}</td></tr>"
        for k, v in s.items())
    rc = p.risk_contributions()
    rc_rows = "".join(f"<tr><td>{html.escape(t)}</td><td>{_pct(r.Weight)}</td><td>{_pct(r.Volatility)}</td>"
                      f"<td>{_pct(r['Risk contribution'])}</td></tr>" for t, r in rc.iterrows())
    st = p.stress_tests()
    st_rows = "".join(f"<tr><td>{n}</td><td class='{'neg' if r.Portfolio < 0 else 'pos'}'>{_pct(r.Portfolio)}</td>"
                      f"<td>{_pct(r.Benchmark)}</td></tr>" for n, r in st.iterrows()) or \
        "<tr><td colspan=3>Not enough history for the stress periods.</td></tr>"
    mc_img, mc_end = _monte_carlo(p)
    top = rc.index[0]
    insights = [
        f"<b>{top}</b> is {_pct(rc.loc[top, 'Weight'])} of the money but {_pct(rc.loc[top, 'Risk contribution'])} of the risk.",
        f"On a bad day (worst 5%), expect to lose at least <b>{_pct(s['1-day VaR (95%)'])}</b>; the average of those bad days is {_pct(s['1-day CVaR (95%)'])}.",
        f"Beta of <b>{s['Beta vs benchmark']:.2f}</b>: when {bench_name} moves 1%, this portfolio has typically moved {s['Beta vs benchmark']:.2f}%.",
        f"After one year, the middle 50% of simulated outcomes for $10,000 range from <b>${mc_end.p25:,.0f}</b> to <b>${mc_end.p75:,.0f}</b>.",
    ]
    holdings = ", ".join(f"{t} {w:.0%}" for t, w in p.weights.items())
    start, end = p.returns.index[0].date(), p.returns.index[-1].date()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{--bg:#fafaf9;--card:#fff;--text:#1c1917;--muted:#78716c;--border:#e7e5e4;--accent:{ACCENT}}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}}
.wrap{{max-width:980px;margin:0 auto;padding:32px 16px}}
h1{{margin:0;font-size:28px}} .sub{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;margin-top:16px;overflow-x:auto}}
.card h2{{font-size:16px;margin:0 0 10px}} img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{padding:6px 8px;border-bottom:1px solid var(--border);text-align:right}}
td:first-child,th:first-child{{text-align:left}} th{{color:var(--muted);font-weight:500}}
.neg{{color:#dc2626}} .pos{{color:#16a34a}} ul{{margin:0;padding-left:20px}} li{{margin:4px 0}}
footer{{color:var(--muted);font-size:12px;margin-top:24px}}
</style></head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(holdings)} · benchmark {bench_name} · {start} to {end}</p>
<div class="card"><h2>Key takeaways</h2><ul>{''.join(f'<li>{i}</li>' for i in insights)}</ul></div>
<div class="card"><img src="{_growth(p)}" alt="Growth chart"><img src="{_drawdown(p)}" alt="Drawdown chart"></div>
<div class="grid">
<div class="card"><h2>Risk & return</h2><table><tr><th></th><th>Portfolio</th><th>{bench_name}</th></tr>{stat_rows}</table></div>
<div class="card"><h2>Stress tests</h2><table><tr><th>Period</th><th>Portfolio</th><th>{bench_name}</th></tr>{st_rows}</table></div>
</div>
<div class="grid">
<div class="card"><h2>Holdings</h2><table><tr><th>Ticker</th><th>Weight</th><th>Volatility</th><th>Risk share</th></tr>{rc_rows}</table>
<img src="{_risk_bars(p)}" alt="Risk contribution chart"></div>
<div class="card"><img src="{_corr(p)}" alt="Correlation heatmap"></div>
</div>
<div class="card"><img src="{mc_img}" alt="Monte Carlo chart"></div>
<footer>Generated {date.today()} by portfolio-risk-report. Daily rebalanced to target weights; historical results and simulations are not predictions. Not investment advice.</footer>
</div></body></html>"""
