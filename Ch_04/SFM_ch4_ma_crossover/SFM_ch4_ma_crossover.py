"""
SFM_ch4_ma_crossover
====================
Moving Average (MA) crossover strategy on S&P 500 (real data).

Description:
- Download S&P 500 (^GSPC) 2010-2024 via yfinance
- Compute MA(50) and MA(200) ("Golden Cross" / "Death Cross")
- Signal: long when MA(50) > MA(200), flat otherwise
- Compare cumulative returns of strategy vs buy-and-hold
- Include cost per trade (0.05%) to show cost impact

Output:
- sfm_ch4_ma_crossover.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path

try:
    CHARTS = Path(__file__).resolve().parents[3] / "charts"
except NameError:
    CHARTS = Path.cwd().resolve().parents[2] / "charts"
CHARTS.mkdir(exist_ok=True)

MAIN_BLUE = (26 / 255, 58 / 255, 110 / 255)
CRIMSON = (220 / 255, 53 / 255, 69 / 255)
FOREST = (46 / 255, 125 / 255, 50 / 255)
AMBER = (181 / 255, 133 / 255, 63 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})

df = yf.download("^GSPC", start="2010-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)
P = df["Close"].squeeze().dropna()
ma_short = P.rolling(50).mean()
ma_long = P.rolling(200).mean()

# Signals
signal = (ma_short > ma_long).astype(int)
# Daily returns
r = np.log(P).diff().fillna(0.0)
# Strategy returns: signal aplies next day
strat = signal.shift(1).fillna(0) * r
# Cost: 0.05% per trade (round-trip tracked via signal changes)
changes = signal.diff().abs().fillna(0)
cost = changes * 0.0005
strat_net = strat - cost

# Cumulative
cum_bh = np.exp(r.cumsum())
cum_strat_gross = np.exp(strat.cumsum())
cum_strat_net = np.exp(strat_net.cumsum())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Top: price + MAs
ax1.plot(P.index, P, color="gray", linewidth=0.7, alpha=0.6, label=r"S&P 500")
ax1.plot(ma_short.index, ma_short, color=MAIN_BLUE, linewidth=1.3,
         label=r"MA(50)")
ax1.plot(ma_long.index, ma_long, color=CRIMSON, linewidth=1.5,
         label=r"MA(200)")
# Golden crosses
gc = (signal.diff() == 1)
dc = (signal.diff() == -1)
ax1.scatter(P.index[gc], P[gc], s=40, color=FOREST, marker="^",
            zorder=5, label="Golden Cross")
ax1.scatter(P.index[dc], P[dc], s=40, color=CRIMSON, marker="v",
            zorder=5, label="Death Cross")
ax1.set_ylabel(r"price $P_t$")
ax1.set_title(r"S&P 500 with MA(50) and MA(200) --- Golden/Death Cross")
ax1.grid(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5,
           frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.0)

# Bottom: cumulative returns
ax2.plot(cum_bh.index, cum_bh, color=MAIN_BLUE, linewidth=1.6,
         label="Buy \\& Hold")
ax2.plot(cum_strat_gross.index, cum_strat_gross, color=FOREST, linewidth=1.6,
         label="MA gross")
ax2.plot(cum_strat_net.index, cum_strat_net, color=CRIMSON, linewidth=1.6,
         label="MA net (5bp/trade)")
ax2.set_xlabel("Year")
ax2.set_ylabel(r"cumulative value (\$1 invested)")
ax2.set_title(r"Strategy performance vs Buy \& Hold")
ax2.grid(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
           frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.2)

fig.subplots_adjust(bottom=0.22, wspace=0.25)
fig.savefig(CHARTS / "sfm_ch4_ma_crossover.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)

# Print stats
annual = 252
total_years = (P.index[-1] - P.index[0]).days / 365.25
ret_bh = cum_bh.iloc[-1] ** (1 / total_years) - 1
ret_strat_gross = cum_strat_gross.iloc[-1] ** (1 / total_years) - 1
ret_strat_net = cum_strat_net.iloc[-1] ** (1 / total_years) - 1
sigma_bh = r.std() * np.sqrt(annual)
sigma_strat = strat.std() * np.sqrt(annual)
n_trades = int(changes.sum())

print(f"Period: {P.index[0].date()} - {P.index[-1].date()} ({total_years:.1f} years)")
print(f"Trades: {n_trades}")
print(f"Buy & Hold:   CAGR={ret_bh:.2%}, sigma={sigma_bh:.2%}")
print(f"MA gross:     CAGR={ret_strat_gross:.2%}, sigma={sigma_strat:.2%}")
print(f"MA net cost:  CAGR={ret_strat_net:.2%}")
