"""
SFM_ch4_active_passive
======================
S&P 500 vs average active mutual fund cumulative return.

Description:
- Download real S&P 500 total return index (^SP500TR) 2005-2024 via yfinance
- Proxy the average active mutual fund with underperformance of 1.3%/year
  (SPIVA Scorecard U.S. median underperformance 2024)
- Plot cumulative value of $1 invested in 2005 for both

Output:
- sfm_ch4_active_passive.pdf
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
AMBER = (181 / 255, 133 / 255, 63 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})

np.random.seed(7)

df = yf.download("^SP500TR", start="2005-01-01", end="2024-12-31",
                  auto_adjust=True, progress=False)
prices = df["Close"].squeeze().dropna()
# Annual returns
annual = prices.resample("YE").last().pct_change().dropna()
years = annual.index.year.values
rets_sp = annual.values

# SPIVA: active median underperforms ~1.3% annually
underperf = 0.013
rets_active = rets_sp - underperf + np.random.normal(0, 0.015, len(rets_sp))
cum_sp = np.cumprod(1 + rets_sp)
cum_act = np.cumprod(1 + rets_active)

fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(years, cum_sp, color=MAIN_BLUE, linewidth=2, marker="o",
        markersize=4, label=r"S&P 500 TR (passive)")
ax.plot(years, cum_act, color=CRIMSON, linewidth=2, marker="s",
        markersize=4, label="active funds median (SPIVA)")
ax.fill_between(years, cum_act, cum_sp, where=(cum_sp > cum_act),
                color=AMBER, alpha=0.22, label="performance gap")
ax.set_xlabel("Year")
ax.set_ylabel(r"cumulative value (\$1 invested in 2005)")
ax.set_title(r"Active vs Passive --- SPIVA: $>90\%$ underperform")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_active_passive.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print(f"S&P 500 TR final: ${cum_sp[-1]:.2f}")
print(f"Active mean final: ${cum_act[-1]:.2f}")
print(f"Gap: ${cum_sp[-1] - cum_act[-1]:.2f}")
