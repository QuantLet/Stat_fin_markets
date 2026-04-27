"""
SFM_ch4_rw_sim
==============
Random walk simulation + comparison with real S&P 500.

Description:
- Simulate a random walk with iid Gaussian increments
- Also download a real S&P 500 price series
- Compare visually: both look similar (stochastic trend)
- Plot returns: stationary, oscillating around zero

Output:
- sfm_ch4_rw_sim.pdf
"""
import numpy as np
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

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})

np.random.seed(2026)

df = yf.download("^GSPC", start="2020-01-01", end="2022-01-01",
                  auto_adjust=True, progress=False)
prices_real = df["Close"].squeeze().dropna()
r_real = np.log(prices_real).diff().dropna()
# Align so both have the same length (drop t=0 of prices)
prices_aligned = prices_real.iloc[1:]
T = len(r_real)
t = np.arange(T)

mu = r_real.mean()
sigma = r_real.std()
eps = np.random.normal(mu, sigma, T)
log_P_sim = np.cumsum(eps) + np.log(prices_real.iloc[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
ax1.plot(t, prices_aligned.values, color=MAIN_BLUE, linewidth=1.4,
         label=r"S&P 500 real (2020--2022)")
ax1.plot(t, np.exp(log_P_sim), color=FOREST, linewidth=1.2, linestyle="--",
         label=r"RW simulated $P_t = P_{t-1} e^{\varepsilon_t}$")
ax1.set_xlabel(r"$t$ (day)")
ax1.set_ylabel(r"$P_t$")
ax1.set_title(r"Price: real S&P 500 vs simulated RW")
ax1.grid(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
           ncol=1, frameon=False)

ax2.plot(t, r_real.values * 100, color=CRIMSON, linewidth=0.7,
         label=r"$r_t$ real (%)")
ax2.axhline(0, color="black", linewidth=0.4)
ax2.set_xlabel(r"$t$ (day)")
ax2.set_ylabel(r"$r_t$ (%)")
ax2.set_title("S&P 500 returns: stationary, near-iid")
ax2.grid(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
           ncol=1, frameon=False)
fig.tight_layout()
fig.savefig(CHARTS / "sfm_ch4_rw_sim.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print(f"Saved. Real S&P 500 mean r = {mu*100:.3f}%, sigma = {sigma*100:.3f}%")
