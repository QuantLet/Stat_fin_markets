"""
SFM_ch4_rw_vs_nonrw
===================
Random walk vs non-random walk: visual + ACF comparison.

Description:
- Simulate three return processes (T = 1500):
    (a) Pure random walk: r_t = epsilon_t (iid Normal)
    (b) Momentum: r_t = 0.4 r_{t-1} + epsilon_t  (positive AR(1))
    (c) Mean reversion: r_t = -0.4 r_{t-1} + epsilon_t  (negative AR(1))
- Plus real S&P 500 daily 2018-2025
- Build cumulative log-prices: P_t = P_0 * exp(sum r_s)
- Top row: 4 price paths (visually all "trending")
- Bottom row: 4 ACF panels of the returns with +/- 1.96/sqrt(n) bands
- Pedagogical message: paths look similar; only ACF separates RW from non-RW

Output:
- sfm_ch4_rw_vs_nonrw.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

try:
    CHARTS = Path(__file__).resolve().parents[3] / "charts"
except NameError:
    CHARTS = Path.cwd().resolve().parents[2] / "charts"
CHARTS.mkdir(exist_ok=True)

MAIN_BLUE = (26 / 255, 58 / 255, 110 / 255)
CRIMSON = (205 / 255, 0 / 255, 0 / 255)
FOREST = (46 / 255, 125 / 255, 50 / 255)
AMBER = (181 / 255, 133 / 255, 63 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5, "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})


def acf(x, max_lag):
    x = np.asarray(x) - np.mean(x)
    g0 = (x * x).mean()
    return np.array([(x[k:] * x[:-k]).mean() / g0
                     for k in range(1, max_lag + 1)])


rng = np.random.default_rng(2026)
T = 1500
sigma = 0.012  # daily std

# (a) Pure random walk
eps = rng.normal(0, sigma, T)
r_rw = eps.copy()

# (b) Momentum AR(1) with phi = +0.4
phi_mom = 0.4
r_mom = np.zeros(T)
eps2 = rng.normal(0, sigma * np.sqrt(1 - phi_mom**2), T)
for t in range(1, T):
    r_mom[t] = phi_mom * r_mom[t - 1] + eps2[t]

# (c) Mean reversion AR(1) with phi = -0.4
phi_mr = -0.4
r_mr = np.zeros(T)
eps3 = rng.normal(0, sigma * np.sqrt(1 - phi_mr**2), T)
for t in range(1, T):
    r_mr[t] = phi_mr * r_mr[t - 1] + eps3[t]

# (d) Real S&P 500
df = yf.download("^GSPC", start="2018-01-01", end="2025-12-31",
                 auto_adjust=True, progress=False)
prices_real = df["Close"].squeeze().dropna()
r_real = np.log(prices_real).diff().dropna().values
n_real = len(r_real)
print(f"Real S&P 500: n = {n_real}")

# Cumulative log-prices, normalized to start at 100
def to_price(r):
    return 100 * np.exp(np.cumsum(r))


P_rw = to_price(r_rw)
P_mom = to_price(r_mom)
P_mr = to_price(r_mr)
P_real = 100 * prices_real / prices_real.iloc[0]

# ACF
max_lag = 20
acf_rw = acf(r_rw, max_lag)
acf_mom = acf(r_mom, max_lag)
acf_mr = acf(r_mr, max_lag)
acf_real = acf(r_real, max_lag)
ci_sim = 1.96 / np.sqrt(T)
ci_real = 1.96 / np.sqrt(n_real)
lags = np.arange(1, max_lag + 1)

print(f"rho_1 RW = {acf_rw[0]:.3f}, momentum = {acf_mom[0]:.3f}, "
      f"mean rev = {acf_mr[0]:.3f}, real = {acf_real[0]:.3f}")

fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))

panels = [
    (P_rw, r_rw, acf_rw, ci_sim, MAIN_BLUE,
     r"(a) Random walk: $r_t = \varepsilon_t$"),
    (P_mom, r_mom, acf_mom, ci_sim, FOREST,
     r"(b) Momentum: $r_t = 0.4\,r_{t-1} + \varepsilon_t$"),
    (P_mr, r_mr, acf_mr, ci_sim, AMBER,
     r"(c) Mean reversion: $r_t = -0.4\,r_{t-1} + \varepsilon_t$"),
    (P_real, r_real, acf_real, ci_real, CRIMSON,
     r"(d) Real S&P 500 (2018--2025)"),
]

for col, (P, r, acfv, ci, color, title) in enumerate(panels):
    # Top: price path
    ax = axes[0, col]
    ax.plot(np.arange(len(P)), P, color=color, linewidth=1.0)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("$t$ (days)")
    if col == 0:
        ax.set_ylabel(r"$P_t$ (start = 100)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bottom: ACF of returns
    ax = axes[1, col]
    sig = np.abs(acfv) > ci
    ax.bar(lags[~sig], acfv[~sig], width=0.7, color=color, alpha=0.7,
           edgecolor="black", linewidth=0.3)
    if sig.any():
        ax.bar(lags[sig], acfv[sig], width=0.7, color=CRIMSON, alpha=0.85,
               edgecolor="black", linewidth=0.3)
    ax.axhline(y=ci, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(y=-ci, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(y=0, color="black", linewidth=0.4)
    ax.set_xlabel("lag $k$")
    if col == 0:
        ax.set_ylabel(r"$\hat\rho_k$")
    ax.set_ylim(-0.55, 0.55)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(r"Random walk vs non-random walk --- price paths (top) and return ACF (bottom)",
             fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig(CHARTS / "sfm_ch4_rw_vs_nonrw.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_rw_vs_nonrw.pdf")
