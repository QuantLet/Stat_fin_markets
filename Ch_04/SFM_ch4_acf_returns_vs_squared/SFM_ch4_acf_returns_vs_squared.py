"""
SFM_ch4_acf_returns_vs_squared
==============================
ACF of returns vs ACF of squared returns --- EMH and GARCH coexist.

Description:
- Download S&P 500 (^GSPC) daily 2000-2024 via yfinance
- Compute ACF of r_t (returns) and r_t^2 (squared returns) up to lag 30
- Two-panel plot with +/-1.96/sqrt(n) Bartlett bands on each axis
- Returns ACF inside band; squared returns ACF clearly above band

Output:
- sfm_ch4_acf_returns_vs_squared.pdf
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

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})


def acf(x, max_lag):
    x = np.asarray(x) - np.mean(x)
    g0 = (x * x).mean()
    return np.array([(x[k:] * x[:-k]).mean() / g0
                     for k in range(1, max_lag + 1)])


df = yf.download("^GSPC", start="2000-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)
prices = df["Close"].squeeze().dropna()
r = np.log(prices).diff().dropna().values
r2 = r ** 2
n = len(r)
ci = 1.96 / np.sqrt(n)
max_lag = 30
lags = np.arange(1, max_lag + 1)

acf_r = acf(r, max_lag)
acf_r2 = acf(r2, max_lag)
print(f"S&P 500 n = {n}; max |rho_r| = {np.max(np.abs(acf_r)):.4f}; "
      f"max rho_r2 = {np.max(acf_r2):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=False)

ax1 = axes[0]
sig1 = np.abs(acf_r) > ci
ax1.bar(lags[~sig1], acf_r[~sig1], width=0.65, color=MAIN_BLUE,
        edgecolor="black", linewidth=0.4, label="not significant")
if sig1.any():
    ax1.bar(lags[sig1], acf_r[sig1], width=0.65, color=CRIMSON,
            edgecolor="black", linewidth=0.4, label="significant")
ax1.axhline(y=ci, color="gray", linestyle="--", linewidth=1,
            label=fr"$\pm 1.96/\sqrt{{n}}$")
ax1.axhline(y=-ci, color="gray", linestyle="--", linewidth=1)
ax1.axhline(y=0, color="black", linewidth=0.5)
ax1.set_xlabel("lag $k$")
ax1.set_ylabel(r"$\hat\rho_k(r_t)$")
ax1.set_title(r"(a) ACF of returns $r_t$ --- weak-form EMH")
ax1.set_ylim(-0.2, 0.2)
ax1.grid(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
           ncol=3, frameon=False)

ax2 = axes[1]
sig2 = acf_r2 > ci
ax2.bar(lags[~sig2], acf_r2[~sig2], width=0.65, color=MAIN_BLUE,
        edgecolor="black", linewidth=0.4, label="not significant")
if sig2.any():
    ax2.bar(lags[sig2], acf_r2[sig2], width=0.65, color=CRIMSON,
            edgecolor="black", linewidth=0.4, label="significant")
ax2.axhline(y=ci, color="gray", linestyle="--", linewidth=1,
            label=fr"$\pm 1.96/\sqrt{{n}}$")
ax2.axhline(y=-ci, color="gray", linestyle="--", linewidth=1)
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.set_xlabel("lag $k$")
ax2.set_ylabel(r"$\hat\rho_k(r_t^2)$")
ax2.set_title(r"(b) ACF of squared returns $r_t^2$ --- volatility clustering")
ax2.grid(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
           ncol=3, frameon=False)

fig.tight_layout()
fig.savefig(CHARTS / "sfm_ch4_acf_returns_vs_squared.pdf",
            bbox_inches="tight", pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_acf_returns_vs_squared.pdf")
