"""
SFM_ch4_acf_sp500
=================
ACF of S&P 500 daily returns --- weak-form EMH evidence.

Description:
- Download S&P 500 (^GSPC) daily data 2000-2024 via yfinance
- Compute log-returns and their ACF up to lag 20
- Overlay 1.96/sqrt(n) Bartlett confidence bands
- Most autocorrelations inside bands --> compatible with weak-form EMH

Output:
- sfm_ch4_acf_sp500.pdf --- stored in ../../charts/
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

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})

# ---- Download real data ----
df = yf.download("^GSPC", start="2000-01-01", end="2024-12-31",
                  auto_adjust=True, progress=False)
prices = df["Close"].squeeze().dropna()
r = np.log(prices).diff().dropna().values
n = len(r)

# ---- ACF ----
max_lag = 20
mu = r.mean()
gamma0 = np.mean((r - mu) ** 2)
rhos = np.array([
    np.mean((r[k:] - mu) * (r[:-k] - mu)) / gamma0
    for k in range(1, max_lag + 1)
])
lags = np.arange(1, max_lag + 1)
ci = 1.96 / np.sqrt(n)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(6.8, 4.2))
sig_mask = np.abs(rhos) > ci
ax.bar(lags[~sig_mask], rhos[~sig_mask], width=0.6,
       color=MAIN_BLUE, edgecolor="black", linewidth=0.4,
       label="not significant")
if sig_mask.any():
    ax.bar(lags[sig_mask], rhos[sig_mask], width=0.6,
           color=CRIMSON, edgecolor="black", linewidth=0.4,
           label="significant")
ax.axhline(y=ci, color="gray", linestyle="--", linewidth=1,
           label=f"$\\pm 1.96/\\sqrt{{n}} = \\pm{ci:.4f}$")
ax.axhline(y=-ci, color="gray", linestyle="--", linewidth=1)
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xlabel(r"lag $k$")
ax.set_ylabel(r"$\hat\rho_k$")
ax.set_title(f"ACF of S&P 500 daily returns, 2000--2024 ($n = {n}$)")
ax.set_xticks(lags)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_acf_sp500.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print(f"Saved with n = {n}")
print(f"Max |rho|: {np.max(np.abs(rhos)):.4f}, CI: {ci:.4f}")
