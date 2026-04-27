"""
SFM_ch4_subperiod_efficiency
============================
Sub-period efficiency analysis on S&P 500.

Description:
- Daily S&P 500 (^GSPC) 1995-2024 via yfinance
- 4 sub-periods: 1995-2002, 2003-2010, 2011-2018, 2019-2024
- For each: lag-1 autocorrelation, Ljung-Box Q(10) p-value, VR(5)
- Three-panel bar chart with the three statistics across periods

Output:
- sfm_ch4_subperiod_efficiency.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.stats.diagnostic import acorr_ljungbox

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
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})


def variance_ratio(returns, q):
    returns = np.asarray(returns)
    n = len(returns)
    mu = returns.mean()
    var1 = np.sum((returns - mu) ** 2) / (n - 1)
    Tq = n - q + 1
    rq = np.array([returns[i:i + q].sum() for i in range(Tq)])
    var_q = np.sum((rq - q * mu) ** 2) / (Tq - 1)
    return var_q / (q * var1)


periods = [
    ("1995--2002\n(dotcom)", "1995-01-01", "2002-12-31", MAIN_BLUE),
    ("2003--2010\n(GFC)", "2003-01-01", "2010-12-31", CRIMSON),
    ("2011--2018\n(QE bull)", "2011-01-01", "2018-12-31", FOREST),
    ("2019--2024\n(COVID)", "2019-01-01", "2024-12-31", AMBER),
]

df = yf.download("^GSPC", start="1995-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)
prices = df["Close"].squeeze().dropna()
returns = np.log(prices).diff().dropna()

labels, rho1_arr, p_arr, vr_arr, colors = [], [], [], [], []
for lab, s, e, c in periods:
    sub = returns.loc[s:e].values
    if len(sub) < 200:
        continue
    rho1 = np.corrcoef(sub[:-1], sub[1:])[0, 1]
    lb = acorr_ljungbox(sub, lags=[10], return_df=True)
    p = float(lb["lb_pvalue"].iloc[0])
    vr = variance_ratio(sub, 5)
    labels.append(lab)
    rho1_arr.append(rho1)
    p_arr.append(p)
    vr_arr.append(vr)
    colors.append(c)
    print(f"{lab.replace(chr(10), ' ')}: n = {len(sub)}, rho_1 = {rho1:.4f}, "
          f"LB p = {p:.4f}, VR(5) = {vr:.4f}")

x = np.arange(len(labels))
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))

ax1 = axes[0]
ax1.bar(x, rho1_arr, color=colors, edgecolor="black", linewidth=0.4)
ax1.axhline(0, color="black", linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8.5)
ax1.set_ylabel(r"$\hat\rho_1$")
ax1.set_title(r"(a) Lag-1 autocorrelation")
ax1.grid(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2 = axes[1]
ax2.bar(x, p_arr, color=colors, edgecolor="black", linewidth=0.4)
ax2.axhline(0.05, color=CRIMSON, linestyle="--", linewidth=1,
            label=r"$\alpha = 0.05$")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8.5)
ax2.set_ylabel("p-value")
ax2.set_title(r"(b) Ljung--Box $Q(10)$ p-value")
ax2.set_yscale("log")
ax2.set_ylim(1e-5, 1.0)
ax2.legend(loc="upper right", fontsize=8, frameon=False)
ax2.grid(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax3 = axes[2]
ax3.bar(x, vr_arr, color=colors, edgecolor="black", linewidth=0.4)
ax3.axhline(1.0, color="black", linestyle="--", linewidth=1,
            label=r"$VR = 1$ (RW)")
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=8.5)
ax3.set_ylabel(r"$VR(5)$")
ax3.set_title(r"(c) Variance ratio at $q = 5$")
ax3.legend(loc="upper right", fontsize=8, frameon=False)
ax3.grid(False)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

fig.suptitle(r"S&P 500 weak-form efficiency by sub-period (1995--2024)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(CHARTS / "sfm_ch4_subperiod_efficiency.pdf",
            bbox_inches="tight", pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_subperiod_efficiency.pdf")
