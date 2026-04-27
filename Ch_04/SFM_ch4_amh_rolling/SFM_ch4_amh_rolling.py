"""
SFM_ch4_amh_rolling
===================
Rolling autocorelație lag 1 și rolling p-value Ljung-Box Q(10) pentru
S&P 500 --- evidență AMH (Lo, 2004).

Description:
- Date: ^GSPC zilnic 2000-2024 via yfinance
- Fereastră rolling de 252 de zile (~1 an de tranzacționare)
- Calculează rho_1 și p-value Q(10) pe fiecare fereastră
- Plotează ambele pe sub-axe stivuite, evidențiind perioadele de criză
- Stil: fond transparent, legendă centrată jos sub întreaga figură

Output:
- sfm_ch4_amh_rolling.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})


df = yf.download("^GSPC", start="2000-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)
prices = df["Close"].squeeze().dropna()
returns = np.log(prices).diff().dropna()
print(f"S&P 500: T = {len(returns)}")

window = 252  # 1 an
rho1 = []
pvals = []
dates = []
ret_arr = returns.values
for i in range(window, len(ret_arr)):
    chunk = ret_arr[i - window:i]
    rho1.append(np.corrcoef(chunk[:-1], chunk[1:])[0, 1])
    lb = acorr_ljungbox(chunk, lags=[10], return_df=True)
    pvals.append(float(lb["lb_pvalue"].iloc[0]))
    dates.append(returns.index[i])
rho1 = np.array(rho1)
pvals = np.array(pvals)
years = np.array([d.year + (d.dayofyear - 1) / 365.25 for d in dates])

# Crisis markers
crisis_times = [2001.0, 2008.7, 2020.2, 2022.3]
crisis_labels = ["dotcom", "GFC", "COVID", "inflation"]

fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True)

ax1 = axes[0]
ax1.axhline(0, color="black", linestyle="--", linewidth=1)
ax1.plot(years, rho1, color=MAIN_BLUE, linewidth=1.2,
         label=r"$\hat\rho_1$ rolling (1 year)")
ax1.fill_between(years, 0, rho1, where=(rho1 < -0.05),
                 color=CRIMSON, alpha=0.20,
                 label="significant negative autocorrelation")
for ct, lab in zip(crisis_times, crisis_labels):
    ax1.axvline(ct, color="gray", linestyle=":", linewidth=0.6, alpha=0.7)
    ax1.annotate(lab, (ct, ax1.get_ylim()[1] * 0.85), fontsize=7.5,
                 ha="center", color="gray")
ax1.set_ylabel(r"$\hat\rho_1$")
ax1.set_title(r"Time-varying efficiency --- AMH evidence (Lo, 2004) on S&P 500")
ax1.grid(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2 = axes[1]
ax2.axhline(0.05, color=CRIMSON, linestyle="--", linewidth=1,
            label=r"$\alpha = 0.05$")
ax2.plot(years, pvals, color=FOREST, linewidth=1.2,
         label=r"rolling Ljung--Box $Q(10)$ p-value")
ax2.fill_between(years, 0, 0.05, color=CRIMSON, alpha=0.10,
                 label="weak-form EMH rejection zone")
for ct in crisis_times:
    ax2.axvline(ct, color="gray", linestyle=":", linewidth=0.6, alpha=0.7)
ax2.set_xlabel("Year")
ax2.set_ylabel(r"p-value $Q(10)$")
ax2.set_yscale("log")
ax2.set_ylim(0.001, 1.0)
ax2.grid(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Legendă unificată sub întreaga figură
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center",
           bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False,
           fontsize=8.5)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(CHARTS / "sfm_ch4_amh_rolling.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_amh_rolling.pdf")
