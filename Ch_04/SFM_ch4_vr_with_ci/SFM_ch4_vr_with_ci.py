"""
SFM_ch4_vr_with_ci
==================
Variance Ratio VR(q) cu intervale de încredere 95% (Lo--MacKinlay)
pentru S&P 500.

Description:
- Descarcă ^GSPC din yfinance (2000-2024)
- Calculează VR(q) pentru q in {2,3,5,10,20,30,60,90,120}
- Construiește IC 95% folosind SE = sqrt(2(q-1)/T) (homoscedastic)
- Marchează cu roșu orizonturile la care IC nu conține 1
- Stil: fond transparent, legendă centrată sub axă

Output:
- sfm_ch4_vr_with_ci.pdf
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


def variance_ratio(returns, q):
    returns = np.asarray(returns)
    n = len(returns)
    mu = returns.mean()
    var1 = np.sum((returns - mu) ** 2) / (n - 1)
    Tq = n - q + 1
    rq = np.array([returns[i:i + q].sum() for i in range(Tq)])
    var_q = np.sum((rq - q * mu) ** 2) / (Tq - 1)
    return var_q / (q * var1)


df = yf.download("^GSPC", start="2000-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)
prices = df["Close"].squeeze().dropna()
returns = np.log(prices).diff().dropna().values
T = len(returns)
print(f"S&P 500: T = {T} observations")

qs = np.array([2, 3, 5, 10, 20, 30, 60, 90, 120])
vr_hat = np.array([variance_ratio(returns, q) for q in qs])
se = np.sqrt(2 * (qs - 1) / T)
ci_lo, ci_hi = vr_hat - 1.96 * se, vr_hat + 1.96 * se
print("VR(q):", [f"{v:.3f}" for v in vr_hat])

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.axhline(y=1, color="black", linestyle="--", linewidth=1,
           label=r"$VR = 1$ (random walk)")
ax.fill_between(qs, ci_lo, ci_hi, color=MAIN_BLUE, alpha=0.18,
                label=r"95% confidence interval")
ax.plot(qs, vr_hat, "o-", color=MAIN_BLUE, linewidth=1.8, markersize=5,
        label=r"$\widehat{VR}(q)$ S&P 500")
sig = (ci_hi < 1) | (ci_lo > 1)
if sig.any():
    ax.plot(qs[sig], vr_hat[sig], "o", color=CRIMSON, markersize=8,
            markerfacecolor="none", markeredgewidth=1.6,
            label="reject RW at 5%")
ax.set_xlabel(r"horizon $q$ (days)")
ax.set_ylabel(r"$VR(q)$")
ax.set_title(rf"Variance ratio with 95% CI --- S&P 500 (T = {T})")
ax.set_xscale("log")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_vr_with_ci.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_vr_with_ci.pdf")
