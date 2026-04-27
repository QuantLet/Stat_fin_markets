"""
SFM_ch4_cross_asset
===================
Comparație cross-asset a eficienței slabe a pieței.

Description:
- 6 active via yfinance: S&P 500, EUR/USD, Aur, EM Equities ETF (EEM),
  BET (^BETI), Bitcoin
- Perioada: 2020-2025 zilnic
- Metrici: p-value Ljung-Box Q(10), |VR(5) - 1|
- Două panouri orizontale (a) p-values, (b) magnitudine deviație VR
- Stil: fond transparent, fără legendă (etichete pe axa Y)

Output:
- sfm_ch4_cross_asset.pdf
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
AMBER = (181 / 255, 133 / 255, 63 / 255)
FOREST = (46 / 255, 125 / 255, 50 / 255)
PURPLE = (142 / 255, 68 / 255, 173 / 255)
TEAL = (0 / 255, 128 / 255, 128 / 255)

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


tickers = [
    ("^GSPC", "S&P 500", MAIN_BLUE),
    ("EURUSD=X", "EUR/USD", FOREST),
    ("GC=F", "Gold", AMBER),
    ("EEM", "EM Equities ETF", PURPLE),
    ("^BETI", "BET (BVB)", TEAL),
    ("BTC-USD", "Bitcoin", CRIMSON),
]

assets, lb_p, vr_dev, colors = [], [], [], []
for ticker, label, color in tickers:
    try:
        df = yf.download(ticker, start="2020-01-01", end="2025-12-31",
                         auto_adjust=True, progress=False)
        prices = df["Close"].squeeze().dropna()
        if len(prices) < 100:
            print(f"{ticker}: prea puține observații, sar")
            continue
        ret = np.log(prices).diff().dropna().values
        lb = acorr_ljungbox(ret, lags=[10], return_df=True)
        p = float(lb["lb_pvalue"].iloc[0])
        vr = variance_ratio(ret, 5)
        assets.append(label)
        lb_p.append(p)
        vr_dev.append(abs(vr - 1))
        colors.append(color)
        print(f"{label}: n = {len(ret)}, LB-p = {p:.4f}, |VR(5)-1| = {abs(vr-1):.3f}")
    except Exception as e:
        print(f"{ticker} eșec ({e}), sar")

lb_p = np.array(lb_p)
vr_dev = np.array(vr_dev)

fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))

ax1 = axes[0]
ax1.barh(assets, lb_p, color=colors, edgecolor="black", linewidth=0.4)
ax1.axvline(x=0.05, color=CRIMSON, linestyle="--", linewidth=1)
ax1.text(0.052, -0.4, r"$\alpha = 0.05$", color=CRIMSON, fontsize=8.5)
ax1.set_xlabel(r"Ljung--Box $Q(10)$ p-value")
ax1.set_title(r"(a) Ljung--Box test: $p > 0.05 \Rightarrow$ weak-form EMH ok")
ax1.invert_yaxis()
ax1.grid(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2 = axes[1]
ax2.barh(assets, vr_dev, color=colors, edgecolor="black", linewidth=0.4)
ax2.set_xlabel(r"$|VR(5) - 1|$ (deviation from random walk)")
ax2.set_title(r"(b) Magnitude of VR(5) deviation")
ax2.invert_yaxis()
ax2.grid(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle(r"Cross-asset efficiency --- 2020--2025",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(CHARTS / "sfm_ch4_cross_asset.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_cross_asset.pdf")
