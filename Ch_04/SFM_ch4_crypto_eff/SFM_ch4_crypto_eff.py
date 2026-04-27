"""
SFM_ch4_crypto_eff
==================
Eficiență crypto vs piață dezvoltată: VR(q) cu IC 95%.

Description:
- Date zilnice via yfinance: BTC-USD, ETH-USD, ^GSPC (2018-2025)
- Calculează VR(q) pe q in {2,3,5,10,20,30,60,90,120}
- Construiește IC 95% (Lo--MacKinlay) pentru fiecare activ
- Plotează profilul cu bandă umbrită; linie de referință VR=1
- Stil: fond transparent, legendă centrată sub axă

Output:
- sfm_ch4_crypto_eff.pdf
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
PURPLE = (142 / 255, 68 / 255, 173 / 255)

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
    ("^GSPC", "S&P 500 (developed)", MAIN_BLUE, "o-"),
    ("BTC-USD", "Bitcoin (BTC)", CRIMSON, "s-"),
    ("ETH-USD", "Ethereum (ETH)", PURPLE, "D-"),
]

qs = np.array([2, 3, 5, 10, 20, 30, 60, 90, 120])

fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.axhline(y=1, color="black", linestyle="--", linewidth=1,
           label=r"$VR = 1$ (random walk)")

for ticker, label, color, style in tickers:
    try:
        df = yf.download(ticker, start="2018-01-01", end="2025-12-31",
                         auto_adjust=True, progress=False)
        prices = df["Close"].squeeze().dropna()
        ret = np.log(prices).diff().dropna().values
        T = len(ret)
        if T < 200:
            print(f"{ticker}: prea puține observații, sar")
            continue
        vr = np.array([variance_ratio(ret, q) for q in qs])
        se = np.sqrt(2 * (qs - 1) / T)
        ax.fill_between(qs, vr - 1.96 * se, vr + 1.96 * se,
                        color=color, alpha=0.15)
        ax.plot(qs, vr, style, color=color, linewidth=1.8, markersize=5,
                label=label)
        print(f"{ticker}: T = {T}, VR(120) = {vr[-1]:.3f}")
    except Exception as e:
        print(f"{ticker} eșec ({e}), sar")

ax.set_xlabel(r"horizon $q$ (days)")
ax.set_ylabel(r"$VR(q)$")
ax.set_title(r"Crypto vs developed --- VR profile with 95% CI (2018--2025)")
ax.set_xscale("log")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_crypto_eff.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_crypto_eff.pdf")
