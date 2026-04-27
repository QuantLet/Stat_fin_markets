"""
SFM_ch4_bvb_case
================
Studiu de caz BVB: ACF pentru BET, TLV, FP (2020-2025).

Description:
- Date zilnice via yfinance: ^BETI (BET index), TLV.RO, FP.RO
- Calculează ACF pe lag-urile 1..15
- Plotează cele 3 ACF-uri ca bare grupate, cu banda +/- 1.96/sqrt(n)
- Evidențiază autocorelația semnificativă la lag 1 cauzată de thin trading
- Stil: fond transparent, legendă centrată sub axă

Output:
- sfm_ch4_bvb_case.pdf
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
AMBER = (181 / 255, 133 / 255, 63 / 255)
FOREST = (46 / 255, 125 / 255, 50 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})


def autocorr(x, max_lag):
    x = np.asarray(x)
    x = x - x.mean()
    g0 = (x * x).mean()
    return np.array([(x[k:] * x[:-k]).mean() / g0
                     for k in range(1, max_lag + 1)])


tickers = [
    ("^BETI", "BET (BVB index)", MAIN_BLUE),
    ("TLV.RO", "TLV (Banca Transilvania)", FOREST),
    ("FP.RO", "FP (Fondul Proprietatea)", AMBER),
]

acfs = []
labels = []
colors = []
n_max = 0
for ticker, label, color in tickers:
    try:
        df = yf.download(ticker, start="2020-01-01", end="2025-12-31",
                         auto_adjust=True, progress=False)
        prices = df["Close"].squeeze().dropna()
        if len(prices) < 100:
            print(f"{ticker}: prea puține observații, sar")
            continue
        ret = np.log(prices).diff().dropna().values
        rho = autocorr(ret, 15)
        acfs.append(rho)
        labels.append(label)
        colors.append(color)
        n_max = max(n_max, len(ret))
        print(f"{label}: n = {len(ret)}, rho_1 = {rho[0]:.3f}, rho_2 = {rho[1]:.3f}")
    except Exception as e:
        print(f"{ticker} eșec ({e}), sar")

n_use = n_max if n_max > 0 else 1250
ci = 1.96 / np.sqrt(n_use)
lags = np.arange(1, 16)
fig, ax = plt.subplots(figsize=(7.4, 4.3))
n_series = len(acfs)
total_width = 0.8
width = total_width / max(n_series, 1)
for i, (rho, label, color) in enumerate(zip(acfs, labels, colors)):
    offset = (i - (n_series - 1) / 2) * width
    ax.bar(lags + offset, rho, width=width * 0.95, color=color,
           edgecolor="black", linewidth=0.3, label=label)
ax.axhline(y=ci, color=CRIMSON, linestyle="--", linewidth=1,
           label=fr"$\pm 1{{,}}96/\sqrt{{n}} = \pm{ci:.3f}$")
ax.axhline(y=-ci, color=CRIMSON, linestyle="--", linewidth=1)
ax.axhline(y=0, color="gray", linewidth=0.4)
ax.set_xlabel("lag $k$")
ax.set_ylabel(r"$\hat\rho_k$")
ax.set_title(r"ACF on BVB --- thin trading and lag-1 autocorrelation")
ax.set_xticks(lags)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_bvb_case.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_bvb_case.pdf")
