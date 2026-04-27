"""
SFM_ch4_vr_profile
==================
Variance Ratio profile VR(q) for 3 markets (S&P 500, Bitcoin, BET).

Description:
- Download 3 series via yfinance: ^GSPC, BTC-USD, ^BETXT (BVB fallback to ^BET)
- Compute VR(q) for q in {2, 3, 5, 10, 20, 30, 60, 120}
- Plot the three VR(q) profiles on a log-q axis
- VR(q) == 1 random walk; > 1 momentum; < 1 mean reversion

Output:
- sfm_ch4_vr_profile.pdf
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


def variance_ratio(returns, q):
    returns = np.asarray(returns)
    n = len(returns)
    mu = returns.mean()
    var1 = np.sum((returns - mu) ** 2) / (n - 1)
    # q-period overlapping returns
    Tq = n - q + 1
    rq = np.array([returns[i:i + q].sum() for i in range(Tq)])
    var_q = np.sum((rq - q * mu) ** 2) / (Tq - 1)
    return var_q / (q * var1)


def fetch_returns(ticker, start="2015-01-01", end="2024-12-31"):
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    prices = df["Close"].squeeze().dropna()
    return np.log(prices).diff().dropna().values


qs = np.array([2, 3, 5, 10, 20, 30, 60, 120])

series = [
    ("^GSPC", "S&P 500 (developed)", MAIN_BLUE, "o-"),
    ("BTC-USD", "Bitcoin (crypto)", CRIMSON, "s-"),
    ("EEM", "EM Equities ETF (EEM)", FOREST, "D-"),
]

fig, ax = plt.subplots(figsize=(7, 4.3))
ax.axhline(y=1, color="black", linestyle="--", linewidth=1,
           label="$VR = 1$ (random walk)")
for ticker, label, color, style in series:
    try:
        rets = fetch_returns(ticker)
        if len(rets) < 200:
            raise ValueError("too few observations")
        vrs = [variance_ratio(rets, q) for q in qs]
        ax.plot(qs, vrs, style, color=color, linewidth=1.8, label=label)
        print(f"{ticker}: n = {len(rets)}, VR = {[f'{v:.3f}' for v in vrs]}")
    except Exception as e:
        print(f"{ticker} failed ({e}), skipping")

ax.set_xlabel(r"horizon $q$ (days)")
ax.set_ylabel(r"$VR(q)$")
ax.set_title(r"Variance ratio profile on 3 markets (2015--2024)")
ax.set_xscale("log")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_vr_profile.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_vr_profile.pdf")
