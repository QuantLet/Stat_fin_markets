"""
SFM_ch4_mv_test
===============
Chow & Denning (1993) Multiple Variance Ratio test.

Description:
- For each of 5 markets (S&P 500, EUR/USD, Gold, EEM, Bitcoin):
    - Download daily prices via yfinance, compute log-returns
    - For q in {2, 4, 8, 16} compute heteroscedasticity-robust
      Z*(q) statistic (Lo & MacKinlay 1988, eq. (15))
    - Compute MV1 = max_q |Z*(q)|
- Compare to SMM critical value at 5% (m=4 horizons, infty df): SMM_0.05 = 2.491
- Bar plot per asset of |Z*(q)| with the SMM threshold overlay
- Reject H0 (joint random walk on all q) when MV1 > SMM threshold

References:
- Chow, K.V., Denning, K.C. (1993). A simple multiple variance ratio test.
  Journal of Econometrics 58, 385-401.
- Lo, A.W., MacKinlay, A.C. (1988). Stock market prices do not follow
  random walks. Review of Financial Studies 1, 41-66.

Output:
- sfm_ch4_mv_test.pdf
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
PURPLE = (142 / 255, 68 / 255, 173 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})


def vr_z_robust(returns, q):
    """Heteroscedasticity-robust VR test statistic Z*(q) of Lo & MacKinlay."""
    r = np.asarray(returns, dtype=float)
    nq = len(r)
    mu = r.mean()
    sigma2_a = np.sum((r - mu) ** 2) / nq  # under-biased (Lo-MacKinlay)
    # q-period overlapping returns
    Tq = nq - q + 1
    rq = np.array([r[i:i + q].sum() for i in range(Tq)])
    sigma2_c = np.sum((rq - q * mu) ** 2) / (q * nq * (1 - q / nq))
    vr = sigma2_c / sigma2_a
    # Heteroscedasticity-consistent variance (Lo-MacKinlay 1988, Eq. 15)
    delta_sum = 0.0
    eta = (r - mu) ** 2
    for j in range(1, q):
        num = np.sum(eta[j:] * eta[:-j])
        den = np.sum(eta) ** 2 / nq  # eta_bar^2 * nq
        delta_j = num / den
        delta_sum += (2 * (q - j) / q) ** 2 * delta_j
    z_star = (vr - 1) * np.sqrt(nq) / np.sqrt(delta_sum) if delta_sum > 0 else np.nan
    return vr, z_star


# Studentized Maximum Modulus critical value at 5% with m=4 horizons,
# infty df (Stoline & Ury 1979 Table 2; Chow & Denning 1993)
SMM_4 = 2.491
QS = np.array([2, 4, 8, 16])

tickers = [
    ("^GSPC", "S&P 500", MAIN_BLUE),
    ("EURUSD=X", "EUR/USD", FOREST),
    ("GC=F", "Gold", AMBER),
    ("EEM", "EEM (EM ETF)", PURPLE),
    ("BTC-USD", "Bitcoin", CRIMSON),
]

results = []
for ticker, label, color in tickers:
    try:
        df = yf.download(ticker, start="2018-01-01", end="2025-12-31",
                         auto_adjust=True, progress=False)
        prices = df["Close"].squeeze().dropna()
        ret = np.log(prices).diff().dropna().values
        zs = []
        vrs = []
        for q in QS:
            vr, z = vr_z_robust(ret, q)
            vrs.append(vr)
            zs.append(z)
        zs = np.array(zs)
        mv1 = np.max(np.abs(zs))
        reject = mv1 > SMM_4
        results.append({
            "label": label, "color": color, "n": len(ret),
            "vrs": vrs, "zs": zs, "mv1": mv1, "reject": reject,
        })
        print(f"{label}: n = {len(ret)}, "
              f"|Z*| = {[f'{abs(z):.2f}' for z in zs]}, "
              f"MV1 = {mv1:.3f}, reject H0: {reject}")
    except Exception as e:
        print(f"{ticker} failed: {e}")

fig, ax = plt.subplots(figsize=(8.0, 4.4))
n_assets = len(results)
n_q = len(QS)
total_width = 0.8
width = total_width / n_q
xs = np.arange(n_assets)

for i, q in enumerate(QS):
    offset = (i - (n_q - 1) / 2) * width
    bars = ax.bar(xs + offset,
                   [abs(r["zs"][i]) for r in results],
                   width=width * 0.95,
                   color=[r["color"] for r in results],
                   alpha=0.55 + 0.10 * i,
                   edgecolor="black", linewidth=0.3,
                   label=fr"$|Z^*(q={q})|$")

ax.axhline(y=SMM_4, color=CRIMSON, linestyle="--", linewidth=1.4,
           label=fr"SMM $_{{0.05, m=4}}$ = {SMM_4:.3f}")
ax.axhline(y=1.96, color="gray", linestyle=":", linewidth=1,
           label=r"individual $|z| = 1.96$ (uncorrected, ignores multiplicity)")

# MV1 markers
for i, r in enumerate(results):
    ax.plot(xs[i], r["mv1"], marker="*",
            markersize=14, color="black", zorder=5)
ax.plot([], [], "k*", markersize=10, label=r"MV1 $= \max_q |Z^*(q)|$")

ax.set_xticks(xs)
ax.set_xticklabels([r["label"] for r in results], fontsize=9)
ax.set_ylabel(r"$|Z^*(q)|$ (heteroscedasticity-robust)")
ax.set_title(r"Chow--Denning Multiple Variance Ratio test on 5 markets (2018--2025)")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                ncol=3, frameon=False, fontsize=8.5)
leg.get_frame().set_alpha(0)
leg.get_frame().set_facecolor("none")
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
fig.savefig(CHARTS / "sfm_ch4_mv_test.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True, facecolor="none",
            edgecolor="none")
print("Saved sfm_ch4_mv_test.pdf")
