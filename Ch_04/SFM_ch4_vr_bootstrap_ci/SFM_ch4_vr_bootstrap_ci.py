"""
SFM_ch4_vr_bootstrap_ci
=======================
Asymptotic CI vs block-bootstrap CI for VR(q) on Bitcoin.

Description:
- BTC-USD daily 2018-2025 via yfinance
- For each q in {2,5,10,20,30,60,120}:
    - Asymptotic Lo--MacKinlay 95% CI: VR_hat +/- 1.96 * sqrt(2(q-1)/T)
    - Block-bootstrap 95% CI: 1000 resamples with block length sqrt(T)
- Plot both CI bands; bootstrap is generally wider on heavy tails

Output:
- sfm_ch4_vr_bootstrap_ci.pdf
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


def block_bootstrap_vr(returns, q, B=1000, block=None, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(returns)
    if block is None:
        block = max(int(np.sqrt(n)), q + 1)
    n_blocks = int(np.ceil(n / block))
    vrs = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([returns[s:s + block] for s in starts])[:n]
        vrs[b] = variance_ratio(sample, q)
    return np.percentile(vrs, [2.5, 97.5])


df = yf.download("BTC-USD", start="2018-01-01", end="2025-12-31",
                 auto_adjust=True, progress=False)
prices = df["Close"].squeeze().dropna()
returns = np.log(prices).diff().dropna().values
T = len(returns)
print(f"BTC: T = {T}")

qs = np.array([2, 5, 10, 20, 30, 60, 120])
vr_hat = np.array([variance_ratio(returns, q) for q in qs])
se = np.sqrt(2 * (qs - 1) / T)
asy_lo, asy_hi = vr_hat - 1.96 * se, vr_hat + 1.96 * se

boot_lo = np.empty_like(vr_hat)
boot_hi = np.empty_like(vr_hat)
for i, q in enumerate(qs):
    boot_lo[i], boot_hi[i] = block_bootstrap_vr(returns, q, B=500)
    print(f"q = {q}: VR_hat = {vr_hat[i]:.3f}, "
          f"asy = [{asy_lo[i]:.3f}, {asy_hi[i]:.3f}], "
          f"boot = [{boot_lo[i]:.3f}, {boot_hi[i]:.3f}]")

fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.axhline(y=1, color="black", linestyle="--", linewidth=1,
           label=r"$VR = 1$ (random walk)")
# Asymptotic CI
ax.fill_between(qs, asy_lo, asy_hi, color=MAIN_BLUE, alpha=0.18,
                label=r"asymptotic 95% CI (Lo--MacKinlay)")
ax.plot(qs, vr_hat, "o-", color=MAIN_BLUE, linewidth=1.8, markersize=5,
        label=r"$\widehat{VR}(q)$ Bitcoin")
# Bootstrap CI as error bars
err = np.array([vr_hat - boot_lo, boot_hi - vr_hat])
ax.errorbar(qs, vr_hat, yerr=err, fmt="none", ecolor=CRIMSON,
            elinewidth=1.4, capsize=4,
            label=r"block-bootstrap 95% CI (B = 500)")

ax.set_xlabel(r"horizon $q$ (days)")
ax.set_ylabel(r"$VR(q)$")
ax.set_title(r"VR(q) on Bitcoin: asymptotic vs block-bootstrap CI (2018--2025)")
ax.set_xscale("log")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_vr_bootstrap_ci.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_vr_bootstrap_ci.pdf")
