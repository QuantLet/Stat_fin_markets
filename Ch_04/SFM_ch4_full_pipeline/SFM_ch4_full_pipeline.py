"""
SFM_ch4_full_pipeline
=====================
End-to-end EMH weak-form pipeline on S&P 500 (real data via yfinance).

Pipeline:
  1. Download ^GSPC 2015-01-01 to 2024-12-31, log-returns
  2. Sample autocorrelations rho_1..rho_20 with +/- 1.96/sqrt(n) band
  3. Ljung-Box Q(10), Q(20) with Box-Pierce comparison
  4. Wald-Wolfowitz runs test on signs (Z statistic)
  5. Variance Ratio profile VR(q) for q in {2,5,10,20,60,120}
     - asymptotic Lo-MacKinlay heteroskedasticity-robust 95% CI
  6. Chow-Denning MV1 multiple-VR statistic vs SMM critical value 2.491
  7. Stationary block bootstrap 95% CI for VR(q), B=500
  8. Rolling 1-year window: rho_1 and Ljung-Box p-value (AMH check)

Outputs (printed to stdout for inclusion in slides):
  - PIPELINE STATS block with all numerical results
  - Figure: sfm_ch4_full_pipeline.pdf (4-panel summary)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
from scipy import stats

try:
    CHARTS = Path(__file__).resolve().parents[3] / "charts"
except NameError:
    CHARTS = Path.cwd().resolve().parents[2] / "charts"
CHARTS.mkdir(exist_ok=True)

MAIN_BLUE = (26 / 255, 58 / 255, 110 / 255)
CRIMSON = (205 / 255, 0 / 255, 0 / 255)
FOREST = (46 / 255, 125 / 255, 50 / 255)
AMBER = (181 / 255, 133 / 255, 63 / 255)
GRAY = (0.55, 0.55, 0.55)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})

np.random.seed(20260427)

# -------- 1. data --------
df = yf.download("^GSPC", start="2015-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)
P = df["Close"].squeeze().dropna()
r = np.log(P).diff().dropna().values
n = len(r)
mu = r.mean()
sigma = r.std(ddof=1)

# -------- 2. ACF --------
def acf(x, K):
    x = np.asarray(x) - x.mean()
    var = np.sum(x ** 2)
    return np.array([np.sum(x[:len(x) - k] * x[k:]) / var for k in range(K + 1)])

K = 20
rho = acf(r, K)[1:]
band = 1.96 / np.sqrt(n)
lags_signif = [k + 1 for k, v in enumerate(rho) if abs(v) > band]
rho1 = rho[0]
rho_max_k = int(np.argmax(np.abs(rho))) + 1
rho_max_v = rho[rho_max_k - 1]

# -------- 3. Ljung-Box --------
def ljung_box(x, m):
    x = np.asarray(x) - x.mean()
    var = np.sum(x ** 2) / len(x)
    rhos = np.array([np.sum(x[:len(x) - k] * x[k:]) / (len(x) * var)
                     for k in range(1, m + 1)])
    n_ = len(x)
    Q = n_ * (n_ + 2) * np.sum(rhos ** 2 / (n_ - np.arange(1, m + 1)))
    p = 1 - stats.chi2.cdf(Q, df=m)
    return Q, p

Q10, p10 = ljung_box(r, 10)
Q20, p20 = ljung_box(r, 20)

# -------- 4. Runs test on signs --------
signs = np.sign(r)
signs = signs[signs != 0]
n_pos = int(np.sum(signs > 0))
n_neg = int(np.sum(signs < 0))
N = n_pos + n_neg
R = 1 + int(np.sum(signs[1:] != signs[:-1]))
ER = (2 * n_pos * n_neg) / N + 1
VR = (2 * n_pos * n_neg * (2 * n_pos * n_neg - N)) / (N ** 2 * (N - 1))
Z_runs = (R - ER) / np.sqrt(VR)

# -------- 5. VR(q) with Lo-MacKinlay heteroskedasticity-robust CI --------
def variance_ratio(returns, q):
    returns = np.asarray(returns)
    n_ = len(returns)
    mu_ = returns.mean()
    var1 = np.sum((returns - mu_) ** 2) / (n_ - 1)
    Tq = n_ - q + 1
    rq = np.array([returns[i:i + q].sum() for i in range(Tq)])
    var_q = np.sum((rq - q * mu_) ** 2) / (Tq - 1)
    return var_q / (q * var1)

def lo_mackinlay_phi_star(returns, q):
    """Heteroskedasticity-robust asymptotic variance of VR(q) - 1."""
    x = np.asarray(returns) - returns.mean()
    n_ = len(x)
    var1 = np.sum(x ** 2) / n_
    phi = 0.0
    for k in range(1, q):
        delta_num = np.sum((x[k:] ** 2) * (x[:n_ - k] ** 2))
        delta_den = (np.sum(x ** 2)) ** 2 / n_
        delta_k = delta_num / delta_den
        weight = (2 * (q - k) / q) ** 2
        phi += weight * delta_k
    return phi

qs = [2, 4, 8, 16]
vr_results = []
for q in qs:
    vr = variance_ratio(r, q)
    phi_star = lo_mackinlay_phi_star(r, q)
    se_vr = np.sqrt(phi_star / n)
    z_star = (vr - 1) / se_vr
    ci_lo, ci_hi = vr - 1.96 * se_vr, vr + 1.96 * se_vr
    reject = abs(z_star) > 1.96
    vr_results.append((q, vr, se_vr, z_star, ci_lo, ci_hi, reject))

# -------- 6. Chow-Denning MV1 --------
mv1 = max(abs(zs) for _, _, _, zs, _, _, _ in vr_results)
SMM_05 = 2.491  # Studentized Maximum Modulus, m=4, infinite df (Stoline-Ury 1979)
mv1_reject = mv1 > SMM_05

# -------- 7. Stationary block bootstrap --------
def stationary_bootstrap_indices(n_, mean_block_len, rng):
    p_resample = 1.0 / mean_block_len
    idx = np.empty(n_, dtype=int)
    idx[0] = rng.integers(0, n_)
    for t in range(1, n_):
        if rng.random() < p_resample:
            idx[t] = rng.integers(0, n_)
        else:
            idx[t] = (idx[t - 1] + 1) % n_
    return idx

B = 500
mean_block = int(round(np.sqrt(n)))
rng = np.random.default_rng(20260427)
boot_results = {}
for q in qs:
    samples = np.empty(B)
    for b in range(B):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        samples[b] = variance_ratio(r[idx], q)
    boot_results[q] = (np.percentile(samples, 2.5), np.percentile(samples, 97.5))

# -------- 8. Rolling AMH (1-year ~252 days) --------
W = 252
rho1_roll, p_roll, dates_roll = [], [], []
idx_dates = pd.to_datetime(df.index)[1:]
for t in range(W, n):
    win = r[t - W:t]
    rho1_roll.append(acf(win, 1)[1])
    _, p_ = ljung_box(win, 5)
    p_roll.append(p_)
    dates_roll.append(idx_dates[t - 1])
rho1_roll = np.array(rho1_roll)
p_roll = np.array(p_roll)
share_reject = np.mean(p_roll < 0.05)

# ============ PRINT REAL NUMBERS ============
print("=" * 60)
print("PIPELINE STATS  ^GSPC  2015-01-01 to 2024-12-31")
print("=" * 60)
print(f"n = {n}, mean = {mu:.6f}, sigma = {sigma:.6f}")
print(f"annualized sigma = {sigma * np.sqrt(252):.4f}")
print()
print("ACF:")
for k in [1, 2, 5, 10, 15, 20]:
    flag = "*" if abs(rho[k - 1]) > band else " "
    print(f"  rho_{k:2d} = {rho[k - 1]:+.4f} {flag}  (band = +/- {band:.4f})")
print(f"  significant lags (k=1..20): {lags_signif}")
print(f"  max |rho_k| at lag {rho_max_k}: rho = {rho_max_v:+.4f}")
print()
print(f"Ljung-Box Q(10) = {Q10:.2f}, p = {p10:.4f}")
print(f"Ljung-Box Q(20) = {Q20:.2f}, p = {p20:.4f}")
print()
print(f"Runs: R = {R}, n+ = {n_pos}, n- = {n_neg}")
print(f"  E[R] = {ER:.2f}, sd = {np.sqrt(VR):.3f}")
print(f"  Z = {Z_runs:+.3f}, p = {2 * (1 - stats.norm.cdf(abs(Z_runs))):.4f}")
print()
print("Variance ratio VR(q) with Lo-MacKinlay heteroskedasticity-robust 95% CI:")
print(f"  {'q':>3} {'VR(q)':>7} {'SE':>7} {'Z*':>7} {'CI low':>8} {'CI high':>8} reject")
for q, vr, se, zs, lo, hi, rej in vr_results:
    print(f"  {q:>3d} {vr:>7.4f} {se:>7.4f} {zs:>+7.3f} {lo:>8.4f} {hi:>8.4f} {'YES' if rej else ' no'}")
print()
print(f"Chow-Denning MV1 = {mv1:.3f}  vs SMM(0.05) = {SMM_05}")
print(f"  reject EMH-weak (joint test): {mv1_reject}")
print()
print("Stationary block bootstrap 95% CI (B=500, mean block=sqrt(n)):")
for q in qs:
    lo, hi = boot_results[q]
    contains_one = lo <= 1.0 <= hi
    print(f"  q={q:>3d}: [{lo:.4f}, {hi:.4f}]  contains 1: {contains_one}")
print()
print(f"Rolling 1-year window:")
print(f"  rho_1 range: [{rho1_roll.min():+.3f}, {rho1_roll.max():+.3f}]")
print(f"  rho_1 mean:  {rho1_roll.mean():+.4f}")
print(f"  share windows with LB Q(5) p < 0.05: {share_reject:.1%}")
print("=" * 60)

# ============ FIGURE: 4-panel summary ============
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))

# (a) ACF with band
ax = axes[0, 0]
colors_bar = [CRIMSON if abs(v) > band else MAIN_BLUE for v in rho]
ax.bar(np.arange(1, K + 1), rho, color=colors_bar, width=0.7, edgecolor="none")
ax.axhline(band, color=GRAY, linestyle="--", linewidth=0.9)
ax.axhline(-band, color=GRAY, linestyle="--", linewidth=0.9)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel(r"lag $k$")
ax.set_ylabel(r"$\hat\rho_k$")
ax.set_title(r"(a) ACF of $\hat r_t$ with $\pm 1.96/\sqrt{n}$ band")
ax.set_xticks(np.arange(1, K + 1, 2))
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# (b) VR profile with Lo-MacKinlay CI and bootstrap CI
ax = axes[0, 1]
qs_arr = np.array([q for q, *_ in vr_results])
vr_arr = np.array([v for _, v, *_ in vr_results])
lo_lm = np.array([lo for _, _, _, _, lo, _, _ in vr_results])
hi_lm = np.array([hi for _, _, _, _, _, hi, _ in vr_results])
lo_bs = np.array([boot_results[q][0] for q in qs])
hi_bs = np.array([boot_results[q][1] for q in qs])
ax.fill_between(qs_arr, lo_lm, hi_lm, color=MAIN_BLUE, alpha=0.18,
                label="Lo--MacKinlay 95\\% CI")
ax.errorbar(qs_arr, vr_arr, yerr=[vr_arr - lo_bs, hi_bs - vr_arr],
            fmt="none", ecolor=CRIMSON, capsize=3, linewidth=1.2,
            label="block-bootstrap 95\\% CI")
ax.plot(qs_arr, vr_arr, "o-", color=MAIN_BLUE, linewidth=1.6,
        markersize=5, label=r"$\widehat{VR}(q)$")
ax.axhline(1, color="black", linestyle="--", linewidth=0.8,
           label=r"$VR=1$ (RW)")
ax.set_xscale("log")
ax.set_xlabel(r"horizon $q$ (days)")
ax.set_ylabel(r"$VR(q)$")
ax.set_title("(b) Variance Ratio profile, asymptotic vs bootstrap CI")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", frameon=False, fontsize=7)

# (c) MV1 (Chow-Denning) bars
ax = axes[1, 0]
zs_arr = np.array([abs(zs) for _, _, _, zs, _, _, _ in vr_results])
colors_mv = [CRIMSON if abs(z_) > SMM_05 else MAIN_BLUE for z_ in zs_arr]
bars = ax.bar([str(q) for q in qs], zs_arr, color=colors_mv,
              width=0.65, edgecolor="none")
ax.axhline(SMM_05, color=CRIMSON, linestyle="--", linewidth=1.0,
           label=f"SMM$_{{0.05}}={SMM_05}$")
ax.axhline(1.96, color=GRAY, linestyle=":", linewidth=0.9,
           label=r"$1.96$ (necorectat)")
ax.scatter([str(qs[int(np.argmax(zs_arr))])], [zs_arr.max()],
           marker="*", s=120, color="black", zorder=5,
           label=f"MV1={zs_arr.max():.2f}")
ax.set_xlabel(r"horizon $q$")
ax.set_ylabel(r"$|Z^*(q)|$ (heteroskedasticity-robust)")
ax.set_title("(c) Chow--Denning MV1 vs Studentized Maximum Modulus")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=7)

# (d) Rolling rho_1 and LB p-value
ax = axes[1, 1]
ax2 = ax.twinx()
ax.plot(dates_roll, rho1_roll, color=MAIN_BLUE, linewidth=1.0,
        label=r"$\hat\rho_1$ (1-yr rolling)")
ax.axhline(0, color="black", linewidth=0.4)
ax2.plot(dates_roll, p_roll, color=CRIMSON, linewidth=0.9,
         alpha=0.85, label="LB Q(5) p-value")
ax2.axhline(0.05, color=CRIMSON, linestyle="--", linewidth=0.7)
ax2.fill_between(dates_roll, 0, p_roll, where=(p_roll < 0.05),
                 color=CRIMSON, alpha=0.18)
ax.set_xlabel("year")
ax.set_ylabel(r"$\hat\rho_1$", color=MAIN_BLUE)
ax2.set_ylabel("LB $p$-value", color=CRIMSON)
ax2.set_yscale("log")
ax.set_title(r"(d) AMH check --- rolling $\hat\rho_1$ \& LB $p$-value")
ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
ax.tick_params(axis="y", colors=MAIN_BLUE)
ax2.tick_params(axis="y", colors=CRIMSON)

fig.suptitle(r"S\&P 500 (\textasciicircum GSPC) 2015--2024 --- EMH weak-form full pipeline",
             y=1.005, fontsize=11)
fig.tight_layout()
fig.savefig(CHARTS / "sfm_ch4_full_pipeline.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_full_pipeline.pdf")
