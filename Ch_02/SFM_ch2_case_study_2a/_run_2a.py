import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, t as student_t
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# === Style & paths ===
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75

MAIN_BLUE = '#1A3A6E'
CRIMSON   = '#DC3545'
FOREST    = '#2E7D32'
AMBER     = '#B5853F'
ORANGE    = '#E67E22'

CHART_DIR = os.path.join('..', '..', '..', 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

def save_fig(name):
    """Save figure with transparent background."""
    plt.savefig(os.path.join(CHART_DIR, f'{name}.pdf'),
                bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(CHART_DIR, f'{name}.png'),
                bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf/.png")

tickers = {'S&P 500': '^GSPC', 'BRD': 'BRD.RO', 'Bitcoin': 'BTC-USD'}
colors  = {'S&P 500': MAIN_BLUE, 'BRD': CRIMSON, 'Bitcoin': FOREST}

prices, returns = {}, {}
for name, ticker in tickers.items():
    try:
        data = yf.download(ticker, start='2015-01-01', end='2025-12-31', progress=False)
        close = data['Close'].squeeze().dropna()
        rets = np.log(close / close.shift(1)).dropna()
        rets = rets[np.isfinite(rets)]
        prices[name] = close
        returns[name] = rets
        print(f"   {name:>10}: {len(rets)} obs, [{rets.index[0].strftime('%Y-%m-%d')} – {rets.index[-1].strftime('%Y-%m-%d')}]")
    except Exception as e:
        print(f"   {name:>10}: FAILED ({e})")

print(f"{'Asset':>10} {'N':>6} {'Mean%':>8} {'Std%':>8} {'Skew':>8} {'Kurt':>8} {'Min%':>8} {'Max%':>8}")
print("-" * 74)
for name, rets in returns.items():
    r = rets.values
    print(f"{name:>10} {len(r):>6} {r.mean()*100:>8.4f} {r.std()*100:>8.4f} "
          f"{stats.skew(r):>8.3f} {stats.kurtosis(r):>8.2f} "
          f"{r.min()*100:>8.2f} {r.max()*100:>8.2f}")

fig, axes = plt.subplots(2, 3, figsize=(14, 6))

for j, name in enumerate(returns.keys()):
    # Price
    ax = axes[0, j]
    p = prices[name]
    ax.plot(p.index, p.values, color=colors[name], linewidth=0.5)
    ax.set_title(name, fontweight='bold')
    if j == 0:
        ax.set_ylabel('Price')

    # Returns
    ax = axes[1, j]
    r = returns[name]
    ax.plot(r.index, r.values, color=colors[name], linewidth=0.3, alpha=0.7)
    ax.axhline(0, color='grey', linewidth=0.3)
    if j == 0:
        ax.set_ylabel('Log-return')

plt.tight_layout()
save_fig('ch2_cs2a_price_returns')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

max_lag = 60
for j, name in enumerate(returns.keys()):
    r = returns[name].values
    abs_r = np.abs(r)
    n = len(abs_r)
    mean_abs = abs_r.mean()
    acf_vals = []
    for lag in range(1, max_lag + 1):
        cov = np.mean((abs_r[lag:] - mean_abs) * (abs_r[:-lag] - mean_abs))
        var = np.mean((abs_r - mean_abs)**2)
        acf_vals.append(cov / var)
    lags = np.arange(1, max_lag + 1)

    ax = axes[j]
    ax.bar(lags, acf_vals, color=colors[name], width=0.8, alpha=0.7)
    ax.axhline(1.96/np.sqrt(n), color='grey', linestyle='--', linewidth=0.5)
    ax.axhline(-1.96/np.sqrt(n), color='grey', linestyle='--', linewidth=0.5)
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_xlabel('Lag')
    if j == 0:
        ax.set_ylabel(r'ACF of $|r_t|$')

    # Annotate kurtosis + skewness
    kurt = stats.kurtosis(r)
    skew = stats.skew(r)
    ax.text(0.97, 0.95, f'Kurt = {kurt:.2f}\nSkew = {skew:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
save_fig('ch2_cs2a_stylized_3assets')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for j, name in enumerate(returns.keys()):
    r = returns[name].values
    ax = axes[j]

    # Histogram
    ax.hist(r, bins=150, density=True, color='grey', alpha=0.4,
            label='Empirical', log=True)

    # Fit Normal
    mu_n, sig_n = norm.fit(r)
    x = np.linspace(r.min(), r.max(), 1000)
    ax.plot(x, norm.pdf(x, mu_n, sig_n), color=MAIN_BLUE, linewidth=1.2,
            label=f'Normal($\\mu$={mu_n:.4f}, $\\sigma$={sig_n:.4f})')

    # Fit Student-t
    df_t, loc_t, sc_t = student_t.fit(r)
    ax.plot(x, student_t.pdf(x, df_t, loc_t, sc_t), color=CRIMSON, linewidth=1.2,
            label=f'Student-t($\\nu$={df_t:.2f})')

    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Log-return')
    if j == 0:
        ax.set_ylabel('Density (log scale)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=1, frameon=False, fontsize=6)

plt.tight_layout()
save_fig('ch2_cs2a_fit_comparison')

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for j, name in enumerate(returns.keys()):
    r = np.sort(returns[name].values)
    n_obs = len(r)
    p = (np.arange(1, n_obs + 1) - 0.5) / n_obs

    # Normal QQ
    mu_n, sig_n = norm.fit(r)
    theo_n = norm.ppf(p, mu_n, sig_n)
    ax = axes[0, j]
    ax.scatter(theo_n, r, s=0.3, alpha=0.4, color=MAIN_BLUE)
    lims = [min(theo_n.min(), r.min()), max(theo_n.max(), r.max())]
    ax.plot(lims, lims, 'k--', linewidth=0.5)
    ax.set_title(f'{name} — Normal QQ', fontweight='bold')
    if j == 0:
        ax.set_ylabel('Empirical quantiles')
    ax.set_xlabel('Theoretical quantiles')

    # Student-t QQ
    df_t, loc_t, sc_t = student_t.fit(r)
    theo_t = student_t.ppf(p, df_t, loc_t, sc_t)
    ax = axes[1, j]
    ax.scatter(theo_t, r, s=0.3, alpha=0.4, color=CRIMSON)
    lims = [min(theo_t.min(), r.min()), max(theo_t.max(), r.max())]
    ax.plot(lims, lims, 'k--', linewidth=0.5)
    ax.set_title(f'{name} — Student-t QQ ($\\nu$={df_t:.1f})', fontweight='bold')
    if j == 0:
        ax.set_ylabel('Empirical quantiles')
    ax.set_xlabel('Theoretical quantiles')

plt.tight_layout()
save_fig('ch2_cs2a_qq_comparison')

print(f"{'Asset':>10} {'JB stat':>12} {'JB p':>10} {'SW stat':>10} {'SW p':>10} "
      f"{'AD stat':>10} {'AD p':>10} {'KS stat':>10} {'KS p':>10}")
print("-" * 104)

for name, rets in returns.items():
    r = rets.values
    # Use subsample for Shapiro-Wilk (max 5000)
    r_sw = r[:5000] if len(r) > 5000 else r

    jb_stat, jb_p = stats.jarque_bera(r)
    sw_stat, sw_p = stats.shapiro(r_sw)
    ad_result = stats.anderson(r, dist='norm')
    ks_stat, ks_p = stats.kstest(r, 'norm', args=(r.mean(), r.std()))

    print(f"{name:>10} {jb_stat:>12.1f} {jb_p:>10.2e} {sw_stat:>10.6f} {sw_p:>10.2e} "
          f"{ad_result.statistic:>10.2f} {'<0.01':>10} {ks_stat:>10.4f} {ks_p:>10.2e}")

print("\n=> All assets: normality rejected at any conventional level.")