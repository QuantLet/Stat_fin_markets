import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, t as student_t, levy_stable, gennorm, genpareto
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

returns = {}
for name, ticker in tickers.items():
    try:
        data = yf.download(ticker, start='2015-01-01', end='2025-12-31', progress=False)
        close = data['Close'].squeeze().dropna()
        rets = np.log(close / close.shift(1)).dropna()
        rets = rets[np.isfinite(rets)]
        returns[name] = rets
        print(f"   {name:>10}: {len(rets)} obs")
    except Exception as e:
        print(f"   {name:>10}: FAILED ({e})")

fit_results = {}

for name, rets in returns.items():
    r = rets.values
    res = {}

    # Normal
    mu_n, sig_n = norm.fit(r)
    res['Normal'] = {'params': (mu_n, sig_n), 'k': 2,
                     'll': np.sum(norm.logpdf(r, mu_n, sig_n))}

    # Student-t
    df_t, loc_t, sc_t = student_t.fit(r)
    res['Student-t'] = {'params': (df_t, loc_t, sc_t), 'k': 3,
                        'll': np.sum(student_t.logpdf(r, df_t, loc_t, sc_t))}

    # GED
    beta_g, loc_g, sc_g = gennorm.fit(r)
    res['GED'] = {'params': (beta_g, loc_g, sc_g), 'k': 3,
                  'll': np.sum(gennorm.logpdf(r, beta_g, loc_g, sc_g))}

    # Stable (with timeout-safe initial guess)
    try:
        alpha_s, beta_s, loc_s, sc_s = levy_stable.fit(r, floc=np.mean(r))
        ll_stable = np.sum(levy_stable.logpdf(r, alpha_s, beta_s, loc=loc_s, scale=sc_s))
        res['Stable'] = {'params': (alpha_s, beta_s, loc_s, sc_s), 'k': 4, 'll': ll_stable}
    except Exception as e:
        print(f"   {name} Stable fit failed: {e}")
        res['Stable'] = None

    fit_results[name] = res
    print(f"   {name}: fits done")

print(f"{'Asset':>10} {'Student-t nu':>14} {'GED beta':>10} {'Stable alpha':>14} {'Stable beta':>13}")
print("-" * 65)
for name, res in fit_results.items():
    nu = res['Student-t']['params'][0]
    beta_g = res['GED']['params'][0]
    if res['Stable'] is not None:
        alpha_s = res['Stable']['params'][0]
        beta_s = res['Stable']['params'][1]
        print(f"{name:>10} {nu:>14.2f} {beta_g:>10.3f} {alpha_s:>14.4f} {beta_s:>13.4f}")
    else:
        print(f"{name:>10} {nu:>14.2f} {beta_g:>10.3f} {'N/A':>14} {'N/A':>13}")

# AIC table
print(f"\n{'Asset':>10} {'Normal AIC':>12} {'t AIC':>12} {'GED AIC':>12} {'Stable AIC':>12}")
print("-" * 62)
for name, res in fit_results.items():
    n = len(returns[name])
    row = f"{name:>10}"
    for dist in ['Normal', 'Student-t', 'GED', 'Stable']:
        if res[dist] is not None:
            aic = -2*res[dist]['ll'] + 2*res[dist]['k']
            row += f" {aic:>12.1f}"
        else:
            row += f" {'N/A':>12}"
    print(row)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for j, name in enumerate(returns.keys()):
    r = returns[name].values
    res = fit_results[name]
    ax = axes[j]

    ax.hist(r, bins=150, density=True, color='grey', alpha=0.35, label='Empirical', log=True)
    x = np.linspace(r.min(), r.max(), 1000)

    # Student-t
    p = res['Student-t']['params']
    ax.plot(x, student_t.pdf(x, *p), color=MAIN_BLUE, linewidth=1.0,
            label=f'Student-t ($\\nu$={p[0]:.1f})')

    # GED
    p = res['GED']['params']
    ax.plot(x, gennorm.pdf(x, *p), color=FOREST, linewidth=1.0,
            label=f'GED ($\\beta$={p[0]:.2f})')

    # Stable
    if res['Stable'] is not None:
        p = res['Stable']['params']
        ax.plot(x, levy_stable.pdf(x, p[0], p[1], loc=p[2], scale=p[3]),
                color=ORANGE, linewidth=1.0,
                label=f'Stable ($\\alpha$={p[0]:.2f})')

    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Log-return')
    if j == 0:
        ax.set_ylabel('Density (log scale)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False, fontsize=6)

plt.tight_layout()
save_fig('ch2_cs2b_advanced_fits')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for j, name in enumerate(returns.keys()):
    r = returns[name].values
    losses = -r  # Work with losses
    losses_sorted = np.sort(losses[losses > 0])[::-1]
    n = len(losses_sorted)
    survival = np.arange(1, n + 1) / n

    ax = axes[j]
    ax.loglog(losses_sorted, survival, '.', color=colors[name],
              markersize=1.5, alpha=0.5, label='Empirical')

    # Normal survival
    mu_n, sig_n = norm.fit(r)
    x_grid = np.logspace(np.log10(losses_sorted[-1]), np.log10(losses_sorted[0]), 200)
    ax.loglog(x_grid, norm.sf(x_grid, loc=-mu_n, scale=sig_n),
              color=AMBER, linewidth=1.0, label='Normal')

    # Student-t survival
    df_t, loc_t, sc_t = student_t.fit(r)
    ax.loglog(x_grid, student_t.sf(x_grid, df_t, loc=-loc_t, scale=sc_t),
              color=CRIMSON, linewidth=1.0, label=f'Student-t')

    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Loss threshold x')
    if j == 0:
        ax.set_ylabel(r'$P(\mathrm{Loss} > x)$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=7)

plt.tight_layout()
save_fig('ch2_cs2b_tail_loglog')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for j, name in enumerate(returns.keys()):
    r = returns[name].values
    losses = np.sort(np.abs(r))[::-1]
    n = len(losses)
    k_range = np.arange(20, min(n // 2, 800))
    hill_est = []
    for k in k_range:
        hill_est.append(1.0 / np.mean(np.log(losses[:k]) - np.log(losses[k])))
    hill_est = np.array(hill_est)

    ax = axes[j]
    ax.plot(k_range, hill_est, color=colors[name], linewidth=0.6)
    # Stable region: median of middle third
    mid_start = len(k_range) // 3
    mid_end = 2 * len(k_range) // 3
    alpha_hat = np.median(hill_est[mid_start:mid_end])
    ax.axhline(alpha_hat, color='grey', linestyle='--', linewidth=0.5)
    ax.text(0.97, 0.95, f'$\\hat{{\\alpha}} \\approx$ {alpha_hat:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Order statistic k')
    if j == 0:
        ax.set_ylabel('Hill estimator (tail index)')
    ax.set_ylim(0, 8)

plt.tight_layout()
save_fig('ch2_cs2b_hill_3assets')

alpha_levels = [0.01, 0.025, 0.05]
risk_results = {}

for name, rets in returns.items():
    r = rets.values
    mu_n, sig_n = norm.fit(r)
    df_t, loc_t, sc_t = student_t.fit(r)

    # EVT: GPD on losses above 95th percentile
    losses = -r
    threshold = np.percentile(losses, 95)
    exceedances = losses[losses > threshold] - threshold
    shape_gpd, _, scale_gpd = genpareto.fit(exceedances, floc=0)
    n_total = len(losses)
    n_exceed = len(exceedances)

    risk = {}
    for alpha in alpha_levels:
        # Normal VaR/ES
        var_n = -(mu_n + sig_n * norm.ppf(alpha))
        es_n = -(mu_n - sig_n * norm.pdf(norm.ppf(alpha)) / alpha)

        # Student-t VaR/ES
        var_t = -(loc_t + sc_t * student_t.ppf(alpha, df_t))
        # ES for Student-t
        t_ppf = student_t.ppf(alpha, df_t)
        es_t_factor = (student_t.pdf(t_ppf, df_t) / alpha) * ((df_t + t_ppf**2) / (df_t - 1))
        es_t = -(loc_t + sc_t * (-es_t_factor))

        # EVT VaR/ES (GPD)
        p_exceed = n_exceed / n_total
        if shape_gpd != 0:
            var_evt = threshold + (scale_gpd / shape_gpd) * ((alpha / p_exceed)**(-shape_gpd) - 1)
            es_evt = var_evt / (1 - shape_gpd) + (scale_gpd - shape_gpd * threshold) / (1 - shape_gpd)
        else:
            var_evt = threshold + scale_gpd * np.log(p_exceed / alpha)
            es_evt = var_evt + scale_gpd

        risk[alpha] = {
            'Normal': (var_n * 100, es_n * 100),
            'Student-t': (var_t * 100, es_t * 100),
            'EVT-GPD': (var_evt * 100, es_evt * 100)
        }
    risk_results[name] = risk

# Print table
print(f"{'Asset':>10} {'alpha':>6} {'Norm VaR%':>10} {'t VaR%':>10} {'EVT VaR%':>10} "
      f"{'Norm ES%':>10} {'t ES%':>10} {'EVT ES%':>10}")
print("-" * 78)
for name in returns.keys():
    for alpha in alpha_levels:
        rn = risk_results[name][alpha]
        print(f"{name:>10} {alpha:>6.3f} {rn['Normal'][0]:>10.3f} {rn['Student-t'][0]:>10.3f} "
              f"{rn['EVT-GPD'][0]:>10.3f} {rn['Normal'][1]:>10.3f} {rn['Student-t'][1]:>10.3f} "
              f"{rn['EVT-GPD'][1]:>10.3f}")

# Grouped bar chart: VaR at 1% for all assets
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
asset_names = list(returns.keys())
model_names = ['Normal', 'Student-t', 'EVT-GPD']
model_colors = [MAIN_BLUE, CRIMSON, FOREST]
alpha_plot = 0.01

x = np.arange(len(asset_names))
width = 0.25

# VaR
ax = axes[0]
for i, model in enumerate(model_names):
    vals = [risk_results[a][alpha_plot][model][0] for a in asset_names]
    ax.bar(x + i * width, vals, width, label=model, color=model_colors[i], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(asset_names)
ax.set_ylabel('VaR (1%) [%]')
ax.set_title('Value-at-Risk (1%)', fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, frameon=False, fontsize=7)

# ES
ax = axes[1]
for i, model in enumerate(model_names):
    vals = [risk_results[a][alpha_plot][model][1] for a in asset_names]
    ax.bar(x + i * width, vals, width, label=model, color=model_colors[i], alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(asset_names)
ax.set_ylabel('ES (1%) [%]')
ax.set_title('Expected Shortfall (1%)', fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, frameon=False, fontsize=7)

plt.tight_layout()
save_fig('ch2_cs2b_evt_risk_comparison')