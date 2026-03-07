import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.stats import norm, t as student_t, gennorm, genpareto
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

returns, prices = {}, {}
for name, ticker in tickers.items():
    try:
        data = yf.download(ticker, start='2005-01-01', end='2025-12-31', progress=False)
        close = data['Close'].squeeze().dropna()
        rets = np.log(close / close.shift(1)).dropna()
        rets = rets[np.isfinite(rets)]
        prices[name] = close
        returns[name] = rets
        print(f"   {name:>10}: {len(rets)} obs")
    except Exception as e:
        print(f"   {name:>10}: FAILED ({e})")

def compute_aic_bic(ll, k, n):
    aic = -2.0 * ll + 2.0 * k
    bic = -2.0 * ll + k * np.log(n)
    return aic, bic

dist_names = ['Normal', 'Student-t', 'GED', 'Laplace', 'Logistic', 'Cauchy']
aic_matrix = np.zeros((len(dist_names), len(tickers)))
bic_matrix = np.zeros_like(aic_matrix)
fit_params = {}  # store for later use

for j, (asset, rets) in enumerate(returns.items()):
    r = rets.values
    n = len(r)
    fit_params[asset] = {}

    # Normal
    mu, sig = norm.fit(r)
    ll = np.sum(norm.logpdf(r, mu, sig))
    aic_matrix[0, j], bic_matrix[0, j] = compute_aic_bic(ll, 2, n)
    fit_params[asset]['Normal'] = (mu, sig)

    # Student-t
    df_t, loc_t, sc_t = student_t.fit(r)
    ll = np.sum(student_t.logpdf(r, df_t, loc_t, sc_t))
    aic_matrix[1, j], bic_matrix[1, j] = compute_aic_bic(ll, 3, n)
    fit_params[asset]['Student-t'] = (df_t, loc_t, sc_t)

    # GED
    b_g, loc_g, sc_g = gennorm.fit(r)
    ll = np.sum(gennorm.logpdf(r, b_g, loc_g, sc_g))
    aic_matrix[2, j], bic_matrix[2, j] = compute_aic_bic(ll, 3, n)
    fit_params[asset]['GED'] = (b_g, loc_g, sc_g)

    # Laplace
    loc_l, sc_l = stats.laplace.fit(r)
    ll = np.sum(stats.laplace.logpdf(r, loc_l, sc_l))
    aic_matrix[3, j], bic_matrix[3, j] = compute_aic_bic(ll, 2, n)
    fit_params[asset]['Laplace'] = (loc_l, sc_l)

    # Logistic
    loc_lo, sc_lo = stats.logistic.fit(r)
    ll = np.sum(stats.logistic.logpdf(r, loc_lo, sc_lo))
    aic_matrix[4, j], bic_matrix[4, j] = compute_aic_bic(ll, 2, n)
    fit_params[asset]['Logistic'] = (loc_lo, sc_lo)

    # Cauchy
    loc_c, sc_c = stats.cauchy.fit(r)
    ll = np.sum(stats.cauchy.logpdf(r, loc_c, sc_c))
    aic_matrix[5, j], bic_matrix[5, j] = compute_aic_bic(ll, 2, n)
    fit_params[asset]['Cauchy'] = (loc_c, sc_c)

    print(f"   {asset}: all distributions fitted")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
asset_list = list(returns.keys())

for idx, (matrix, title) in enumerate([(aic_matrix, 'AIC'), (bic_matrix, 'BIC')]):
    ax = axes[idx]
    # Normalize per column (lower is better, so invert for color: best=dark)
    delta = matrix - matrix.min(axis=0, keepdims=True)
    im = ax.imshow(delta, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(asset_list)))
    ax.set_xticklabels(asset_list)
    ax.set_yticks(range(len(dist_names)))
    ax.set_yticklabels(dist_names)
    ax.set_title(f'{title} (relative to best)', fontweight='bold')

    # Annotate with values
    for i in range(len(dist_names)):
        for jj in range(len(asset_list)):
            val = delta[i, jj]
            color = 'white' if val > delta.max() * 0.6 else 'black'
            ax.text(jj, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    # Mark best per column
    best_idx = matrix.argmin(axis=0)
    for jj, bi in enumerate(best_idx):
        ax.add_patch(plt.Rectangle((jj-0.5, bi-0.5), 1, 1,
                     fill=False, edgecolor=FOREST, linewidth=2))

    plt.colorbar(im, ax=ax, shrink=0.8, label=f'$\\Delta${title}')

plt.tight_layout()
save_fig('ch2_cs2c_model_selection_heatmap')

# Align S&P 500 and BRD on common dates
common_idx = returns['S&P 500'].index.intersection(returns['BRD'].index)
r_sp = returns['S&P 500'].loc[common_idx].values
r_brd = returns['BRD'].loc[common_idx].values

# Empirical copula: rank transform to uniform
u_sp = stats.rankdata(r_sp) / (len(r_sp) + 1)
u_brd = stats.rankdata(r_brd) / (len(r_brd) + 1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Full copula
ax = axes[0]
ax.scatter(u_sp, u_brd, s=0.5, alpha=0.3, color=MAIN_BLUE)
ax.set_xlabel('S&P 500 (rank)')
ax.set_ylabel('BRD (rank)')
ax.set_title('Empirical copula', fontweight='bold')
rho = np.corrcoef(r_sp, r_brd)[0, 1]
tau = stats.kendalltau(r_sp, r_brd).statistic
ax.text(0.03, 0.97, f'$\\rho$ = {rho:.3f}\n$\\tau$ = {tau:.3f}',
        transform=ax.transAxes, ha='left', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Lower tail zoom (both < 10th percentile)
ax = axes[1]
mask_lower = (u_sp < 0.10) & (u_brd < 0.10)
ax.scatter(u_sp[mask_lower], u_brd[mask_lower], s=3, alpha=0.5, color=CRIMSON)
ax.set_xlim(0, 0.10)
ax.set_ylim(0, 0.10)
ax.set_xlabel('S&P 500 (rank)')
ax.set_ylabel('BRD (rank)')
ax.set_title('Lower tail (< 10%)', fontweight='bold')
# Lower tail dependence coefficient
lambda_L = np.mean(mask_lower) / 0.10
ax.text(0.97, 0.97, f'$\\lambda_L$ = {lambda_L:.3f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Upper tail zoom
ax = axes[2]
mask_upper = (u_sp > 0.90) & (u_brd > 0.90)
ax.scatter(u_sp[mask_upper], u_brd[mask_upper], s=3, alpha=0.5, color=FOREST)
ax.set_xlim(0.90, 1.0)
ax.set_ylim(0.90, 1.0)
ax.set_xlabel('S&P 500 (rank)')
ax.set_ylabel('BRD (rank)')
ax.set_title('Upper tail (> 90%)', fontweight='bold')
lambda_U = np.mean(mask_upper) / 0.10
ax.text(0.97, 0.97, f'$\\lambda_U$ = {lambda_U:.3f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
save_fig('ch2_cs2c_copula_scatter')

# Compute VaR(1%) and ES(1%) for all 6 distributions x 3 assets
alpha_q = 0.01
var_all = np.zeros((len(dist_names), len(tickers)))
es_all  = np.zeros_like(var_all)

for j, (asset, rets) in enumerate(returns.items()):
    r = rets.values
    fp = fit_params[asset]

    # Normal
    mu, sig = fp['Normal']
    var_all[0, j] = -(mu + sig * norm.ppf(alpha_q)) * 100
    es_all[0, j] = -(mu - sig * norm.pdf(norm.ppf(alpha_q)) / alpha_q) * 100

    # Student-t
    df_t, loc_t, sc_t = fp['Student-t']
    var_all[1, j] = -(loc_t + sc_t * student_t.ppf(alpha_q, df_t)) * 100
    t_q = student_t.ppf(alpha_q, df_t)
    es_factor = (student_t.pdf(t_q, df_t) / alpha_q) * ((df_t + t_q**2) / (df_t - 1))
    es_all[1, j] = -(loc_t - sc_t * es_factor) * 100

    # GED
    b_g, loc_g, sc_g = fp['GED']
    var_all[2, j] = -(gennorm.ppf(alpha_q, b_g, loc_g, sc_g)) * 100
    # ES via numerical integration
    x_grid = np.linspace(gennorm.ppf(1e-6, b_g, loc_g, sc_g),
                         gennorm.ppf(alpha_q, b_g, loc_g, sc_g), 5000)
    es_all[2, j] = -(np.trapz(x_grid * gennorm.pdf(x_grid, b_g, loc_g, sc_g), x_grid) / alpha_q) * 100

    # Laplace
    loc_l, sc_l = fp['Laplace']
    var_all[3, j] = -(stats.laplace.ppf(alpha_q, loc_l, sc_l)) * 100
    x_grid = np.linspace(stats.laplace.ppf(1e-6, loc_l, sc_l),
                         stats.laplace.ppf(alpha_q, loc_l, sc_l), 5000)
    es_all[3, j] = -(np.trapz(x_grid * stats.laplace.pdf(x_grid, loc_l, sc_l), x_grid) / alpha_q) * 100

    # Logistic
    loc_lo, sc_lo = fp['Logistic']
    var_all[4, j] = -(stats.logistic.ppf(alpha_q, loc_lo, sc_lo)) * 100
    x_grid = np.linspace(stats.logistic.ppf(1e-6, loc_lo, sc_lo),
                         stats.logistic.ppf(alpha_q, loc_lo, sc_lo), 5000)
    es_all[4, j] = -(np.trapz(x_grid * stats.logistic.pdf(x_grid, loc_lo, sc_lo), x_grid) / alpha_q) * 100

    # Cauchy
    loc_c, sc_c = fp['Cauchy']
    var_all[5, j] = -(stats.cauchy.ppf(alpha_q, loc_c, sc_c)) * 100
    # ES for Cauchy is infinite, use empirical
    tail_vals = r[r < np.percentile(r, 1)]
    es_all[5, j] = -np.mean(tail_vals) * 100 if len(tail_vals) > 0 else var_all[5, j]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
asset_list = list(returns.keys())
x = np.arange(len(asset_list))
width = 0.13
dist_colors = [MAIN_BLUE, CRIMSON, FOREST, AMBER, ORANGE, '#8E44AD']

for panel, (data, ylabel, title) in enumerate(
    [(var_all, 'VaR (1%) [%]', 'Value-at-Risk (1%)'),
     (es_all, 'ES (1%) [%]', 'Expected Shortfall (1%)')]):
    ax = axes[panel]
    for i in range(len(dist_names)):
        ax.bar(x + i * width, data[i, :], width, label=dist_names[i],
               color=dist_colors[i], alpha=0.85)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(asset_list)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=3, frameon=False, fontsize=7)

plt.tight_layout()
save_fig('ch2_cs2c_var_es_all')

# Use S&P 500 for backtest (longest history)
sp_rets = returns['S&P 500']
r_arr = sp_rets.values
dates = sp_rets.index
window = 250
alpha_bt = 0.01

var_normal = np.full(len(r_arr), np.nan)
var_t = np.full(len(r_arr), np.nan)
var_hist = np.full(len(r_arr), np.nan)

for i in range(window, len(r_arr)):
    w = r_arr[i - window:i]
    mu_w, sig_w = norm.fit(w)
    var_normal[i] = mu_w + sig_w * norm.ppf(alpha_bt)

    df_w, loc_w, sc_w = student_t.fit(w)
    var_t[i] = loc_w + sc_w * student_t.ppf(alpha_bt, df_w)

    var_hist[i] = np.percentile(w, alpha_bt * 100)

# Plot two crisis windows
crisis_windows = [
    ('2007-06-01', '2009-06-01', 'Global Financial Crisis 2008'),
    ('2019-06-01', '2021-06-01', 'COVID-19 Crisis 2020')
]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

for idx, (start, end, title) in enumerate(crisis_windows):
    mask = (dates >= start) & (dates <= end)
    if mask.sum() == 0:
        continue
    ax = axes[idx]
    d = dates[mask]
    r_plot = r_arr[mask]

    ax.plot(d, r_plot, color='grey', linewidth=0.4, alpha=0.6, label='Returns')
    ax.plot(d, var_normal[mask], color=MAIN_BLUE, linewidth=0.8, label='Normal VaR')
    ax.plot(d, var_t[mask], color=CRIMSON, linewidth=0.8, label='Student-t VaR')
    ax.plot(d, var_hist[mask], color=FOREST, linewidth=0.8, label='Historical VaR')

    # Mark exceedances (Normal)
    exceed_n = (r_plot < var_normal[mask]) & np.isfinite(var_normal[mask])
    if exceed_n.any():
        ax.scatter(d[exceed_n], r_plot[exceed_n], color=CRIMSON,
                   s=12, zorder=5, marker='x', linewidths=0.8)

    ax.axhline(0, color='grey', linewidth=0.3)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Log-return')

    n_days = np.sum(np.isfinite(var_normal[mask]))
    n_exc_n = np.sum(exceed_n)
    ax.text(0.97, 0.03, f'Normal exceedances: {n_exc_n}/{n_days} ({n_exc_n/n_days*100:.1f}%)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False, fontsize=7)

plt.tight_layout()
save_fig('ch2_cs2c_backtest_crises')

# Basel traffic light: 250-day backtest
# Green: 0-4 exceedances, Yellow: 5-9, Red: 10+

def traffic_light_color(n_exc):
    if n_exc <= 4:
        return FOREST, 'Green'
    elif n_exc <= 9:
        return AMBER, 'Yellow'
    else:
        return CRIMSON, 'Red'

model_labels = ['Normal', 'Student-t', 'Historical']
asset_list = list(returns.keys())

# Compute exceedances for last 250 days for each asset/model
results_tl = {}
for asset in asset_list:
    r_a = returns[asset].values
    n_a = len(r_a)
    if n_a < 500:
        # Need at least 500 obs (250 window + 250 test)
        results_tl[asset] = {m: (0, 250) for m in model_labels}
        continue

    test_start = n_a - 250
    exc_counts = {}
    for i in range(test_start, n_a):
        w = r_a[i - 250:i]
        actual = r_a[i]

        mu_w, sig_w = norm.fit(w)
        var_n = mu_w + sig_w * norm.ppf(alpha_bt)

        df_w, loc_w, sc_w = student_t.fit(w)
        var_ti = loc_w + sc_w * student_t.ppf(alpha_bt, df_w)

        var_h = np.percentile(w, alpha_bt * 100)

        for m, v in zip(model_labels, [var_n, var_ti, var_h]):
            if m not in exc_counts:
                exc_counts[m] = 0
            if actual < v:
                exc_counts[m] += 1

    results_tl[asset] = exc_counts
    print(f"   {asset}: {exc_counts}")

# Plot traffic light matrix
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-0.5, len(model_labels) - 0.5)
ax.set_ylim(-0.5, len(asset_list) - 0.5)
ax.set_xticks(range(len(model_labels)))
ax.set_xticklabels(model_labels, fontweight='bold')
ax.set_yticks(range(len(asset_list)))
ax.set_yticklabels(asset_list, fontweight='bold')
ax.set_title('Basel Traffic Light Test (250-day backtest, VaR 1%)', fontweight='bold')
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

for i, asset in enumerate(asset_list):
    for jj, model in enumerate(model_labels):
        n_exc = results_tl[asset].get(model, 0)
        color, label = traffic_light_color(n_exc)
        circle = plt.Circle((jj, i), 0.35, color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(jj, i, f'{n_exc}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=FOREST,
           markersize=10, label='Green (0-4)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=AMBER,
           markersize=10, label='Yellow (5-9)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=CRIMSON,
           markersize=10, label='Red (10+)')
]
ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=8)

ax.set_aspect('equal')
plt.tight_layout()
save_fig('ch2_cs2c_basel_traffic_light')