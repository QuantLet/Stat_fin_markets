"""
SFM_ch4_rolling_hurst
=====================
Rolling Hurst exponent on S&P 500 --- AMH empirical evidence.

Description:
- Download real S&P 500 (^GSPC) daily 2000-2024 via yfinance
- Compute rolling Hurst exponent via R/S analysis on 2-year window
- Plot time-varying H with crisis periods highlighted
- Consistent with Lo (2004) Adaptive Markets Hypothesis

Output:
- sfm_ch4_rolling_hurst.pdf
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

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})


def hurst_rs(x, min_n=10, max_n=None):
    x = np.asarray(x, dtype=float)
    T = len(x)
    if max_n is None:
        max_n = T // 2
    ns = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), 20).astype(int))
    rs_values = []
    for n in ns:
        if n < 2:
            continue
        # split into chunks
        n_chunks = T // n
        rs_list = []
        for i in range(n_chunks):
            chunk = x[i * n:(i + 1) * n]
            mu = chunk.mean()
            z = np.cumsum(chunk - mu)
            R = z.max() - z.min()
            S = chunk.std(ddof=0)
            if S > 0 and R > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        else:
            rs_values.append(np.nan)
    ns_valid = ns[:len(rs_values)]
    rs_values = np.array(rs_values)
    mask = ~np.isnan(rs_values) & (rs_values > 0)
    if mask.sum() < 4:
        return np.nan
    slope, _ = np.polyfit(np.log(ns_valid[mask]), np.log(rs_values[mask]), 1)
    return slope


df = yf.download("^GSPC", start="1998-01-01", end="2024-12-31",
                  auto_adjust=True, progress=False)
prices = df["Close"].squeeze().dropna()
r = np.log(prices).diff().dropna()

window = 504  # 2 years of daily data
H = []
dates = []
for end in range(window, len(r), 21):  # monthly stepping
    segment = r.iloc[end - window:end].values
    H.append(hurst_rs(segment, min_n=10, max_n=window // 4))
    dates.append(r.index[end])

H = np.array(H)
dates_numeric = np.array([d.year + (d.dayofyear - 1) / 365.25 for d in dates])

fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1,
           label="$H = 0{,}5$ (random walk)")
ax.fill_between(dates_numeric, 0.5, H, where=(H > 0.55),
                color=CRIMSON, alpha=0.20, label="perioade de ineficiență")
ax.plot(dates_numeric, H, color=MAIN_BLUE, linewidth=1.4,
        label=r"$\hat H$ rolling (2 ani)")

# Crisis annotations
crises = {"dotcom": 2001.2, "GFC": 2008.5, "COVID": 2020.2, "inflație": 2022.3}
for label, ct in crises.items():
    ax.axvline(ct, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.annotate(label, (ct, np.nanmax(H) * 0.95), fontsize=8,
                ha="center", color="gray")

ax.set_xlabel("An")
ax.set_ylabel(r"$\hat H$")
ax.set_title(r"Exponent Hurst rolling --- S&P 500 (evidență AMH)")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_rolling_hurst.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print(f"Saved with {len(H)} rolling estimates")
print(f"Mean H = {np.nanmean(H):.3f}, range = [{np.nanmin(H):.3f}, {np.nanmax(H):.3f}]")
