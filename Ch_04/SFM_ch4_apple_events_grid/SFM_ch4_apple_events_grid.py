"""
SFM_ch4_apple_events_grid
=========================
Grid of six individual Apple earnings event studies.

Description:
- AAPL + SPY daily via yfinance
- For each of 6 selected earnings dates: estimate alpha,beta on [-250,-11]
- Compute AR_t and CAR_t over [-10, +15]
- 2x3 panel grid: each panel shows one event with its own CAR curve
- Highlights heterogeneity across events (some show drift, others sharp jumps)

Output:
- sfm_ch4_apple_events_grid.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

try:
    CHARTS = Path(__file__).resolve().parents[3] / "charts"
except NameError:
    CHARTS = Path.cwd().resolve().parents[2] / "charts"
CHARTS.mkdir(exist_ok=True)

MAIN_BLUE = (26 / 255, 58 / 255, 110 / 255)
CRIMSON = (205 / 255, 0 / 255, 0 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})

PRE, POST = -10, 15
EST_LEN, EST_GAP = 240, 10

events = pd.to_datetime([
    "2020-01-28",  # Q1 FY20 - pre-COVID record
    "2020-04-30",  # Q2 FY20 - COVID quarter
    "2020-07-30",  # Q3 FY20 - WFH boom
    "2022-10-27",  # Q4 FY22 - macro headwinds
    "2023-08-03",  # Q3 FY23 - China weakness
    "2024-08-01",  # Q3 FY24 - AI narrative
])

aapl = yf.download("AAPL", start="2018-06-01", end="2024-12-31",
                   auto_adjust=True, progress=False)
spy = yf.download("SPY", start="2018-06-01", end="2024-12-31",
                  auto_adjust=True, progress=False)
ret = pd.concat([
    np.log(aapl["Close"].squeeze()).diff().rename("aapl"),
    np.log(spy["Close"].squeeze()).diff().rename("spy"),
], axis=1).dropna()

t_rel = np.arange(PRE, POST + 1)

fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True)
for ax, ev in zip(axes.ravel(), events):
    pos = ret.index.searchsorted(ev)
    est = ret.iloc[pos + PRE - EST_GAP - EST_LEN:pos + PRE - EST_GAP]
    rm, ri = est["spy"].values, est["aapl"].values
    cov = np.cov(ri, rm, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    alpha = ri.mean() - beta * rm.mean()
    win = ret.iloc[pos + PRE:pos + POST + 1]
    ar = win["aapl"].values - (alpha + beta * win["spy"].values)
    car = np.cumsum(ar) * 100
    sigma_eps = (ri - (alpha + beta * rm)).std(ddof=1)
    se = sigma_eps * np.sqrt(np.abs(t_rel) + 1) * 100
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.fill_between(t_rel, car - 1.96 * se, car + 1.96 * se,
                    color=CRIMSON, alpha=0.15)
    ax.plot(t_rel, car, "o-", color=CRIMSON, linewidth=1.4, markersize=3)
    ax.set_title(f"AAPL earnings {ev.strftime('%Y-%m-%d')}",
                 fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

for ax in axes[1, :]:
    ax.set_xlabel(r"days relative to event $t$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"CAR (%)")

fig.suptitle(r"AAPL: six individual earnings event studies (CAR with 95% CI)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(CHARTS / "sfm_ch4_apple_events_grid.pdf",
            bbox_inches="tight", pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_apple_events_grid.pdf")
