"""
SFM_ch4_car_with_ci
===================
Event study cu CAR și interval de încredere 95% pe Apple (AAPL).

Description:
- Date: AAPL + SPY zilnice via yfinance (2015-2024)
- Eveniment: anunțuri trimestriale earnings AAPL (yfinance get_earnings_dates)
- Fereastra de estimare [-250, -11]; fereastra evenimentului [-10, +15]
- Modelul de piață: R_i = alpha + beta * R_m + eps
- AR_t = R_i,t - (alpha_hat + beta_hat * R_m,t)
- CAR(t) cumulat pe fereastră; IC = +/- 1.96 * sigma_eps * sqrt(|t|/N)
- Stil: fond transparent, legendă centrată sub axă

Output:
- sfm_ch4_car_with_ci.pdf
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
AMBER = (181 / 255, 133 / 255, 63 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})

PRE = -10
POST = 15
EST_LEN = 240  # zile estimare (-250 .. -11)
EST_GAP = 10   # exclude ultimele 10 zile pre-event

aapl_df = yf.download("AAPL", start="2014-06-01", end="2024-12-31",
                       auto_adjust=True, progress=False)
spy_df = yf.download("SPY", start="2014-06-01", end="2024-12-31",
                      auto_adjust=True, progress=False)
ret_aapl = np.log(aapl_df["Close"].squeeze()).diff()
ret_spy = np.log(spy_df["Close"].squeeze()).diff()
returns = pd.concat([ret_aapl.rename("aapl"), ret_spy.rename("spy")],
                    axis=1).dropna()

# Date earnings AAPL --- fallback pe set fix dacă API-ul yfinance nu răspunde
try:
    tk = yf.Ticker("AAPL")
    ed = tk.get_earnings_dates(limit=40)
    earnings = pd.to_datetime(ed.index).tz_localize(None)
    earnings = earnings[earnings >= pd.Timestamp("2015-01-01")]
    earnings = earnings[earnings <= pd.Timestamp("2024-09-30")]
except Exception:
    earnings = pd.to_datetime([
        "2015-01-27", "2015-04-27", "2015-07-21", "2015-10-27",
        "2016-01-26", "2016-04-26", "2016-07-26", "2016-10-25",
        "2017-01-31", "2017-05-02", "2017-08-01", "2017-10-31",
        "2018-02-01", "2018-05-01", "2018-07-31", "2018-11-01",
        "2019-01-29", "2019-04-30", "2019-07-30", "2019-10-30",
        "2020-01-28", "2020-04-30", "2020-07-30", "2020-10-29",
        "2021-01-27", "2021-04-28", "2021-07-27", "2021-10-28",
        "2022-01-27", "2022-04-28", "2022-07-28", "2022-10-27",
        "2023-02-02", "2023-05-04", "2023-08-03", "2023-11-02",
        "2024-02-01", "2024-05-02", "2024-08-01", "2024-10-31",
    ])
print(f"AAPL earnings events: {len(earnings)}")  # noqa

ev_window = np.arange(PRE, POST + 1)
ar_matrix = []
sigma_eps_list = []
for ev_date in earnings:
    pos_idx = returns.index.searchsorted(ev_date)
    if pos_idx >= len(returns) or pos_idx + POST + 1 > len(returns):
        continue
    if pos_idx + PRE - EST_GAP - EST_LEN < 0:
        continue
    est_start = pos_idx + PRE - EST_GAP - EST_LEN
    est_end = pos_idx + PRE - EST_GAP
    est = returns.iloc[est_start:est_end]
    rm = est["spy"].values
    ri = est["aapl"].values
    cov = np.cov(ri, rm, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    alpha = ri.mean() - beta * rm.mean()
    eps = ri - (alpha + beta * rm)
    sigma_eps = eps.std(ddof=1)
    win = returns.iloc[pos_idx + PRE:pos_idx + POST + 1]
    if len(win) < POST - PRE + 1:
        continue
    ar = win["aapl"].values - (alpha + beta * win["spy"].values)
    ar_matrix.append(ar)
    sigma_eps_list.append(sigma_eps)

ar_matrix = np.array(ar_matrix)
N = len(ar_matrix)
sigma_eps = np.mean(sigma_eps_list)
print(f"N = {N} evenimente folosite, sigma_eps = {sigma_eps:.4f}")

ar_mean = ar_matrix.mean(axis=0)
car = np.cumsum(ar_mean) * 100  # %
# SE_t = sigma_eps * sqrt(|t - t_event| + 1) / sqrt(N)
t_rel = np.arange(PRE, POST + 1)
se_t = sigma_eps * np.sqrt(np.abs(t_rel) + 1) / np.sqrt(N) * 100
ci_lo, ci_hi = car - 1.96 * se_t, car + 1.96 * se_t

fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.axhline(0, color="gray", linewidth=0.4)
ax.axvline(0, color="black", linestyle="--", linewidth=1,
           label="event ($t=0$)")
ax.fill_between(t_rel, ci_lo, ci_hi, color=CRIMSON, alpha=0.18,
                label=r"95% confidence interval")
ax.plot(t_rel, car, "o-", color=CRIMSON, linewidth=1.8, markersize=4,
        label=fr"$\overline{{CAR}}(t)$ AAPL ($N = {N}$)")
ax.set_xlabel(r"days relative to event $t$")
ax.set_ylabel(r"cumulative CAR (%)")
ax.set_title(r"AAPL event study --- CAR with 95% CI on earnings announcements")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_car_with_ci.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_car_with_ci.pdf")
