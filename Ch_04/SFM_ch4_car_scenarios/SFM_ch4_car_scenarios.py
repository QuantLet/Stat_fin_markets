"""
SFM_ch4_car_scenarios
=====================
Event study CAR scenarios: pure surprise / leak / PEAD.

Description:
- Compute a real event study on Apple (AAPL) around earnings announcements
- Use market model (SPY as market) on estimation window
- Aggregate AAR and CAR across past 8 earnings events
- Compare with 3 theoretical scenarios

Output:
- sfm_ch4_car_scenarios.pdf
"""
import numpy as np
import pandas as pd
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
AMBER = (181 / 255, 133 / 255, 63 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})


def event_car(ticker, events, window_pre=10, window_post=10, est=120):
    """Compute CAR averaged across event dates using market-model."""
    start = min(events) - pd.Timedelta(days=est + window_pre + 30)
    end = max(events) + pd.Timedelta(days=window_post + 10)
    px = yf.download([ticker, "SPY"], start=start, end=end,
                     auto_adjust=True, progress=False)["Close"]
    rets = np.log(px).diff().dropna()
    cars = []
    for ev in events:
        ev = pd.Timestamp(ev)
        try:
            idx = rets.index.get_indexer([ev], method="nearest")[0]
            if idx < est + window_pre:
                continue
            est_win = rets.iloc[idx - est - window_pre:idx - window_pre]
            evt_win = rets.iloc[idx - window_pre:idx + window_post + 1]
            # Market-model alpha/beta
            x = est_win["SPY"].values
            y = est_win[ticker].values
            beta, alpha = np.polyfit(x, y, 1)
            ar = evt_win[ticker].values - (alpha + beta * evt_win["SPY"].values)
            cars.append(np.cumsum(ar))
        except Exception:
            continue
    return np.mean(cars, axis=0) if cars else None


# Apple earnings announcement dates (historical)
aapl_events = pd.to_datetime([
    "2022-01-27", "2022-04-28", "2022-07-28", "2022-10-27",
    "2023-02-02", "2023-05-04", "2023-08-03", "2023-11-02",
])

car = event_car("AAPL", aapl_events)
t = np.arange(-10, 11)

# Theoretical scenarios
car_surprise = np.where(t >= 0, 3.0, 0)
car_leak = np.where(t >= 0, 2.5 + 0.1 * t, 0.5 * np.maximum(t + 5, 0))
car_pead = np.where(t >= 0, 1.5 + 0.15 * t, 0.0)

fig, ax = plt.subplots(figsize=(7, 4.4))
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="event")
ax.axhline(0, color="gray", linewidth=0.4)
ax.plot(t, car_surprise, "o-", color=MAIN_BLUE, linewidth=1.6, markersize=4,
        label="(a) pure surprise")
ax.plot(t, car_leak, "s-", color=AMBER, linewidth=1.6, markersize=4,
        label="(b) information leak")
ax.plot(t, car_pead, "D-", color=CRIMSON, linewidth=1.6, markersize=4,
        label="(c) PEAD")
if car is not None:
    ax.plot(t, car * 100, "*-", color="black", linewidth=1.8, markersize=6,
            label="AAPL real (8 announcements)")
ax.set_xlabel(r"days relative to event $t$")
ax.set_ylabel(r"CAR (%)")
ax.set_title(r"Event study CAR --- 3 theoretical scenarios + real AAPL")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_car_scenarios.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved CAR scenarios plot.")
if car is not None:
    print(f"AAPL avg CAR at t=+10: {car[-1]*100:.2f}%")
