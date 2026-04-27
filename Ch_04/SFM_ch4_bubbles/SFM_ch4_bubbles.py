"""
SFM_ch4_bubbles
===============
Historical bubbles: Dotcom (Nasdaq), Subprime (CSUSHPINSA), Bitcoin 2021.

Description:
- Download real data for 3 bubbles
  * Nasdaq Composite (^IXIC) 1998-2002 dotcom
  * S&P 500 as subprime proxy 2005-2009 (housing data requires FRED)
  * Bitcoin (BTC-USD) 2019-2023 crypto bubble
- Normalize each to peak = 100
- Plot aligned on a common "days from start" axis

Output:
- sfm_ch4_bubbles.pdf
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
FOREST = (46 / 255, 125 / 255, 50 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})


def normalize_to_peak(series, start, end):
    s = series.loc[start:end].squeeze()
    s = s / s.max() * 100
    return np.arange(len(s)), s.values


# Dotcom (Nasdaq 1998-2002)
ix = yf.download("^IXIC", start="1998-01-01", end="2002-12-31",
                 auto_adjust=True, progress=False)["Close"].squeeze()
t_dc, v_dc = normalize_to_peak(ix, "1998-01-01", "2002-12-31")

# Subprime: use ^GSPC as rough proxy (peak Oct 2007)
sp = yf.download("^GSPC", start="2005-01-01", end="2009-06-30",
                 auto_adjust=True, progress=False)["Close"].squeeze()
t_sp, v_sp = normalize_to_peak(sp, "2005-01-01", "2009-06-30")

# Bitcoin 2021 peak
btc = yf.download("BTC-USD", start="2019-01-01", end="2023-06-30",
                  auto_adjust=True, progress=False)["Close"].squeeze()
t_bt, v_bt = normalize_to_peak(btc, "2019-01-01", "2023-06-30")

fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.plot(t_dc, v_dc, color=MAIN_BLUE, linewidth=2, label="Dotcom (Nasdaq, 1998--2002)")
ax.plot(t_sp, v_sp, color=CRIMSON, linewidth=2, label="Subprime (S&P 500, 2005--2009)")
ax.plot(t_bt, v_bt, color=FOREST, linewidth=2, label="Crypto (Bitcoin, 2019--2023)")
ax.set_xlabel("days from window start")
ax.set_ylabel("price (index = 100 at peak)")
ax.set_title(r"Bubbles and crashes --- normalized at peak")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_bubbles.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved bubbles plot with 3 real-data series")
