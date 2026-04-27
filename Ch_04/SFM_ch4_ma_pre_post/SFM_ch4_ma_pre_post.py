"""
SFM_ch4_ma_pre_post
===================
Backtest MA(50)/MA(200) crossover pe S&P 500 1960-2024.

Description:
- Date: ^GSPC zilnic via yfinance (cât mai mult istoric)
- Strategie: long când MA(50) > MA(200), flat altfel
- Calculează excess return anualizat vs Buy & Hold pe ferestre de 5 ani
- Plotează seria pe an, marcând 1990 ca punct de inflexiune (BLL 1992)
- Stil: fond transparent, legendă centrată sub axă

Output:
- sfm_ch4_ma_pre_post.pdf
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
FOREST = (46 / 255, 125 / 255, 50 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})


df = yf.download("^GSPC", start="1960-01-01", end="2024-12-31",
                 auto_adjust=True, progress=False)
prices = df["Close"].squeeze().dropna()
print(f"S&P 500: {len(prices)} zile, {prices.index[0].year}--{prices.index[-1].year}")

ma50 = prices.rolling(50).mean()
ma200 = prices.rolling(200).mean()
signal = (ma50 > ma200).astype(int).shift(1).fillna(0)
ret = np.log(prices).diff().fillna(0)
strat_ret = signal * ret
bh_ret = ret
excess = strat_ret - bh_ret  # excess log-return zilnic

# Excess return anualizat pe an calendaristic
excess_annual = excess.groupby(excess.index.year).sum()
years = excess_annual.index.values
ann = excess_annual.values

# Rolling 5-ani al excess return-ului anual
window = 5
rolling = np.full(len(years), np.nan)
for i in range(window - 1, len(years)):
    rolling[i] = ann[i - window + 1:i + 1].mean()

fig, ax = plt.subplots(figsize=(7.6, 4.3))
ax.axhline(0, color="gray", linewidth=0.4)
ax.axvline(1990, color=CRIMSON, linestyle="--", linewidth=1,
           label="inflection point (1990)")
mask = ~np.isnan(rolling)
ax.fill_between(years[mask], 0, rolling[mask],
                where=(rolling[mask] > 0),
                color=FOREST, alpha=0.25, label="excess return $> 0$")
ax.fill_between(years[mask], 0, rolling[mask],
                where=(rolling[mask] <= 0),
                color=CRIMSON, alpha=0.25, label="excess return $\\leq 0$")
ax.plot(years[mask], rolling[mask], color=MAIN_BLUE, linewidth=1.6,
        label="excess return MA crossover (5-year rolling)")
ax.set_xlabel("Year")
ax.set_ylabel("annualized excess return (vs B&H)")
ax.set_title(r"MA(50)/MA(200) pre vs post 1990 --- AMH evidence (S&P 500)")
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_ma_pre_post.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved sfm_ch4_ma_pre_post.pdf")
