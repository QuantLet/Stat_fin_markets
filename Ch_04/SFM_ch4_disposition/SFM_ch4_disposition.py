"""
SFM_ch4_disposition
===================
Disposition effect: PGR vs PLR across investor categories.

Description:
- Published research estimates (Odean 1998, Shapira-Venezia 2001,
  Chen-Kim-Nofsinger-Rui 2007, etc.) of Proportion of Gains Realized
  vs Proportion of Losses Realized for retail and institutional investors
- Demonstrates PGR > PLR: investors sell winners too early and hold losers
  too long, a behavioral bias inconsistent with rational utility

Note: raw individual-account data is proprietary; we use published
aggregate statistics from peer-reviewed research.

Output:
- sfm_ch4_disposition.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    CHARTS = Path(__file__).resolve().parents[3] / "charts"
except NameError:
    CHARTS = Path.cwd().resolve().parents[2] / "charts"
CHARTS.mkdir(exist_ok=True)

FOREST = (46 / 255, 125 / 255, 50 / 255)
CRIMSON = (220 / 255, 53 / 255, 69 / 255)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
})

# Published research estimates (annual PGR/PLR)
# Odean 1998 (US retail), Shapira-Venezia 2001 (Israel retail),
# Chen et al. 2007 (China retail/institutional), Coval-Shumway 2005 (pros)
data = {
    "Retail SUA\n(Odean 1998)":           dict(pgr=0.148, plr=0.098),
    "Retail IL\n(Shapira-Venezia 2001)":  dict(pgr=0.135, plr=0.089),
    "Retail CN\n(Chen et al. 2007)":      dict(pgr=0.104, plr=0.082),
    "Profesioniști\n(Coval-Shumway 2005)":dict(pgr=0.095, plr=0.087),
}

categories = list(data.keys())
pgr = [data[c]["pgr"] for c in categories]
plr = [data[c]["plr"] for c in categories]
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.bar(x - width / 2, pgr, width, color=FOREST, edgecolor="black",
       linewidth=0.4, label="PGR (câștigătoare vândute)")
ax.bar(x + width / 2, plr, width, color=CRIMSON, edgecolor="black",
       linewidth=0.4, label="PLR (pierzătoare vândute)")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8.5)
ax.set_ylabel("proporție")
ax.set_title(r"Efect de dispoziție: PGR $>$ PLR (literatură)")
ax.set_ylim(0, 0.18)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
          ncol=2, frameon=False)
fig.savefig(CHARTS / "sfm_ch4_disposition.pdf", bbox_inches="tight",
            pad_inches=0.2, transparent=True)
print("Saved disposition effect plot.")
