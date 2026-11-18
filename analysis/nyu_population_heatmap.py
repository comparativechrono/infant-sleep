"""Population view of the NYU infant-sleep data: share of logged infants asleep by age and clock time.

Usage: python3 -I nyu_population_heatmap.py <infant_sleep_preprocessed.mat> <out.png>
"""
import sys
from pathlib import Path
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperfig import headline, lettered, save  # noqa: E402


d = loadmat(sys.argv[1])["data"]                     # (730 days, 144 bins, 700 infants)
OUT = sys.argv[2]
N_DAYS = d.shape[0]
MIN_N = 20                                           # rows with fewer logged infants are greyed out

logged = ~np.isnan(d).all(1)                         # (days, infants)
n_per_day = logged.sum(1)
share = np.nanmean(d, axis=2)                        # mean share asleep per bin over logged infants
share[n_per_day < MIN_N] = np.nan

SURFACE, INK, INK_2, MUTED, HAIRLINE, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                              "#e1e0d9", "#c3c2b7")
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
cmap.set_bad(HAIRLINE)
for f in glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 9})

DPI = 220
L, MAIN_W, GAP, SIDE_W, R = 1.05, 5.0, 0.35, 1.25, 0.35
TOP, PANEL_H, BOTTOM = 2.25, 8.3, 1.75
W, H = L + MAIN_W + GAP + SIDE_W + R, TOP + PANEL_H + BOTTOM
fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
ax = fig.add_axes([L / W, BOTTOM / H, MAIN_W / W, PANEL_H / H])
ax2 = fig.add_axes([(L + MAIN_W + GAP) / W, BOTTOM / H, SIDE_W / W, PANEL_H / H], sharey=ax)

im = ax.imshow(np.ma.masked_invalid(share), cmap=cmap, vmin=0, vmax=1, aspect="auto",
               interpolation="antialiased", extent=(0, 24, N_DAYS, 0))
ax.set_xlim(0, 24)
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])
ax.set_xticks(np.arange(0, 25, 2), minor=True)

ages = [0, 91, 183, 274, 365, 456, 548, 639, 730]
labels = ["Birth", "3 months", "6 months", "9 months", "1 year", "15 months", "18 months",
          "21 months", "2 years"]
ax.set_yticks(ages)
ax.set_yticklabels(labels)
ax.set_ylim(N_DAYS, 0)

ax2.plot(n_per_day, np.arange(N_DAYS) + 0.5, color=INK_2, linewidth=1.2)
ax2.set_xlim(0, max(700, n_per_day.max()) * 1.02)
ax2.set_xticks([0, 200, 400, 600])
ax2.axvline(MIN_N, color=AXIS, linewidth=0.6)
ax2.grid(axis="x", color=HAIRLINE, linewidth=0.6)
ax2.set_axisbelow(True)
ax2.tick_params(axis="y", left=False, labelleft=False)

for a in (ax, ax2):
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(axis="x", which="both", top=True, labeltop=True, colors=MUTED, labelcolor=MUTED,
                  labelsize=8, length=3, width=0.6, pad=3)
    a.tick_params(axis="x", which="minor", length=1.8)
ax.tick_params(axis="y", colors=AXIS, labelcolor=INK, labelsize=8.5, length=3, width=0.6)

for a, title in ((ax, lettered("Share of logged infants asleep")), (ax2, lettered("Infants logged"))):
    a.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points",
               fontsize=11, fontweight="semibold", color=INK, va="bottom", ha="left")
ax.annotate("Time of day (local clock)", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -26),
            textcoords="offset points", ha="center", va="top", fontsize=9, color=INK_2)
ax2.annotate("Infants per day", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -26),
             textcoords="offset points", ha="center", va="top", fontsize=9, color=INK_2)

cax = fig.add_axes([L / W, (BOTTOM - 1.02) / H, 2.6 / W, 0.1 / H])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
cb.set_ticklabels(["0", "25", "50", "75", "100"])
cax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=7.5, length=2.5, width=0.6)
cax.annotate(f"Asleep (%) · grey: fewer than {MIN_N} infants logged that day", xy=(0, 1),
             xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", ha="left",
             va="bottom", fontsize=8, color=INK_2)

headline(fig, L / W, 1 - 0.38 / H, "700 infants, two years of sleep", fontsize=21, fontweight="bold",
         color=INK, ha="left", va="top")
headline(fig, L / W, 1 - 0.88 / H,
         "NYU Baby Sleep Study, logged by parents in the Baby Connect app and preprocessed by Liu et al.\n"
         "(2019). Each row is one day of age; colour is the share of infants logged that day who were\n"
         "asleep in each 10-minute bin. Infants contribute only on the days their parents logged.",
         fontsize=9.5, color=INK_2, ha="left", va="top", linespacing=1.45)
headline(fig, L / W, 0.22 / H,
         "Data: Liu, Cheng, Brooks, Mackey, Heeger, Tabak & Fernandez-Granda, arXiv:1904.04780 "
         "(2019), infant_sleep_preprocessed.mat.",
         fontsize=7.5, color=MUTED, ha="left", va="bottom")
save(fig, OUT, DPI, "Figure1", SURFACE)
print("saved", OUT, f"{W * DPI:.0f}x{H * DPI:.0f}px; min/max infants per day {n_per_day.min()}/{n_per_day.max()}")
