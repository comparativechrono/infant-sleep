"""Figures for the NYU per-infant rhythm analysis.

Usage: python3 -I nyu_rhythm_figures.py <infant_sleep_preprocessed.mat> <analysis_dir>
Reads nyu_rhythm_windows.npz and nyu_infant_rhythm_summary.csv from <analysis_dir> and writes
nyu_24h_rhythm_emergence.png and nyu_example_actograms.png there.
"""
import sys
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperfig import headline, lettered, save  # noqa: E402


MAT, OUT = sys.argv[1], Path(sys.argv[2])
z = np.load(OUT / "nyu_rhythm_windows.npz")
s24 = z["s24"].astype(float)
summ = pd.read_csv(OUT / "nyu_infant_rhythm_summary.csv")
data = loadmat(MAT)["data"]

SURFACE, INK, INK_2, MUTED, HAIRLINE, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                              "#e1e0d9", "#c3c2b7")
BLUE, BLUE_100 = "#2a78d6", "#cde2fb"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
seq = LinearSegmentedColormap.from_list("seq_blue", RAMP)
seq.set_bad(HAIRLINE)
grey = LinearSegmentedColormap.from_list("sleep", [SURFACE, INK])
grey.set_bad(HAIRLINE)
for f in glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 9})
DPI = 220
MAX_AGE = 240
MONTH = 365.25 / 12


def style(ax, xlabel_top=False):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=8, length=3, width=0.6)
    if xlabel_top:
        ax.tick_params(axis="x", top=True, labeltop=True)


def month_axis(ax, label=True):
    ax.set_xlim(0, MAX_AGE)
    ax.set_xticks([m * MONTH for m in range(9)])
    ax.set_xticklabels([str(m) for m in range(9)] if label else [])


def title(ax, text, dy=8):
    ax.annotate(lettered(text), xy=(0, 1), xycoords="axes fraction", xytext=(0, dy),
                textcoords="offset points", fontsize=11, fontweight="semibold", color=INK,
                va="bottom", ha="left")


# ======================= Figure 1: emergence summary =======================
age = np.arange(1, s24.shape[1] + 1)
med = np.nanmedian(s24, axis=0)
q1, q3 = np.nanquantile(s24, 0.25, axis=0), np.nanquantile(s24, 0.75, axis=0)
n_age = np.sum(~np.isnan(s24), axis=0)
ok = n_age >= 50
level6 = med[181]                                     # population median at 6 months (day 182)
cross = int(np.argmax((med >= level6 / 2) & ok)) + 1  # first day the median reaches half of it

el = summ[summ["eligible"]].copy()
t50 = el["t50_day"].to_numpy()
t50_med, t50_q1, t50_q3 = np.median(t50), np.quantile(t50, 0.25), np.quantile(t50, 0.75)
order = el.sort_values(["t50_day", "infant"])["infant"].to_numpy() - 1
rel = s24[order, :MAX_AGE] / el.set_index("infant").loc[order + 1, "plateau_s24"].to_numpy()[:, None]

L, PW, R = 1.15, 6.85, 0.4
H_A, H_B, H_C = 2.3, 1.45, 4.6
G_AB, G_BC = 0.75, 0.75
TOP, BOTTOM = 1.95, 1.5
W = L + PW + R
H = TOP + H_A + G_AB + H_B + G_BC + H_C + BOTTOM
fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
yC = BOTTOM
yB = yC + H_C + G_BC
yA = yB + H_B + G_AB
axA = fig.add_axes([L / W, yA / H, PW / W, H_A / H])
axB = fig.add_axes([L / W, yB / H, PW / W, H_B / H])
axC = fig.add_axes([L / W, yC / H, PW / W, H_C / H])

# A: population trajectory
sel = ok & (age <= MAX_AGE)
axA.fill_between(age[sel], q1[sel] * 100, q3[sel] * 100, color=BLUE_100, linewidth=0)
axA.plot(age[sel], med[sel] * 100, color=BLUE, linewidth=2)
axA.axhline(level6 / 2 * 100, color=AXIS, linewidth=0.6, zorder=0)
axA.plot([cross], [med[cross - 1] * 100], "o", color=INK, markersize=4.5, zorder=5)
axA.annotate(f"half the 6-month level\nreached at day {cross}", xy=(cross, med[cross - 1] * 100),
             xytext=(10, -26), textcoords="offset points", fontsize=8.5, color=INK_2, va="top")
axA.annotate("median", xy=(200, med[199] * 100), xytext=(0, 7), textcoords="offset points",
             fontsize=8.5, color=INK_2, ha="center")
axA.annotate("middle 50% of infants", xy=(200, q1[199] * 100), xytext=(0, -6),
             textcoords="offset points", fontsize=8.5, color=INK_2, ha="center", va="top")
axA.set_ylim(0, 60)
axA.set_yticks([0, 20, 40, 60])
axA.grid(axis="y", color=HAIRLINE, linewidth=0.6)
axA.set_axisbelow(True)
month_axis(axA)
style(axA)
title(axA, "How strong the 24-hour sleep rhythm is, by age")
axA.set_ylabel("Variance explained (%)", color=INK_2, fontsize=8.5)

# B: distribution of emergence ages
bins = np.arange(0, MAX_AGE + 7, 7)
counts, edges = np.histogram(t50, bins=bins)
axB.bar(edges[:-1] + 0.5, counts, width=6, align="edge", color=BLUE, linewidth=0)
axB.axvline(t50_med, color=INK, linewidth=0.8)
axB.annotate(f"median day {t50_med:g} (line)\nmiddle 50%: days {t50_q1:.0f}–{t50_q3:.0f}",
             xy=(105, counts.max()), fontsize=8.5, color=INK_2, va="top", linespacing=1.4)
axB.grid(axis="y", color=HAIRLINE, linewidth=0.6)
axB.set_axisbelow(True)
month_axis(axB)
style(axB)
title(axB, f"Age at which each infant's rhythm reached half its plateau ({len(el)} infants)")
axB.set_ylabel("Infants", color=INK_2, fontsize=8.5)

# C: individual trajectories sorted by emergence age
axC.imshow(np.ma.masked_invalid(rel), cmap=seq, vmin=0, vmax=1, aspect="auto",
           interpolation="nearest", extent=(0, MAX_AGE, len(order), 0))
month_axis(axC)
style(axC)
axC.set_yticks([])
title(axC, "Each infant, sorted by that age (earliest at top)")
axC.set_xlabel("Age (months)", color=INK_2, fontsize=9)
cax = fig.add_axes([L / W, (BOTTOM - 0.85) / H, 2.4 / W, 0.09 / H])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=seq, norm=plt.Normalize(0, 1)), cax=cax,
                  orientation="horizontal")
cb.outline.set_visible(False)
cb.set_ticks([0, 0.5, 1])
cb.set_ticklabels(["0", "half", "plateau or above"])
cax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=7.5, length=2.5, width=0.6)
cax.annotate("24-h strength relative to the infant's own plateau · grey: too few days logged",
             xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
             fontsize=8, color=INK_2, va="bottom")

headline(fig, L / W, 1 - 0.36 / H, "When does the 24-hour sleep rhythm emerge?", fontsize=20,
         fontweight="bold", color=INK, ha="left", va="top")
headline(fig, L / W, 1 - 0.82 / H,
         "NYU Baby Sleep Study, 700 infants. Lomb–Scargle periodogram of each infant's sleep in 14-day "
         "windows;\nstrength is the share of variance a 24-h sinusoid explains. Plateau = the infant's "
         "median over days 120–240.\nBottom panels: infants logged from their first 3 weeks.",
         fontsize=9, color=INK_2, ha="left", va="top", linespacing=1.45)
headline(fig, L / W, 0.18 / H, "Data: Liu et al., arXiv:1904.04780 (2019), infant_sleep_preprocessed.mat.",
         fontsize=7.5, color=MUTED, ha="left", va="bottom")
save(fig, OUT / "nyu_24h_rhythm_emergence.png", DPI, "Figure3", SURFACE)
plt.close(fig)
print(f"fig1: population half-level day {cross}; t50 median {t50_med:.0f} IQR {t50_q1:.0f}-{t50_q3:.0f}")

# ======================= Figure 2: example actograms =======================
DAYS = 180
cov = np.array([np.mean(~np.isnan(data[:DAYS, :, i]).all(1)) for i in range(data.shape[2])])
good = el[cov[el["infant"] - 1] >= 0.9]
picks = []
for lab, q in (("Early", 0.1), ("Typical", 0.5), ("Late", 0.9)):
    target = np.quantile(t50, q)
    row = good.iloc[int(np.argmin(np.abs(good["t50_day"] - target)))]
    picks.append((lab, int(row["infant"]), float(row["t50_day"]), float(row["plateau_s24"])))
picks.append(("Caution", 446, np.nan, float(summ.loc[summ.infant == 446, "plateau_s24"].iloc[0])))

CELL_ACT, CELL_STRIP, CELL_GAP, COL_GAP = 2.75, 0.85, 0.3, 0.55
L2, R2 = 0.75, 0.3
CELL_H, ROW_GAP = 4.0, 1.25
TOP2, BOTTOM2 = 1.95, 0.95
W2 = L2 + 2 * (CELL_ACT + CELL_GAP + CELL_STRIP) + COL_GAP + R2
H2 = TOP2 + 2 * CELL_H + ROW_GAP + BOTTOM2
fig = plt.figure(figsize=(W2, H2), dpi=DPI, facecolor=SURFACE)
for k, (lab, inf, t50_i, plat) in enumerate(picks):
    col, rowi = k % 2, k // 2
    x0 = L2 + col * (CELL_ACT + CELL_GAP + CELL_STRIP + COL_GAP)
    y0 = BOTTOM2 + (1 - rowi) * (CELL_H + ROW_GAP)
    ax = fig.add_axes([x0 / W2, y0 / H2, CELL_ACT / W2, CELL_H / H2])
    axs = fig.add_axes([(x0 + CELL_ACT + CELL_GAP) / W2, y0 / H2, CELL_STRIP / W2, CELL_H / H2],
                       sharey=ax)
    x = data[:DAYS + 1, :, inf - 1]
    dbl = np.concatenate([x[:DAYS], x[1:DAYS + 1]], axis=1)
    ax.imshow(np.ma.masked_invalid(dbl), cmap=grey, vmin=0, vmax=1, aspect="auto",
              interpolation="nearest", extent=(0, 48, DAYS, 0))
    ax.set_xticks([0, 12, 24, 36, 48])
    ax.set_xticklabels(["00", "12", "00", "12", "24"])
    ax.set_yticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_ylim(DAYS, 0)
    style(ax)
    ax.tick_params(axis="y", colors=AXIS, labelcolor=MUTED)
    ax.set_ylabel("Day of age", color=INK_2, fontsize=8)
    sv = s24[inf - 1, :DAYS] * 100
    axs.plot(sv, np.arange(DAYS) + 0.5, color=BLUE, linewidth=1.2)
    axs.axvline(plat * 50, color=AXIS, linewidth=0.6)            # half of plateau, in %
    axs.set_xlim(0, 80)
    axs.set_xticks([0, 40, 80])
    style(axs)
    axs.tick_params(axis="y", left=False, labelleft=False)
    if not np.isnan(t50_i):
        for a in (ax, axs):
            a.axhline(t50_i - 0.5, color=INK_2, linewidth=0.7)
        axs.annotate(f"day {t50_i:.0f}", xy=(80, t50_i), xytext=(0, 3), textcoords="offset points",
                     ha="right", va="bottom", fontsize=7.5, color=INK_2)
    if lab == "Caution":
        ttl = f"Caution: infant {inf}"
        sub = "Sleep jumps 10–14 h for ~2 weeks (days ~76–92),\nlikely travel; it fakes a ~25-h periodogram peak"
        for a in (ax, axs):
            a.axhspan(75.5, 92.5, xmin=0, xmax=1, facecolor="none", edgecolor=INK_2, linewidth=0.8)
    else:
        ttl = f"{lab} emerger: infant {inf}"
        sub = f"Half its plateau by day {t50_i:.0f}; line marks that day"
    ax.annotate(lettered(ttl), xy=(0, 1), xycoords="axes fraction", xytext=(0, 22 + 12 * sub.count("\n")),
                textcoords="offset points",
                fontsize=10.5, fontweight="semibold", color=INK, va="bottom")
    ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
                fontsize=8, color=INK_2, va="bottom", linespacing=1.3)
    ax.annotate("Clock time, two days per row", xy=(0.5, 0), xycoords="axes fraction",
                xytext=(0, -20), textcoords="offset points", ha="center", va="top", fontsize=8,
                color=INK_2)
    axs.annotate("24-h strength (%)", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -20),
                 textcoords="offset points", ha="center", va="top", fontsize=8, color=INK_2)

headline(fig, L2 / W2, 1 - 0.36 / H2, "Four infants up close: first six months", fontsize=20,
         fontweight="bold", color=INK, ha="left", va="top")
headline(fig, L2 / W2, 1 - 0.82 / H2,
         "Double-plotted sleep (dark = asleep, grey = not logged) beside each infant's 24-h rhythm "
         "strength.\nThe thin vertical line in each strip is half the infant's plateau; emergence is "
         "when the trace stays past it.",
         fontsize=9, color=INK_2, ha="left", va="top", linespacing=1.45)
headline(fig, L2 / W2, 0.18 / H2, "Data: Liu et al., arXiv:1904.04780 (2019), infant_sleep_preprocessed.mat.",
         fontsize=7.5, color=MUTED, ha="left", va="bottom")
save(fig, OUT / "nyu_example_actograms.png", DPI, "Figure2", SURFACE)
plt.close(fig)
print("fig2 picks:", picks)
