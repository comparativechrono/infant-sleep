"""Figures for nyu_sleep_timing.py: timing by age with stability, and a gallery of flagged jumps.

Usage: python3 -I nyu_timing_figures.py <infant_sleep_preprocessed.mat> <analysis_dir>
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
from matplotlib.patches import Rectangle
from scipy.io import loadmat
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figsave import headline, save  # noqa: E402


MAT, OUT = sys.argv[1], Path(sys.argv[2])
data = loadmat(MAT)["data"]
weekly = pd.read_csv(OUT / "nyu_sleep_timing_weekly.csv")
summ = pd.read_csv(OUT / "nyu_infant_timing_summary.csv")
jumps = pd.read_csv(OUT / "nyu_phase_jumps.csv").sort_values(["infant", "start_day"])

SURFACE, INK, INK_2, MUTED, HAIRLINE, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                              "#e1e0d9", "#c3c2b7")
BLUE, BLUE_100, BLUE_200 = "#2a78d6", "#cde2fb", "#9ec5f4"
grey = LinearSegmentedColormap.from_list("sleep", [SURFACE, INK])
grey.set_bad(HAIRLINE)
for f in glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 9})
DPI = 220
MONTH = 365.25 / 12


def noon_ref(h):
    """Clock hours (0-24) -> hours after the previous midday's midnight, so nights are continuous."""
    h = np.asarray(h, dtype=float)
    return np.where(h < 12, h + 24, h)


def clock(h):
    h = float(h) % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def clock5(h):
    """Clock label rounded to 5 minutes (the data are in 10-minute bins)."""
    return clock(round(float(h) * 12) / 12)


def style(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=8, length=3, width=0.6)


def title(ax, text, sub=None):
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 26 if sub else 8),
                textcoords="offset points", fontsize=11, fontweight="semibold", color=INK, va="bottom")
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                    fontsize=8.5, color=INK_2, va="bottom")


# ================= Figure 1: timing by age + stability =================
wk = weekly.copy()
for c in ("bedtime_h", "wake_time_h", "mid_sleep_h"):
    wk[c] = noon_ref(wk[c])
g = wk.groupby("week_of_age")
pop = pd.DataFrame({
    "n": g.size(),
    "bed": g["bedtime_h"].median(), "wake": g["wake_time_h"].median(),
    "mid": g["mid_sleep_h"].median(),
    "mid_q1": g["mid_sleep_h"].quantile(0.25), "mid_q3": g["mid_sleep_h"].quantile(0.75),
})
pop = pop[pop["n"] >= 30]
age = (pop.index.to_numpy() - 0.5) * 7 / MONTH          # week centre in months

L, PW, R = 1.15, 6.85, 0.4
H_A, GAP, H_B = 4.1, 1.15, 3.3
TOP, BOTTOM = 1.95, 0.95
W, H = L + PW + R, TOP + H_A + GAP + H_B + BOTTOM
fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
axA = fig.add_axes([L / W, (BOTTOM + H_B + GAP) / H, PW / W, H_A / H])
axA.fill_between(age, pop["bed"], pop["wake"], color=BLUE_100, linewidth=0)
axA.fill_between(age, pop["mid_q1"], pop["mid_q3"], color=BLUE_200, linewidth=0)
axA.plot(age, pop["mid"], color=BLUE, linewidth=2)
yt = np.arange(18, 34, 2)
axA.set_yticks(yt)
axA.set_yticklabels([clock(h) for h in yt])
axA.set_ylim(18, 33)
axA.set_xlim(0, 24)
axA.set_xticks(range(0, 25, 3))
axA.grid(axis="y", color=HAIRLINE, linewidth=0.6)
axA.set_axisbelow(True)
style(axA)
title(axA, "Night sleep by age", "Median across infants of each infant's weekly median (jump days excluded)")
for x, y, txt, va in ((16, float(np.interp(16, age, pop["bed"])), "bedtime", "top"),
                      (16, float(np.interp(16, age, pop["wake"])), "wake time", "bottom")):
    axA.annotate(txt, xy=(x, y), xytext=(0, -5 if va == "top" else 5), textcoords="offset points",
                 ha="center", va=va, fontsize=8.5, color=INK_2)
axA.annotate("mid-sleep: median and middle 50% of infants",
             xy=(16, float(np.interp(16, age, pop["mid_q3"]))), xytext=(0, 5),
             textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK_2)
for m in (1, 6, 12, 24):
    i = int(np.argmin(np.abs(age - (m - 0.5 if m == 1 else m))))
    axA.annotate(f"{clock5(pop['bed'].iloc[i])}", xy=(age[i], pop["bed"].iloc[i]), xytext=(0, -4),
                 textcoords="offset points", ha="center", va="top", fontsize=7.5, color=MUTED)
    axA.annotate(f"{clock5(pop['wake'].iloc[i])}", xy=(age[i], pop["wake"].iloc[i]), xytext=(0, 4),
                 textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=MUTED)
axA.set_xlabel("Age (months)", color=INK_2, fontsize=9)

# B: stability of mid-sleep, 6 vs 12 months
axB = fig.add_axes([L / W, BOTTOM / H, H_B * 0.95 / W, H_B / H])
x, y = noon_ref(summ["mid_sleep_6mo"]), noon_ref(summ["mid_sleep_12mo"])
ok = ~np.isnan(x) & ~np.isnan(y)
rho = pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())
lo, hi = 22.5, 28.5
axB.plot([lo, hi], [lo, hi], color=AXIS, linewidth=0.8, zorder=1)
axB.scatter(x[ok], y[ok], s=11, color=BLUE, alpha=0.55, linewidths=0, zorder=2)
ticks = np.arange(23, 29, 1)
for setter, lim in ((axB.set_xticks, axB.set_xlim), (axB.set_yticks, axB.set_ylim)):
    setter(ticks)
    lim(lo, hi)
axB.set_xticklabels([clock(t) for t in ticks])
axB.set_yticklabels([clock(t) for t in ticks])
axB.grid(color=HAIRLINE, linewidth=0.6)
axB.set_axisbelow(True)
style(axB)
axB.set_xlabel("Mid-sleep at 6 months", color=INK_2, fontsize=9)
axB.set_ylabel("Mid-sleep at 12 months", color=INK_2, fontsize=9)
title(axB, "Infants keep their timing", f"Each dot an infant (n = {ok.sum()}); Spearman ρ = {rho:.2f}")

# C: stability across age gaps, as text table beside B
pairs = [("6mo", "12mo", "6 → 12 months"), ("12mo", "18mo", "12 → 18 months"),
         ("18mo", "24mo", "18 → 24 months"), ("6mo", "24mo", "6 → 24 months")]
tx = (L + H_B * 0.95 + 0.75) / W
fig.text(tx, (BOTTOM + H_B) / H + 26 / 72 / H, "Mid-sleep rank correlation", fontsize=11,
         fontweight="semibold", color=INK, va="bottom")
fig.text(tx, (BOTTOM + H_B) / H + 8 / 72 / H, "Spearman ρ between the same infants' medians",
         fontsize=8.5, color=INK_2, va="bottom")
for k, (a, b, lab) in enumerate(pairs):
    xa, xb = noon_ref(summ[f"mid_sleep_{a}"]), noon_ref(summ[f"mid_sleep_{b}"])
    m = ~np.isnan(xa) & ~np.isnan(xb)
    r = pd.Series(xa[m]).rank().corr(pd.Series(xb[m]).rank())
    yy = (BOTTOM + H_B - 0.45 - k * 0.62) / H
    fig.text(tx, yy, lab, fontsize=9.5, color=INK, va="center")
    fig.text(tx + 1.55 / W, yy, f"{r:.2f}", fontsize=13, fontweight="semibold", color=INK, va="center")
    fig.text(tx + 2.15 / W, yy, f"n = {m.sum()}", fontsize=8.5, color=MUTED, va="center")

headline(fig, L / W, 1 - 0.36 / H, "Sleep timing from birth to two years", fontsize=20, fontweight="bold",
         color=INK, ha="left", va="top")
headline(fig, L / W, 1 - 0.82 / H,
         "NYU Baby Sleep Study, parent-logged, local clock time. Night sleep period = bedtime to final "
         "morning waking,\nbridging night wakings of up to an hour; mid-sleep is its midpoint. Medians use "
         "weeks with 30+ infants.",
         fontsize=9, color=INK_2, ha="left", va="top", linespacing=1.45)
headline(fig, L / W, 0.18 / H, "Data: Liu et al., arXiv:1904.04780 (2019), infant_sleep_preprocessed.mat.",
         fontsize=7.5, color=MUTED, ha="left", va="bottom")
save(fig, OUT / "nyu_sleep_timing_by_age.png", DPI, SURFACE)
plt.close(fig)
print(f"fig1 saved; stability 6v12 rho {rho:.2f} n {ok.sum()}")

# ================= Figure 2: gallery of flagged jumps =================
NC, PRE, SPAN = 3, 20, 61
cell_w, cell_h, gx, gy = 2.35, 2.45, 0.42, 1.0
L2, R2, TOP2, BOTTOM2 = 0.7, 0.3, 1.95, 0.6
nr = int(np.ceil(len(jumps) / NC))
W2 = L2 + NC * cell_w + (NC - 1) * gx + R2
H2 = TOP2 + nr * cell_h + (nr - 1) * gy + BOTTOM2 + 0.35
fig = plt.figure(figsize=(W2, H2), dpi=DPI, facecolor=SURFACE)
for k, (_, e) in enumerate(jumps.iterrows()):
    r_, c_ = divmod(k, NC)
    x0 = L2 + c_ * (cell_w + gx)
    y0 = BOTTOM2 + 0.35 + (nr - 1 - r_) * (cell_h + gy)
    ax = fig.add_axes([x0 / W2, y0 / H2, cell_w / W2, cell_h / H2])
    a = max(int(e.start_day) - 1 - PRE, 0)
    b = min(a + SPAN, data.shape[0] - 1)
    x = data[a:b + 1, :, int(e.infant) - 1]
    dbl = np.concatenate([x[:-1], x[1:]], axis=1)
    ax.imshow(np.ma.masked_invalid(dbl), cmap=grey, vmin=0, vmax=1, aspect="auto",
              interpolation="nearest", extent=(0, 48, a + dbl.shape[0] + 1, a + 1))
    ax.add_patch(Rectangle((0, e.start_day - 0.5), 48, e.end_day - e.start_day + 1, fill=False,
                           edgecolor=BLUE, linewidth=1.4))
    ax.set_xticks([0, 12, 24, 36, 48])
    ax.set_xticklabels(["00", "12", "00", "12", "24"])
    style(ax)
    ax.tick_params(axis="y", colors=AXIS, labelcolor=MUTED, labelsize=7.5)
    sign = "+" if e.shift_h > 0 else "−"
    ax.annotate(f"Infant {int(e.infant)}", xy=(0, 1), xycoords="axes fraction", xytext=(0, 22),
                textcoords="offset points", fontsize=10, fontweight="semibold", color=INK, va="bottom")
    ax.annotate(f"days {int(e.start_day)}–{int(e.end_day)} · {sign}{abs(e.shift_h):.1f} h",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                fontsize=8, color=INK_2, va="bottom")
headline(fig, L2 / W2, 1 - 0.36 / H2, f"All {len(jumps)} flagged sleep-timing jumps", fontsize=20,
         fontweight="bold", color=INK, ha="left", va="top")
headline(fig, L2 / W2, 1 - 0.82 / H2,
         "Each panel: about two months of one infant's sleep, double-plotted (dark = asleep, grey = not "
         "logged).\nBlue box: days whose sleep timing sat 4+ hours away from the previous fortnight. "
         "Rows are days of age.",
         fontsize=9, color=INK_2, ha="left", va="top", linespacing=1.45)
headline(fig, L2 / W2, 0.18 / H2, "Data: Liu et al., arXiv:1904.04780 (2019), infant_sleep_preprocessed.mat.",
         fontsize=7.5, color=MUTED, ha="left", va="bottom")
save(fig, OUT / "nyu_phase_jump_gallery.png", DPI, SURFACE)
plt.close(fig)
print("fig2 saved")
