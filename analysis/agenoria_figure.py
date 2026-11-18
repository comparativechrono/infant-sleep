"""Figure: the two Agenoria infants against the NYU cohort (external check).

Usage: python3 -I agenoria_figure.py <agenoria_out_dir> <nyu_out_dir>
Reads agenoria_emergence_summary.csv and agenoria_s24_by_day.csv (from agenoria_emergence.py)
and nyu_infant_rhythm_summary.csv; writes agenoria_vs_nyu.png to <agenoria_out_dir>.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figsave import headline, save  # noqa: E402


SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
HAIRLINE, AXIS, BAND, BARS = "#e1e0d9", "#c3c2b7", "#e8e7e1", "#c9c8bf"
BLUE, YELLOW = "#2a78d6", "#eda100"

for f in glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 9, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                     "axes.spines.top": False, "axes.spines.right": False})


def ordinal(p):
    n = int(round(p))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def main(ag_dir, nyu_dir):
    ag_dir, nyu_dir = Path(ag_dir), Path(nyu_dir)
    summ = pd.read_csv(ag_dir / "agenoria_emergence_summary.csv")
    day = pd.read_csv(ag_dir / "agenoria_s24_by_day.csv")
    nyu = pd.read_csv(nyu_dir / "nyu_infant_rhythm_summary.csv")
    t50 = nyu.loc[nyu.eligible, "t50_day"].dropna().to_numpy()
    babies = [("Baby 1", "baby1_s24", BLUE), ("Baby 2", "baby2_s24", YELLOW)]

    fig = plt.figure(figsize=(7.4, 7.6), facecolor=SURFACE)
    headline(fig, 0.08, 0.965, "External check: both Agenoria infants fall within the NYU distribution",
             fontsize=14, fontweight="bold", color=INK)
    headline(fig, 0.08, 0.935, "Same pipeline as the NYU infants: Lomb–Scargle share of variance explained by a "
             "24-h sinusoid in 14-day windows.\nHalf-plateau = first age at which that share reaches half "
             "the infant's median over days 120–240 and stays there.",
             fontsize=8.5, color=INK_2, va="top", linespacing=1.5)

    # A: 24-h strength by age
    ax = fig.add_axes([0.08, 0.47, 0.86, 0.33], facecolor=SURFACE)
    wk = day.day_of_age / 7
    ax.fill_between(wk, 100 * day.nyu_q1, 100 * day.nyu_q3, color=BAND, lw=0,
                    label="NYU middle 50% (260 infants)")
    ax.plot(wk, 100 * day.nyu_median, color=MUTED, lw=1.5, label="NYU median")
    for label, col, colour in babies:
        ax.plot(wk, 100 * day[col], color=colour, lw=2, label=label, solid_capstyle="round")
        r = summ.loc[summ.infant == label].iloc[0]
        d = int(r.t50_day)
        y = 100 * day.loc[day.day_of_age == d, col].iloc[0]
        ax.plot(d / 7, y, "o", ms=7, color=colour, mec=SURFACE, mew=2, zorder=5)
    r1 = summ.loc[summ.infant == "Baby 1"].iloc[0]
    r2 = summ.loc[summ.infant == "Baby 2"].iloc[0]
    y1 = 100 * day.loc[day.day_of_age == int(r1.t50_day), "baby1_s24"].iloc[0]
    y2 = 100 * day.loc[day.day_of_age == int(r2.t50_day), "baby2_s24"].iloc[0]
    ax.annotate(f"Baby 1 reaches half-plateau\nat day {int(r1.t50_day)}", (r1.t50_day / 7, y1),
                xytext=(0.8, 37), color=INK, fontsize=8.5, va="center",
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.annotate(f"Baby 2 at day {int(r2.t50_day)}", (r2.t50_day / 7, y2), xytext=(17.5, 6),
                color=INK, fontsize=8.5, va="center",
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.set_xlim(0, 53)
    ax.set_ylim(0, 60)
    ax.set_xticks(range(0, 53, 4))
    ax.set_ylabel("Variance explained at 24 h (%)")
    ax.set_xlabel("Age (weeks)")
    ax.yaxis.grid(True, color=HAIRLINE, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2, labelcolor=INK)
    ax.set_title("How strong the 24-hour sleep rhythm is, by age", loc="left", fontsize=11,
                 fontweight="bold", color=INK, pad=10)

    # B: where each infant's half-plateau age falls among the NYU infants
    bx = fig.add_axes([0.08, 0.10, 0.86, 0.25], facecolor=SURFACE)
    edges = np.arange(0, 26 * 7 + 1, 7)
    counts, _ = np.histogram(t50, bins=edges)
    bx.bar(edges[:-1] / 7 + 0.5, counts, width=0.86, color=BARS, lw=0)
    med = float(np.median(t50))
    bx.axvline(med / 7, color=INK, lw=1.2)
    bx.text(med / 7 + 0.25, 38, f"NYU median, day {med:g}", color=INK, fontsize=8.5, va="center")
    for (label, _, colour), y_lab in zip(babies, (46, 30)):
        r = summ.loc[summ.infant == label].iloc[0]
        x = r.t50_day / 7
        bx.plot([x, x], [0, y_lab - 3], color=colour, lw=2.2, solid_capstyle="butt", zorder=4)
        bx.plot(x, y_lab - 3, "v", ms=7, color=colour, zorder=5)
        ha, dx = ("right", -0.3) if label == "Baby 1" else ("left", 0.3)
        bx.text(x + dx, y_lab, f"{label}: day {int(r.t50_day)}\n({ordinal(r.t50_nyu_percentile)} percentile)",
                color=INK, fontsize=8.5, ha=ha, va="center", linespacing=1.3)
    bx.set_xlim(0, 26)
    bx.set_ylim(0, 52)
    bx.set_xticks(range(0, 27, 2))
    bx.set_xlabel("Age at half-plateau (weeks)")
    bx.set_ylabel("NYU infants")
    bx.yaxis.grid(True, color=HAIRLINE, lw=0.8)
    bx.set_axisbelow(True)
    bx.set_title("Age at half-plateau: the NYU distribution and the two Agenoria infants", loc="left",
                 fontsize=11, fontweight="bold", color=INK, pad=10)

    headline(fig, 0.08, 0.012, "Data: Liu et al., arXiv:1904.04780 (2019); Agenoria, Jiuguang Wang "
             "(github.com/jiuguangw/Agenoria), CC BY-NC-SA 4.0.", fontsize=7.5, color=MUTED)
    print("wrote", save(fig, ag_dir / "agenoria_vs_nyu.png", 220, SURFACE))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
