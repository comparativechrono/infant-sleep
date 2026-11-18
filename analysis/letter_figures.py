"""Figures for the Letter on sudden shifts in clock time (journal versions).

Usage: python3 -I letter_figures.py <infant_sleep_preprocessed.mat> <results_dir> <out_dir>

Reads results/jetlag (jetlag_screen.py, jetlag_profiles.py, jetlag_simulate.py) and the NYU data
file for the example actograms. Writes Figure1 and FigureS1 to <out_dir> as vector PDF and 300-dpi
PNG, in the style of the main paper's figures (journal_style.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import (  # noqa: E402
    WIDTH, INK, INK_2, NA, BLUE, ORANGE, BAND, ACTO, new_fig, ax_at, letter, actogram_axis, save_figure)

MAT, RES, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
J = RES / "jetlag"
data = loadmat(MAT)["data"]
M = pd.read_csv(J / "episode_metrics_3h.csv")
S = M[M["main_set"]].copy()
T = pd.read_csv(J / "trajectories_3h.csv")
screen = pd.read_csv(J / "screen_counts.csv")
sim = pd.read_csv(J / "simulation_runs.csv")
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#ffffff", ORANGE])
DIV.set_bad(NA)
EXAMPLES = [(594, 413), (264, 98)]          # gradual re-alignment; sustained shift ending abruptly
WIN = (-14, 35)                             # days from onset shown in the example actograms


def example(ax, infant, start_day, label_y):
    e = S[(S.infant == infant) & (S.start_day == start_day)].iloc[0]
    a = start_day - 1
    lo, hi = a + WIN[0], a + WIN[1]
    x = data[lo:hi + 2, :, infant - 1]
    dbl = np.concatenate([x[:-1], x[1:]], axis=1)
    ax.imshow(np.ma.masked_invalid(dbl), cmap=ACTO, vmin=0, vmax=1, aspect="auto",
              extent=(0, 48, WIN[1] + 1, WIN[0]))
    actogram_axis(ax)
    ax.set_ylim(WIN[1] + 1, WIN[0])
    ax.set_yticks([-14, 0, 14, 28])
    ax.set_xlabel("Time of day (h)")
    if label_y:
        ax.set_ylabel("Days from onset")
    else:
        ax.tick_params(axis="y", labelleft=False)
    end = int(e.end_day) - start_day
    ax.plot([49.2, 49.2], [0, end + 1], color=ORANGE, linewidth=2.2, solid_capstyle="butt", clip_on=False)
    return e


def figure1():
    fig = new_fig(6.2)
    H = fig.get_size_inches()[1]
    top_y, top_h = 3.25, 2.65
    aw = 2.55
    axA = ax_at(fig, 0.55, top_y, aw, top_h)
    axB = ax_at(fig, 0.55 + aw + 0.55, top_y, aw, top_h)
    eA = example(axA, *EXAMPLES[0], label_y=True)
    eB = example(axB, *EXAMPLES[1], label_y=False)
    for e, xx in ((eA, 0.02), (eB, 0.55 + aw + 0.43)):
        letter(fig, xx, H - 0.05, "A" if e is eA else "B",
               f"Infant {int(e.infant)}, day {int(e.start_day)}: "
               f"{'later' if e.direction == 'later' else 'earlier'} by {e.shift_h:.1f} h")

    # C: every episode, deviation by day from onset, sorted by duration
    S_sorted = S.sort_values(["span_days", "start_day"]).reset_index(drop=True)
    days = np.arange(-7, 22)
    grid = np.full((len(S_sorted), days.size), np.nan)
    on = T[T["align"] == "onset"]
    for r, e in S_sorted.iterrows():
        tr = on[on.episode == e.episode].set_index("day")["deviation_h"]
        grid[r] = tr.reindex(days).to_numpy()
    by, bh = 0.5, 2.25
    axC = ax_at(fig, 0.55, by, 3.35, bh)
    axC.imshow(np.ma.masked_invalid(grid), cmap=DIV, vmin=-6, vmax=6, aspect="auto",
               extent=(days[0] - 0.5, days[-1] + 0.5, len(S_sorted), 0))
    for r, e in S_sorted.iterrows():
        xe = min(e.span_days - 1 + 0.5, days[-1] + 0.5)
        axC.plot([xe, xe], [r + 0.1, r + 0.9], color=INK, linewidth=0.9)
    axC.set_xticks([-7, 0, 7, 14, 21])
    axC.set_xlabel("Days from onset")
    axC.set_yticks([0.5, 9.5, 19.5, 29.5])
    axC.set_yticklabels(["1", "10", "20", "30"])
    axC.set_ylabel("Episode (by duration)")
    cax = ax_at(fig, 0.55 + 3.35 + 0.08, by + bh - 1.1, 0.08, 1.1)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-6, 6), cmap=DIV), cax=cax, ticks=[-6, -3, 0, 3, 6])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    cb.ax.set_yticklabels(["−6", "−3", "0", "3", "6"])
    cb.set_label("Shift (h)", fontsize=7, labelpad=2)

    # D: share of the initial shift still present, median and IQR
    share = grid / S_sorted["shift_h"].to_numpy()[:, None]
    q1, q2, q3 = (np.nanpercentile(share, q, axis=0) for q in (25, 50, 75))
    axD = ax_at(fig, 4.95, by, 1.8, bh)
    axD.fill_between(days, q1, q3, color=BAND, linewidth=0)
    axD.plot(days, q2, color=BLUE, linewidth=1.2)
    axD.axhline(0, color=INK_2, linewidth=0.5, linestyle=(0, (2, 2)))
    axD.axvline(-0.5, color=INK_2, linewidth=0.5, linestyle=(0, (2, 2)))
    axD.set_xlim(-7, 21)
    axD.set_xticks([-7, 0, 7, 14, 21])
    axD.set_ylim(-0.4, 1.3)
    axD.set_yticks([0, 0.5, 1])
    axD.set_xlabel("Days from onset")
    axD.set_ylabel("Share of initial shift")
    axD.legend([Line2D([], [], color=BLUE, linewidth=1.2), Patch(facecolor=BAND, linewidth=0)],
               ["Median", "Interquartile range"], loc="upper right")
    letter(fig, 0.02, by + bh + 0.24, "C")
    letter(fig, 4.38, by + bh + 0.24, "D")
    save_figure(fig, OUT, "Figure1")


def figureS1():
    fig = new_fig(2.55)
    by, bh = 0.5, 1.75
    # A: observed episodes from day 90 against the null model, by threshold
    axA = ax_at(fig, 0.5, by, 1.75, bh)
    x = np.arange(len(screen))
    axA.bar(x, screen["episodes_from90"], width=0.6, color=BLUE, linewidth=0)
    err = np.vstack([screen["null_from90_mean"] - screen["null_from90_lo95"],
                     screen["null_from90_hi95"] - screen["null_from90_mean"]])
    axA.errorbar(x, screen["null_from90_mean"], yerr=err, fmt="o", color=INK, markersize=3,
                 elinewidth=0.8, capsize=2, capthick=0.8)
    axA.set_xticks(x)
    axA.set_xticklabels([f"{t:g}" for t in screen["threshold_h"]])
    axA.set_xlabel("Threshold (h)")
    axA.set_ylabel("Episodes from day 90")
    axA.legend([Patch(facecolor=BLUE, linewidth=0), Line2D([], [], marker="o", color=INK, linestyle="none", markersize=3)],
               ["Found", "Expected by chance"], loc="upper right")

    # B: detection of simulated shifts
    axB = ax_at(fig, 2.85, by, 1.75, bh)
    styles = [(np.inf, INK, "No re-alignment"), (10.0, BLUE, "Time constant 10 days"), (4.0, ORANGE, "Time constant 4 days")]
    sizes = sorted(sim["size_h"].unique())
    for tau, col, lab in styles:
        y = [100 * sim[(sim.size_h == s) & (sim.true_tau_d == tau)].detected.mean() for s in sizes]
        axB.plot(sizes, y, "o-", color=col, markersize=3, linewidth=1.0, label=lab)
    axB.set_xticks(sizes)
    axB.set_xlabel("Size of simulated shift (h)")
    axB.set_ylabel("Detected (%)")
    axB.set_ylim(0, 105)
    axB.legend(loc="lower right", handlelength=1.2, frameon=True, framealpha=1, edgecolor="none",
               borderpad=0.2)

    # C: time constants, simulated and observed
    axC = ax_at(fig, 5.25, by, 1.5, bh)
    det = sim[sim.detected & (sim.size_h >= 4)]
    groups = [("4 days", det[det.true_tau_d == 4.0].tau_d), ("10 days", det[det.true_tau_d == 10.0].tau_d),
              ("Observed", S["tau_d"])]
    rng = np.random.default_rng(1)
    for k, (lab, v) in enumerate(groups):
        v = np.asarray(v, dtype=float)
        fin = v[np.isfinite(v) & (v > 0)]
        jit = rng.uniform(-0.18, 0.18, fin.size)
        axC.scatter(k + jit, np.clip(fin, 1, 100), s=3 if lab != "Observed" else 7,
                    color=(ORANGE if k == 0 else BLUE) if lab != "Observed" else INK,
                    alpha=0.35 if lab != "Observed" else 0.9, linewidths=0)
        axC.plot([k - 0.28, k + 0.28], [np.median(fin)] * 2, color=INK, linewidth=1.2)
        n_inf = int(np.sum(~np.isfinite(v)))
        if n_inf:
            axC.text(k, 125, f"+{n_inf} none", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    axC.set_yscale("log")
    axC.set_ylim(1, 200)
    axC.set_yticks([1, 3, 10, 30, 100])
    axC.set_yticklabels(["1", "3", "10", "30", "100"])
    axC.set_xticks(range(3))
    axC.set_xticklabels(["Sim.\n4 days", "Sim.\n10 days", "Observed"], linespacing=1.0)
    axC.set_xlim(-0.6, 2.6)
    axC.set_ylabel("Estimated time constant (days)")
    H = fig.get_size_inches()[1]
    letter(fig, 0.02, H - 0.05, "A")
    letter(fig, 2.37, H - 0.05, "B")
    letter(fig, 4.72, H - 0.05, "C")
    save_figure(fig, OUT, "FigureS1")


if __name__ == "__main__":
    figure1()
    figureS1()
