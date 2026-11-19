"""Figures comparing infants and adults for the Letter (journal versions).

Usage: python3 -I adult_figures.py <results_dir> <out_dir>

Reads results/adult (adult_data.py, adult_screen.py) and results/jetlag (the infants). Writes
Figure2 and FigureS2 to <out_dir> as vector PDF and 300-dpi PNG (journal_style.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import (  # noqa: E402
    INK, INK_2, BLUE, ORANGE, BAND, ACTO, NA, new_fig, ax_at, letter, actogram_axis, save_figure)

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
A, J = RES / "adult", RES / "jetlag"
WIN = (-14, 28)
GREY = "#9a9a9a"


def figure2():
    fig = new_fig(2.75)
    H = fig.get_size_inches()[1]
    by, bh = 0.47, 1.95

    # A: the adult's sleep in local clock time across a trip of +8 h and -9 h
    z = np.load(A / "adult_records.npz")
    trips = pd.read_csv(A / "trips.csv")
    arr = trips[trips.leg == "arrival"].sort_values("clock_change_h", key=abs).iloc[-1]   # the longest flight
    k = int(arr.adult[1:]) - 1                                # labels A01.. follow the order of the records
    x = z[f"x_{k}"].astype(float)
    n0 = int(arr.night_index)
    stay = int(arr.nights_on_new_clock)
    lo, hi = n0 + WIN[0], n0 + WIN[1]
    # calendar day D = 00:00-12:00 from night D-1 and 12:00-24:00 from night D
    day = np.concatenate([x[lo - 1:hi + 1, 72:], x[lo:hi + 2, :72]], axis=1)
    dbl = np.concatenate([day[:-1], day[1:]], axis=1)
    axA = ax_at(fig, 0.5, by, 2.05, bh)
    axA.imshow(np.ma.masked_invalid(dbl), cmap=ACTO, vmin=0, vmax=1, aspect="auto",
               extent=(0, 48, WIN[1] + 1, WIN[0]))
    actogram_axis(axA)
    axA.set_ylim(WIN[1] + 1, WIN[0])
    axA.set_yticks([-14, 0, 14, 28])
    axA.set_xlabel("Local time (h)")
    axA.set_ylabel("Days from arrival")
    axA.plot([49.2, 49.2], [0.5, stay + 0.5], color=ORANGE, linewidth=2.2, solid_capstyle="butt", clip_on=False)
    axA.text(50.3, 0.6, f"+{arr.clock_change_h:g} h", fontsize=6.5, va="center", ha="left", color=INK, clip_on=False)
    ret = trips[(trips.adult == arr.adult) & (trips.leg == "return") & (trips.night_index == n0 + stay)].iloc[0]
    axA.text(50.3, stay + 0.6, f"−{abs(ret.clock_change_h):g} h", fontsize=6.5, va="center", ha="left",
             color=INK, clip_on=False)
    letter(fig, 0.02, H - 0.05, "A", "Adult, time zone recorded")

    # B: lag behind the new clock while it lasted: infants' episodes and the adult's clock changes
    axB = ax_at(fig, 3.2, by, 1.85, bh)
    days = np.arange(0, 14)
    M = pd.read_csv(J / "episode_metrics_3h.csv")
    S = M[M["main_set"]]
    T = pd.read_csv(J / "trajectories_3h.csv")
    on = T[T["align"] == "onset"]
    grid = np.full((len(S), days.size), np.nan)
    for r, (_, e) in enumerate(S.iterrows()):
        tr = on[on.episode == e.episode].set_index("day")["deviation_h"].reindex(days).to_numpy(copy=True)
        tr[days >= e.span_days] = np.nan                       # only while the shift lasted
        grid[r] = tr
    n_on = (~np.isnan(grid)).sum(0)
    keep = n_on >= 10                                          # days on which >= 10 episodes still ran
    q1, q2, q3 = (np.nanpercentile(grid[:, keep], q, axis=0) for q in (25, 50, 75))
    axB.fill_between(days[keep], q1, q3, color=BAND, linewidth=0)
    axB.plot(days[keep], q2, color=BLUE, linewidth=1.3)
    tt = pd.read_csv(A / "trip_trajectories.csv")
    tt = tt[(tt.night >= 0) & (tt.night <= 13) & tt.on_new_clock]
    for c, g in tt.groupby("change"):
        axB.plot(g.night, g.lag_h, color=GREY, linewidth=0.6, alpha=0.8)
    med = tt.groupby("night")["lag_h"].median()
    axB.plot(med.index, med.values, color=INK, linewidth=1.3)
    axB.axhline(0, color=INK_2, linewidth=0.5, linestyle=(0, (2, 2)))
    axB.set_xlim(-0.3, 13.3)
    axB.set_xticks([0, 7, 13])
    axB.set_ylim(-2.2, 8.6)
    axB.set_yticks([-2, 0, 2, 4, 6, 8])
    axB.set_xlabel("Days on the new clock")
    axB.set_ylabel("Lag behind the new clock (h)")
    axB.legend([Line2D([], [], color=BLUE, linewidth=1.3), Line2D([], [], color=INK, linewidth=1.3),
                Line2D([], [], color=GREY, linewidth=0.6)],
               [f"Infants: {len(S)} episodes", f"Adult: {tt.change.nunique()} clock changes",
                "Adult: each change"], loc="upper right", handlelength=1.4)
    letter(fig, 2.62, H - 0.05, "B")

    # C: night-to-night scatter of sleep timing around each person's own trend
    D = pd.read_csv(A / "day_to_day_scatter.csv")
    axC = ax_at(fig, 5.65, by, 1.1, bh)
    rng = np.random.default_rng(3)
    for j, (grp, col) in enumerate((("infant", BLUE), ("adult", INK))):
        v = D[D.group == grp]["sd_h"].to_numpy()
        axC.scatter(j + rng.uniform(-0.22, 0.22, v.size), v, s=2.5 if grp == "infant" else 6, color=col,
                    alpha=0.25 if grp == "infant" else 0.8, linewidths=0)
        q = np.percentile(v, [25, 50, 75])
        axC.plot([j - 0.3, j + 0.3], [q[1]] * 2, color=INK, linewidth=1.3)
        axC.plot([j, j], [q[0], q[2]], color=INK, linewidth=0.8)
    axC.set_xticks([0, 1])
    axC.set_xticklabels([f"Infants\n(n = {int((D.group == 'infant').sum())})",
                         f"Adults\n(n = {int((D.group == 'adult').sum())})"], linespacing=1.0)
    axC.set_xlim(-0.6, 1.6)
    axC.set_ylim(0, 3.2)
    axC.set_ylabel("Night-to-night SD of timing (h)")
    letter(fig, 5.08, H - 0.05, "C")
    save_figure(fig, OUT, "Figure2")


def figureS2():
    fig = new_fig(2.4)
    H = fig.get_size_inches()[1]
    by, bh = 0.5, 1.65
    sc = pd.read_csv(A / "screen_counts.csv")
    axA = ax_at(fig, 0.55, by, 2.0, bh)
    x = np.arange(len(sc))
    axA.bar(x, sc["episodes"], width=0.6, color=INK_2, linewidth=0)
    err = np.vstack([sc["null_mean"] - sc["null_lo95"], sc["null_hi95"] - sc["null_mean"]])
    axA.errorbar(x, sc["null_mean"], yerr=err, fmt="o", color=INK, markersize=3, elinewidth=0.8, capsize=2,
                 capthick=0.8)
    axA.set_xticks(x)
    axA.set_xticklabels([f"{t:g}" for t in sc["threshold_h"]])
    axA.set_xlabel("Threshold (h)")
    axA.set_ylabel("Episodes in adults")
    axA.legend([Patch(facecolor=INK_2, linewidth=0), Line2D([], [], marker="o", color=INK, linestyle="none",
                                                           markersize=3)],
               ["Found", "Expected by chance"], loc="upper right")
    letter(fig, 0.02, H - 0.05, "A")

    axB = ax_at(fig, 3.25, by, 1.95, bh)
    sim_a = pd.read_csv(A / "simulation_runs.csv")
    sim_i = pd.read_csv(J / "simulation_runs.csv")
    sizes = sorted(sim_a["size_h"].unique())
    styles = [(np.inf, INK, "No re-alignment"), (10.0, BLUE, "Time constant 10 days"),
              (4.0, ORANGE, "Time constant 4 days")]
    for tau, col, lab in styles:
        ya = [100 * sim_a[(sim_a.size_h == s) & (sim_a.true_tau_d == tau)].detected.mean() for s in sizes]
        yi = [100 * sim_i[(sim_i.size_h == s) & (sim_i.true_tau_d == tau)].detected.mean() for s in sizes]
        axB.plot(sizes, ya, "o-", color=col, markersize=3, linewidth=1.0, label=lab)
        axB.plot(sizes, yi, "--", color=col, linewidth=0.7, alpha=0.8)
    axB.set_xticks(sizes)
    axB.set_xlabel("Size of simulated shift (h)")
    axB.set_ylabel("Detected in adult records (%)")
    axB.set_ylim(0, 105)
    h, l = axB.get_legend_handles_labels()
    h.append(Line2D([], [], color=INK_2, linestyle="--", linewidth=0.7))
    l.append("Infant records (Figure S1B)")
    axB.legend(h, l, loc="center left", bbox_to_anchor=(1.03, 0.5), handlelength=1.4, borderaxespad=0)
    letter(fig, 2.67, H - 0.05, "B")
    save_figure(fig, OUT, "FigureS2")


if __name__ == "__main__":
    figure2()
    figureS2()
