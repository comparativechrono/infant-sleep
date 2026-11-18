"""Journal versions of the figures in the paper (Journal of Biological Rhythms).

Usage: python3 -I paper_figures.py <infant_sleep_preprocessed.mat> <results_dir> <out_dir>

Reads the result files that the analysis scripts write to <results_dir>/nyu and
<results_dir>/agenoria (run them first; see run_all.sh) and, for the population heatmap and the
NYU actograms, the NYU data file. Writes Figures 1-6 and S1-S3 to <out_dir>, each as a vector PDF
with embedded fonts and a 300-dpi PNG.

Every figure is drawn at its printed size, 174 mm wide (the two-column width), so the type sizes
below are the printed sizes: Liberation Sans (metrically identical to Arial), 7-8 pt, with 10-pt
bold panel letters. Figures carry no titles or explanatory text; the legends in the paper do.
Colours come from the Okabe-Ito set, which stays distinguishable with colour-vision deficiency.
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.io import loadmat

MAT, RES, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
NYU, AGEN = RES / "nyu", RES / "agenoria"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_style import (  # noqa: E402
    MM, WIDTH, INK, INK_2, NA, BLUE, ORANGE, BAND, NYU_LINE, NYU_BAND, NYU_BARS, RAMP, SEQ, ACTO, T50, MONTH,
    new_fig, ax_at, letter, clock, noon_ref, signed, actogram_axis, save_figure)


def save(fig, name):
    save_figure(fig, OUT, name)


# ----------------------------------------------------------------------------- data
data = loadmat(MAT)["data"]                       # (730 days, 144 ten-minute bins, 700 infants)
N_DAYS = data.shape[0]
s24 = np.load(NYU / "nyu_rhythm_windows.npz")["s24"].astype(float)    # (infants, day of age)
rsum = pd.read_csv(NYU / "nyu_infant_rhythm_summary.csv")
tsum = pd.read_csv(NYU / "nyu_infant_timing_summary.csv")
weekly = pd.read_csv(NYU / "nyu_sleep_timing_weekly.csv")
jumps = pd.read_csv(NYU / "nyu_phase_jumps.csv").sort_values(["infant", "start_day"])
evt = pd.read_csv(NYU / "nyu_emergence_vs_timing.csv")
el = rsum[rsum["eligible"]].copy()
t50 = el["t50_day"].to_numpy(float)


# ----------------------------------------------------------------------------- Figure 1
def figure1():
    logged = ~np.isnan(data).all(1)                          # (days, infants)
    n_day = logged.sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        share = np.nanmean(data, axis=2) * 100               # % of logged infants asleep
    share[n_day < 20] = np.nan

    fig = new_fig(5.55)
    L, B, AW, AH, G, BW = 0.50, 0.98, 4.62, 4.30, 0.16, 1.40
    axA = ax_at(fig, L, B, AW, AH)
    axB = ax_at(fig, L + AW + G, B, BW, AH, sharey=axA)
    axA.imshow(np.ma.masked_invalid(share), cmap=SEQ, vmin=0, vmax=100, aspect="auto",
               extent=(0, 24, N_DAYS, 0))
    axA.set_xlim(0, 24)
    axA.set_xticks(range(0, 25, 6))
    axA.set_xticks(range(0, 25, 2), minor=True)
    axA.set_xlabel("Time of day (h)")
    axA.set_yticks([m * MONTH for m in range(0, 25, 3)])
    axA.set_yticklabels([str(m) for m in range(0, 25, 3)])
    axA.set_ylim(24 * MONTH, 0)
    axA.set_ylabel("Age (months)")

    axB.plot(n_day, np.arange(N_DAYS) + 0.5, color=INK, linewidth=0.8)
    axB.axvline(20, color=INK_2, linewidth=0.5, linestyle=(0, (2, 2)))
    axB.set_xlim(0, 650)
    axB.set_xticks([0, 200, 400, 600])
    axB.set_xlabel("Infants logged")
    axB.tick_params(axis="y", labelleft=False)

    cax = ax_at(fig, L, 0.30, 1.6, 0.07)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 100), cmap=SEQ), cax=cax,
                      orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    cax.set_title("Infants asleep (%)", fontsize=7, loc="left", pad=3)
    H = fig.get_size_inches()[1]
    letter(fig, 0.02, H - 0.05, "A")
    letter(fig, L + AW + G - 0.12, H - 0.05, "B")
    save(fig, "Figure1")


# ----------------------------------------------------------------------------- Figure 2
def figure2():
    DAYS = 180
    cov = np.array([np.mean(~np.isnan(data[:DAYS, :, i]).all(1)) for i in range(data.shape[2])])
    good = el[cov[el["infant"] - 1] >= 0.9]
    picks = []
    for q in (0.1, 0.5, 0.9):                                 # early, typical and late infants
        row = good.iloc[int(np.argmin(np.abs(good["t50_day"] - np.quantile(t50, q))))]
        picks.append((int(row["infant"]), float(row["t50_day"]), float(row["plateau_s24"])))
    assert [p[0] for p in picks] == [195, 9, 321], picks

    fig = new_fig(4.05)
    L, B, AH = 0.48, 0.52, 3.25
    AW, SW, G1, G2 = 1.30, 0.42, 0.14, 0.30
    H = fig.get_size_inches()[1]
    for k, (inf, t50_i, plat) in enumerate(picks):
        x0 = L + k * (AW + G1 + SW + G2)
        ax = ax_at(fig, x0, B, AW, AH)
        axs = ax_at(fig, x0 + AW + G1, B, SW, AH, sharey=ax)
        x = data[:DAYS + 1, :, inf - 1]
        dbl = np.concatenate([x[:DAYS], x[1:DAYS + 1]], axis=1)
        ax.imshow(np.ma.masked_invalid(dbl), cmap=ACTO, vmin=0, vmax=1, aspect="auto",
                  extent=(0, 48, DAYS, 0))
        actogram_axis(ax)
        ax.set_ylim(DAYS, 0)
        ax.set_yticks(range(0, DAYS + 1, 30))
        ax.set_xlabel("Time of day (h)")
        if k == 0:
            ax.set_ylabel("Age (days)")
        else:
            ax.tick_params(axis="y", labelleft=False)
        axs.plot(s24[inf - 1, :DAYS] * 100, np.arange(DAYS) + 0.5, color=BLUE, linewidth=0.9)
        axs.axvline(plat * 50, color=INK_2, linewidth=0.5, linestyle=(0, (2, 1.5)))
        axs.set_xlim(0, 80)
        axs.set_xticks([0, 40, 80])
        axs.set_xlabel("24-h\nstrength (%)", linespacing=1.0)
        axs.tick_params(axis="y", left=False, labelleft=False)
        axs.spines["left"].set_visible(False)
        for a in (ax, axs):
            a.axhline(t50_i - 0.5, color=ORANGE, linewidth=0.9)
        letter(fig, x0 - (0.40 if k == 0 else 0.05), H - 0.05, "ABC"[k],
               f"Infant {inf} ({T50} = day {t50_i:.0f})")
    save(fig, "Figure2")


# ----------------------------------------------------------------------------- Figure 3
def figure3():
    MAX_AGE = 240
    age = np.arange(1, s24.shape[1] + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(s24, axis=0)
        q1, q3 = np.nanquantile(s24, 0.25, axis=0), np.nanquantile(s24, 0.75, axis=0)
    ok = np.sum(~np.isnan(s24), axis=0) >= 50
    level6 = med[181]                                          # cohort median at day 182
    cross = int(np.argmax((med >= level6 / 2) & ok)) + 1
    assert cross == 65, cross
    order = el.sort_values(["t50_day", "infant"])["infant"].to_numpy() - 1
    rel = s24[order, :MAX_AGE] / el.set_index("infant").loc[order + 1, "plateau_s24"].to_numpy()[:, None]

    fig = new_fig(6.15)
    L, R, B = 0.55, 0.72, 0.42
    PW = WIDTH - L - R
    HA, HB, HC, GAP = 1.45, 1.05, 2.55, 0.24
    yC, yB = B, B + HC + GAP
    yA = yB + HB + GAP
    axA = ax_at(fig, L, yA, PW, HA)
    axB = ax_at(fig, L, yB, PW, HB, sharex=axA)
    axC = ax_at(fig, L, yC, PW, HC, sharex=axA)

    sel = ok & (age <= MAX_AGE)
    axA.fill_between(age[sel], q1[sel] * 100, q3[sel] * 100, color=BAND, linewidth=0)
    axA.plot(age[sel], med[sel] * 100, color=BLUE, linewidth=1.2)
    axA.plot([0, cross], [level6 * 50] * 2, color=INK_2, linewidth=0.5, linestyle=(0, (2, 2)))
    axA.plot([cross], [med[cross - 1] * 100], "o", color=INK, markersize=3.2, zorder=5)
    axA.set_ylim(0, 55)
    axA.set_yticks([0, 25, 50])
    axA.set_ylabel("24-h strength (%)")
    axA.legend([Line2D([], [], color=BLUE, linewidth=1.2), Patch(facecolor=BAND, linewidth=0)],
               ["Median", "Interquartile range"], loc="upper left")
    axA.tick_params(axis="x", labelbottom=False)

    bins = np.arange(0, MAX_AGE + 7, 7)
    counts, edges = np.histogram(t50, bins=bins)
    axB.bar(edges[:-1] + 0.6, counts, width=5.8, align="edge", color=BLUE, linewidth=0)
    axB.axvline(np.median(t50), color=INK, linewidth=0.8)
    axB.set_ylabel("Infants")
    axB.set_yticks([0, 10, 20, 30])
    axB.tick_params(axis="x", labelbottom=False)

    im = axC.imshow(np.ma.masked_invalid(rel), cmap=SEQ, vmin=0, vmax=1, aspect="auto",
                    extent=(0, MAX_AGE, len(order), 0))
    axC.set_xlim(0, MAX_AGE)
    axC.set_xticks(range(0, MAX_AGE + 1, 30))
    axC.set_xlabel("Age (days)")
    axC.set_yticks([0.5, 99.5, 199.5, len(order) - 0.5])
    axC.set_yticklabels(["1", "100", "200", str(len(order))])
    axC.set_ylabel(f"Infant (ranked by {T50})")
    cax = ax_at(fig, L + PW + 0.1, yC + HC - 1.1, 0.08, 1.1)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0", "0.5", "≥1"])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    cb.set_label("Relative to plateau", fontsize=7, labelpad=3)

    for y0, h, s in ((yA, HA, "A"), (yB, HB, "B"), (yC, HC, "C")):
        letter(fig, 0.02, y0 + h + 0.17, s)
    save(fig, "Figure3")


# ----------------------------------------------------------------------------- Figure 4
def figure4():
    wk = weekly.copy()
    for c in ("bedtime_h", "wake_time_h", "mid_sleep_h"):
        wk[c] = noon_ref(wk[c])
    g = wk.groupby("week_of_age")
    n = g.size()
    keep = n[n >= 30].index
    age = (keep.to_numpy() - 0.5) * 7 / MONTH                      # week centre, months

    fig = new_fig(2.7)
    B, H_ = 0.45, 2.0
    axA = ax_at(fig, 0.55, B, 3.55, H_)
    axB = ax_at(fig, 4.72, B, H_, H_)
    labels = {"wake_time_h": "Wake time", "mid_sleep_h": "Mid-sleep", "bedtime_h": "Bedtime"}
    lo_all, hi_all = [], []
    for col, lab in labels.items():
        q = g[col].quantile([0.25, 0.5, 0.75]).unstack().loc[keep]
        axA.fill_between(age, q[0.25], q[0.75], color=BAND, linewidth=0)
        axA.plot(age, q[0.5], color=BLUE, linewidth=1.1)
        lo_all.append(q[0.25].min())
        hi_all.append(q[0.75].max())
        tail = q[age >= 20.5]
        axA.text(23.8, tail[0.75].max() + 0.25, lab, ha="right", va="bottom", fontsize=7)
    lo, hi = np.floor(min(lo_all)), np.ceil(max(hi_all)) + 0.5
    yt = np.arange(np.ceil(lo / 2) * 2, hi + 0.01, 2)
    axA.set_ylim(lo, hi)
    axA.set_yticks(yt)
    axA.set_yticklabels([clock(h) for h in yt])
    axA.set_xlim(0, 24)
    axA.set_xticks(range(0, 25, 6))
    axA.set_xticks(range(0, 25, 3), minor=True)
    axA.set_xlabel("Age (months)")
    axA.set_ylabel("Clock time")

    x, y = noon_ref(tsum["mid_sleep_6mo"]), noon_ref(tsum["mid_sleep_12mo"])
    m = ~np.isnan(x) & ~np.isnan(y)
    rho = stats.spearmanr(x[m], y[m])[0]
    assert (round(rho, 2), int(m.sum())) == (0.76, 454), (rho, m.sum())
    lo2, hi2 = 22.5, 29.0
    axB.plot([lo2, hi2], [lo2, hi2], color=INK_2, linewidth=0.5, linestyle=(0, (2, 2)), zorder=1)
    axB.scatter(x[m], y[m], s=5, color=BLUE, alpha=0.5, linewidths=0, zorder=2)
    ticks = np.arange(24, 29, 2)
    for a in (axB.xaxis, axB.yaxis):
        a.set_ticks(ticks)
        a.set_ticklabels([clock(t) for t in ticks])
        a.set_ticks(np.arange(23, 29, 2), minor=True)
    axB.set_xlim(lo2, hi2)
    axB.set_ylim(lo2, hi2)
    axB.set_xlabel("Mid-sleep, 6 months")
    axB.set_ylabel("Mid-sleep, 12 months")
    axB.text(lo2 + 0.2, hi2 - 0.2, f"ρ = {rho:.2f}\nn = {m.sum()}", ha="left", va="top")
    Hf = fig.get_size_inches()[1]
    letter(fig, 0.02, Hf - 0.05, "A")
    letter(fig, 4.13, Hf - 0.05, "B")
    save(fig, "Figure4")


# ----------------------------------------------------------------------------- Figure 5
def figure5():
    df = rsum.merge(tsum, on="infant")
    df = df[df["eligible"]]
    outcomes = [("bedtime", "Bedtime, 12 months", True), ("wake_time", "Wake time, 12 months", True),
                ("mid_sleep", "Mid-sleep, 12 months", True),
                ("night_period", "Night length, 12 months (h)", False)]
    prim = evt[(evt["predictor"] == "t50_day") & (evt["age"] == "12mo")].set_index("outcome")

    fig = new_fig(4.45)
    L, GX, R, B, GY, TOP = 0.62, 0.78, 0.10, 0.42, 0.36, 0.24
    PW = (WIDTH - L - GX - R) / 2
    PH = (fig.get_size_inches()[1] - B - GY - TOP) / 2
    for k, (key, ylab, is_clock) in enumerate(outcomes):
        r_, c_ = divmod(k, 2)
        x0, y0 = L + c_ * (PW + GX), B + (1 - r_) * (PH + GY)
        ax = ax_at(fig, x0, y0, PW, PH)
        x = df["t50_day"].to_numpy(float)
        y = df[f"{key}_12mo"].to_numpy(float)
        y = noon_ref(y) if is_clock else y
        m = ~np.isnan(x) & ~np.isnan(y)
        assert m.sum() == prim.loc[key, "n"] == 187
        ax.scatter(x[m], y[m], s=5, color=BLUE, alpha=0.55, linewidths=0, zorder=2)
        slope, icpt, _, _ = stats.theilslopes(y[m], x[m])
        xx = np.array([x[m].min(), x[m].max()])
        ax.plot(xx, icpt + slope * xx, color=INK, linewidth=1.0, zorder=3)
        lo, hi = np.nanquantile(y[m], [0.005, 0.995])
        pad = (hi - lo) * 0.12
        ax.set_ylim(lo - pad, hi + pad)
        if is_clock:
            ticks = np.arange(np.ceil(lo - pad), np.floor(hi + pad) + 1, 1)
            if len(ticks) > 6:
                ticks = ticks[::2] if ticks[0] % 2 == 0 else ticks[1::2]
            ax.set_yticks(ticks)
            ax.set_yticklabels([clock(t) for t in ticks])
        else:
            ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.set_xlim(0, 180)
        ax.set_xticks(range(0, 181, 30))
        if r_ == 1:
            ax.set_xlabel(f"Half-plateau age, {T50} (days)")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylabel(ylab)
        rho = prim.loc[key, "spearman_rho"]
        ax.text(0.98, 0.97, f"ρ = {signed(rho) if abs(rho) >= 0.005 else '0.00'}",
                transform=ax.transAxes, ha="right", va="top")
        letter(fig, x0 - 0.60, y0 + PH + 0.22, "ABCD"[k])
    save(fig, "Figure5")


# ----------------------------------------------------------------------------- Figure 6
def figure6():
    asum = pd.read_csv(AGEN / "agenoria_emergence_summary.csv").set_index("infant")
    day = pd.read_csv(AGEN / "agenoria_s24_by_day.csv")
    babies = [("Baby 1", "baby1_s24", BLUE), ("Baby 2", "baby2_s24", ORANGE)]

    L, R, B = 0.55, 0.12, 0.42
    PW = WIDTH - L - R
    HB, HA, GAP = 1.30, 1.90, 0.55
    fig = new_fig(B + HB + GAP + HA + 0.24)
    axA = ax_at(fig, L, B + HB + GAP, PW, HA)
    axB = ax_at(fig, L, B, PW, HB)

    axA.fill_between(day["day_of_age"], 100 * day["nyu_q1"], 100 * day["nyu_q3"], color=NYU_BAND,
                     linewidth=0)
    axA.plot(day["day_of_age"], 100 * day["nyu_median"], color=NYU_LINE, linewidth=1.0)
    for lab, col, colour in babies:
        axA.plot(day["day_of_age"], 100 * day[col], color=colour, linewidth=1.0)
        d = int(asum.loc[lab, "t50_day"])
        yv = 100 * day.loc[day["day_of_age"] == d, col].iloc[0]
        axA.plot(d, yv, "o", markersize=4, color=colour, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=5)
    axA.set_xlim(0, 365)
    axA.set_xticks(range(0, 361, 60))
    axA.set_xticks(range(0, 366, 30), minor=True)
    axA.set_ylim(0, 60)
    axA.set_yticks([0, 20, 40, 60])
    axA.set_xlabel("Age (days)")
    axA.set_ylabel("24-h strength (%)")
    handles = [Line2D([], [], color=NYU_LINE), Patch(facecolor=NYU_BAND, linewidth=0),
               Line2D([], [], color=BLUE), Line2D([], [], color=ORANGE)]
    axA.legend(handles, ["NYU median", "NYU interquartile range", "Baby 1", "Baby 2"],
               loc="upper left", ncol=2, columnspacing=1.5)

    edges = np.arange(0, 26 * 7 + 1, 7)
    counts, _ = np.histogram(t50, bins=edges)
    axB.bar(edges[:-1] + 0.6, counts, width=5.8, align="edge", color=NYU_BARS, linewidth=0)
    axB.axvline(np.median(t50), color=INK, linewidth=0.8)
    for (lab, _, colour), ytop in zip(babies, (44, 44)):
        d = int(asum.loc[lab, "t50_day"])
        axB.plot([d, d], [0, ytop - 4], color=colour, linewidth=1.4, zorder=4)
        axB.plot(d, ytop - 4, "v", markersize=4.5, color=colour, zorder=5)
        axB.text(d, ytop - 1, lab, color=INK, ha="center", va="bottom", fontsize=7)
    axB.set_xlim(0, 182)
    axB.set_xticks(range(0, 181, 30))
    axB.set_ylim(0, 52)
    axB.set_yticks([0, 20, 40])
    axB.set_xlabel(f"Half-plateau age, {T50} (days)")
    axB.set_ylabel("NYU infants")
    letter(fig, 0.02, B + HB + GAP + HA + 0.17, "A")
    letter(fig, 0.02, B + HB + 0.17, "B")
    save(fig, "Figure6")


# ----------------------------------------------------------------------------- Figure S1
def figureS1():
    NC, PRE, SPAN = 3, 20, 61
    nr = int(np.ceil(len(jumps) / NC))
    fig = new_fig(8.55)
    L, R, B, TOP, GX, GY = 0.50, 0.08, 0.40, 0.24, 0.34, 0.42
    CW = (WIDTH - L - R - (NC - 1) * GX) / NC
    CH = (fig.get_size_inches()[1] - B - TOP - (nr - 1) * GY) / nr
    last_in_col = {c: max(k for k in range(len(jumps)) if k % NC == c) for c in range(NC)}
    for k, (_, e) in enumerate(jumps.iterrows()):
        r_, c_ = divmod(k, NC)
        x0, y0 = L + c_ * (CW + GX), B + (nr - 1 - r_) * (CH + GY)
        ax = ax_at(fig, x0, y0, CW, CH)
        a = max(int(e.start_day) - 1 - PRE, 0)
        b = min(a + SPAN, data.shape[0] - 1)
        x = data[a:b + 1, :, int(e.infant) - 1]
        dbl = np.concatenate([x[:-1], x[1:]], axis=1)
        ax.imshow(np.ma.masked_invalid(dbl), cmap=ACTO, vmin=0, vmax=1, aspect="auto",
                  extent=(0, 48, a + dbl.shape[0] + 1, a + 1))
        ax.add_patch(Rectangle((0.15, e.start_day - 0.5), 47.7, e.end_day - e.start_day + 1,
                               fill=False, edgecolor=ORANGE, linewidth=1.0, zorder=5))
        actogram_axis(ax)
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        if c_ == 0:
            ax.set_ylabel("Age (days)")
        if k == last_in_col[c_]:
            ax.set_xlabel("Time of day (h)")
        sign = "+" if e.shift_h > 0 else "−"
        letter(fig, x0 - (0.40 if c_ == 0 else 0.26), y0 + CH + 0.20, chr(65 + k),
               f"Infant {int(e.infant)}, {sign}{abs(e.shift_h):.1f} h")
    save(fig, "FigureS1")


# ----------------------------------------------------------------------------- Figures S2-S3
def figure_agenoria():
    z = np.load(AGEN / "agenoria_actograms.npz")
    periods = z["periods_h"]
    lx = np.log10(periods)
    half = (lx[1] - lx[0]) / 2
    vmin = 0.005                                    # about the noise level of a 14-day window
    vmax = float(np.ceil(max(np.nanmax(z["baby1_heat"]), np.nanmax(z["baby2_heat"])) * 10) / 10)
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    lut = np.stack([to_rgb(c) for c in (NA, "#ffffff", "#000000")])    # no record, awake, asleep
    for tag, name in (("baby1", "FigureS2"), ("baby2", "FigureS3")):
        act, heat = z[f"{tag}_act"], z[f"{tag}_heat"]
        daycare = int(z[f"{tag}_daycare_day"])
        n = act.shape[0]
        fig = new_fig(8.0)
        L, B, AW, G, PWd, PH = 0.50, 0.95, 4.05, 0.26, 1.90, 6.55
        ax = ax_at(fig, L, B, AW, PH)
        ax2 = ax_at(fig, L + AW + G, B, PWd, PH, sharey=ax)
        ax.imshow(lut[act], aspect="auto", extent=(0, 48, n, 0))
        actogram_axis(ax)
        ax.set_ylim(n, 0)
        ax.set_yticks(range(0, n + 1, 30))
        ax.set_xlabel("Time of day (h)")
        ax.set_ylabel("Age (days)")
        ax.plot([48.9], [daycare + 0.5], marker="<", markersize=4.5, color=ORANGE,
                clip_on=False, zorder=5)
        im = ax2.imshow(np.ma.masked_invalid(np.clip(heat, vmin, None)), cmap=SEQ, norm=norm,
                        aspect="auto", extent=(lx[0] - half, lx[-1] + half, n, 0))
        ticks = [2, 3, 4, 6, 8, 12, 24]
        ax2.set_xticks(np.log10(ticks))
        ax2.set_xticklabels([str(t) for t in ticks])
        ax2.set_xticks([], minor=True)
        ax2.set_xlabel("Period (h)")
        ax2.tick_params(axis="y", labelleft=False)
        cax = ax_at(fig, L + AW + G, 0.30, PWd, 0.07)
        cticks = [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if t <= vmax]
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=cticks)
        cb.ax.set_xticklabels([f"{t * 100:g}" for t in cticks])
        cb.ax.minorticks_off()
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(width=0.5, length=2)
        cax.set_title("Variance explained (%)", fontsize=7, loc="left", pad=3)
        handles = [Patch(facecolor="#000000", linewidth=0),
                   Patch(facecolor="#ffffff", edgecolor=INK, linewidth=0.5),
                   Patch(facecolor=NA, linewidth=0),
                   Line2D([], [], marker="<", color=ORANGE, linestyle="none", markersize=4.5)]
        Hf = fig.get_size_inches()[1]
        fig.legend(handles, ["Asleep", "Awake", "Not logged, or window incomplete",
                             "First day at daycare"], loc="upper left", ncol=4,
                   bbox_to_anchor=(L / WIDTH - 0.01, 1 - 0.02 / Hf), handlelength=1.0,
                   handleheight=1.0, columnspacing=1.4)
        letter(fig, 0.02, B + PH + 0.20, "A")
        letter(fig, L + AW + G - 0.12, B + PH + 0.20, "B")
        save(fig, name)


if __name__ == "__main__":
    for f in (figure1, figure2, figure3, figure4, figure5, figure6, figureS1, figure_agenoria):
        f()
