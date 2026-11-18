"""Double-plotted sleep actogram + sliding-window Lomb-Scargle periodogram, one figure per baby.

Usage: python3 -I agenoria_actograms.py <agenoria_data_dir> <out_dir>
Needs rhythm.py next to this script.
Data: Jiuguang Wang, Agenoria (github.com/jiuguangw/Agenoria), CC BY-NC-SA 4.0.
"""
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figsave import headline, save  # noqa: E402
from rhythm import load_sessions, lomb_scargle, minute_states, sleep_fraction

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
BABIES = [("zyw", "Baby 1", "2018-11-21"), ("zlw", "Baby 2", "2022-02-21")]
N_DAYS = 365              # rows: age day 0 (date of birth) .. 364
PAD = 7                   # days of padding either side for the sliding windows
ACT_BIN = 3               # minutes per actogram column
PG_BIN = 10               # minutes per periodogram sample
WIN = 14                  # window length in days, centred on each row
MIN_COVER = 0.8           # share of the window that must be recorded
PERIODS = np.geomspace(2, 30, 400)   # hours, log-spaced

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
HAIRLINE, AXIS = "#e1e0d9", "#c3c2b7"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

for f in glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 9})


def analyse(code, label, bday):
    bday = pd.Timestamp(bday)
    df = load_sessions(DATA, code)
    ext = minute_states(df, bday - pd.Timedelta(days=PAD), N_DAYS + 2 * PAD + 1)  # row k = age k-PAD

    # actogram: row a = age day a (left) + age day a+1 (right), 3-min columns
    left = ext[PAD:PAD + N_DAYS]
    right = ext[PAD + 1:PAD + N_DAYS + 1]
    two = np.concatenate([left, right], axis=1).reshape(N_DAYS, -1, ACT_BIN)
    act = np.where((two == 2).any(-1), 2, np.where((two == 1).any(-1), 1, 0))

    # sliding-window Lomb-Scargle on the share of each 10-min bin spent asleep
    frac = sleep_fraction(ext, PG_BIN).reshape(-1)          # (days * 144,)
    per_day = 1440 // PG_BIN
    t_h = (np.arange(WIN * per_day) + 0.5) * PG_BIN / 60
    heat = np.full((N_DAYS, PERIODS.size), np.nan)
    for a in range(N_DAYS):
        k0 = (a + PAD - WIN // 2) * per_day                  # window: ages a-7 .. a+6
        x = frac[k0:k0 + WIN * per_day]
        if np.mean(~np.isnan(x)) >= MIN_COVER:
            heat[a] = lomb_scargle(t_h, x, PERIODS)

    misc = pd.read_csv(DATA / code / "misc.csv", parse_dates=["Date"])
    daycare = (misc.loc[misc["Daycare"] == 1, "Date"].min() - bday).days
    return dict(label=label, bday=bday, act=act, heat=heat, daycare=daycare, ext=ext)


results = [analyse(*b) for b in BABIES]
VMAX = float(np.ceil(np.nanmax([np.nanmax(r["heat"]) for r in results]) * 10) / 10)

# actogram codes (0 = no record, 1 = awake, 2 = asleep) and periodograms, for paper_figures.py
arrays = {"periods_h": PERIODS}
for r in results:
    tag = r["label"].replace(" ", "").lower()
    arrays.update({f"{tag}_act": r["act"].astype(np.uint8), f"{tag}_heat": r["heat"],
                   f"{tag}_daycare_day": np.int64(r["daycare"])})
np.savez_compressed(OUT / "agenoria_actograms.npz", **arrays)

# ---- per-window summary table ----
rows = []
circ = (PERIODS >= 20) & (PERIODS <= 28)
ultra = (PERIODS >= 2) & (PERIODS <= 8)
i24 = int(np.argmin(np.abs(PERIODS - 24)))
for r in results:
    for a, p in enumerate(r["heat"]):
        if np.isnan(p).all():
            continue
        ic = np.flatnonzero(circ)[np.argmax(p[circ])]
        iu = np.flatnonzero(ultra)[np.argmax(p[ultra])]
        rows.append({"baby": r["label"], "age_day_window_centre": a,
                     "circadian_peak_period_h": round(PERIODS[ic], 2),
                     "circadian_peak_var_explained": round(p[ic], 4),
                     "var_explained_at_24h": round(p[i24], 4),
                     "ultradian_peak_period_h_2_to_8": round(PERIODS[iu], 2),
                     "ultradian_peak_var_explained": round(p[iu], 4)})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "periodogram_summary_by_day.csv", index=False)

# ---- figures ----
month_starts = [int(round(k * 365.25 / 12)) for k in range(12)]
act_lut = np.stack([matplotlib.colors.to_rgb(c) for c in (HAIRLINE, SURFACE, INK)])
cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
cmap.set_bad(HAIRLINE)
VMIN = 0.005   # ~noise level for a 14-day window of 10-min bins (mean LS power ~ 2/N)
norm = LogNorm(vmin=VMIN, vmax=VMAX, clip=True)
lx = np.log10(PERIODS)
half = (lx[1] - lx[0]) / 2

DPI = 220
L, ACT_W, GAP, PER_W, R = 1.0, 4.2, 0.55, 2.3, 0.35
TOP, PANEL_H, BOTTOM = 2.8, 8.3, 1.75
W, H = L + ACT_W + GAP + PER_W + R, TOP + PANEL_H + BOTTOM

for r in results:
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes([L / W, BOTTOM / H, ACT_W / W, PANEL_H / H])
    ax2 = fig.add_axes([(L + ACT_W + GAP) / W, BOTTOM / H, PER_W / W, PANEL_H / H])

    ax.imshow(act_lut[r["act"]], aspect="auto", interpolation="nearest",
              extent=(0, 48, N_DAYS, 0))
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 12, 24, 36, 48])
    ax.set_xticklabels(["00:00", "12:00", "00:00", "12:00", "24:00"])
    ax.set_xticks(np.arange(0, 49, 6), minor=True)

    ax2.imshow(np.ma.masked_invalid(np.clip(r["heat"], VMIN, None)), cmap=cmap, norm=norm,
               aspect="auto",
               interpolation="nearest", extent=(lx[0] - half, lx[-1] + half, N_DAYS, 0))
    ticks = [2, 3, 4, 6, 8, 12, 24]
    ax2.set_xticks(np.log10(ticks))
    ax2.set_xticklabels([str(t) for t in ticks])
    ax2.set_xticks([], minor=True)

    for a in (ax, ax2):
        a.set_ylim(N_DAYS, 0)
        for s in a.spines.values():
            s.set_visible(False)
        a.tick_params(axis="x", which="both", top=True, labeltop=True, colors=MUTED,
                      labelcolor=MUTED, labelsize=8, length=3, width=0.6, pad=3)
        a.tick_params(axis="x", which="minor", length=1.8)
        a.set_yticks(month_starts + [N_DAYS])
        a.tick_params(axis="y", left=True, labelleft=False, colors=AXIS, length=3, width=0.6)

    for a, title in ((ax, "Double-plotted actogram"), (ax2, "Lomb–Scargle periodogram")):
        a.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 24),
                   textcoords="offset points", fontsize=11, fontweight="semibold",
                   color=INK, va="bottom", ha="left")

    for k, d in enumerate(month_starts):
        ax.annotate(f"MONTH {k + 1}", xy=(0, d), xycoords=("axes fraction", "data"),
                    xytext=(-8, -1), textcoords="offset points", ha="right", va="top",
                    fontsize=8, fontweight="semibold", color=INK)
        ax.annotate(f"Day {d + 1}", xy=(0, d), xycoords=("axes fraction", "data"),
                    xytext=(-8, -11.5), textcoords="offset points", ha="right", va="top",
                    fontsize=7.5, color=MUTED)
    ax.annotate(f"Day {N_DAYS}", xy=(0, N_DAYS), xycoords=("axes fraction", "data"),
                xytext=(-8, 0), textcoords="offset points", ha="right", va="bottom",
                fontsize=7.5, color=MUTED)
    ax.plot([48.7], [r["daycare"] + 0.5], marker="<", markersize=5.5, color=INK_2,
            clip_on=False, zorder=5)

    # axis titles
    ax.annotate("Local clock time, two consecutive days per row", xy=(0.5, 0),
                xycoords="axes fraction", xytext=(0, -26), textcoords="offset points",
                ha="center", va="top", fontsize=9, color=INK_2)
    ax2.annotate("Period (hours, log scale)", xy=(0.5, 0), xycoords="axes fraction",
                 xytext=(0, -26), textcoords="offset points", ha="center", va="top",
                 fontsize=9, color=INK_2)

    # colour bar under the periodogram
    cax = fig.add_axes([(L + ACT_W + GAP) / W, (BOTTOM - 1.08) / H, PER_W / W, 0.1 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.outline.set_visible(False)
    cticks = [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if t <= VMAX]
    cb.set_ticks(cticks)
    cb.set_ticklabels([f"{t * 100:g}" for t in cticks])
    cb.minorticks_off()
    cax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=7.5, length=2.5, width=0.6)
    cax.annotate("Variance explained (%, log scale)", xy=(0, 1),
                 xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                 ha="left", va="bottom", fontsize=8, color=INK_2)

    # header
    headline(fig, L / W, 1 - 0.38 / H, f"{r['label']}: actogram and periodogram, first year",
             fontsize=20, fontweight="bold", color=INK, ha="left", va="top")
    headline(fig, L / W, 1 - 0.85 / H,
             f"Born {r['bday']:%b %Y}. Left: sleep double-plotted, each row showing day n and "
             "day n + 1 in local clock time.\nRight: Lomb–Scargle periodogram of a 14-day window "
             "centred on each row (10-min bins), shaded by the\nvariance a sinusoid at that "
             "period explains. Peaks at 12, 8, 6 and 4.8 h can be harmonics of 24 h.",
             fontsize=9.5, color=INK_2, ha="left", va="top", linespacing=1.45)
    handles = [Patch(facecolor=INK, edgecolor="none"),
               Patch(facecolor=SURFACE, edgecolor=AXIS, linewidth=0.8),
               Patch(facecolor=HAIRLINE, edgecolor="none"),
               Line2D([], [], marker="<", color=INK_2, linestyle="none", markersize=5.5)]
    fig.legend(handles, ["Asleep", "Awake", "No record or incomplete window",
                         "First day of daycare"],
               loc="upper left", bbox_to_anchor=(L / W - 0.006, 1 - 1.72 / H), ncol=4,
               frameon=False, fontsize=9.5, handlelength=1.1, handleheight=1.1,
               handletextpad=0.5, columnspacing=1.6, labelcolor=INK)
    headline(fig, L / W, 0.22 / H,
             "Data: Jiuguang Wang, Agenoria (github.com/jiuguangw/Agenoria), CC BY-NC-SA 4.0. "
             "This chart is shared under the same licence.",
             fontsize=7.5, color=MUTED, ha="left", va="bottom")

    name = f"actogram_periodogram_{r['label'].lower().replace(' ', '')}.png"
    save(fig, OUT / name, DPI, SURFACE)
    plt.close(fig)
    np.save(OUT / f"_act_{r['label'].replace(' ', '')}.npy", r["act"])      # for verification
    np.save(OUT / f"_heat_{r['label'].replace(' ', '')}.npy", r["heat"])
    print("saved", name, f"{W * DPI:.0f}x{H * DPI:.0f}px")

print("VMAX", VMAX)
