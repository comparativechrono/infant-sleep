"""Build a day-by-day sleep raster ("one row per day") for the two Agenoria babies.

Usage: python3 -I agenoria_sleep_chart.py <agenoria_data_dir> <out_dir>

Data: Jiuguang Wang, Agenoria (github.com/jiuguangw/Agenoria), CC BY-NC-SA 4.0.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperfig import headline, save  # noqa: E402


DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

BABIES = [  # (code in repo, label, birthday from the repo config)
    ("zyw", "Baby 1", "2018-11-21"),
    ("zlw", "Baby 2", "2022-02-21"),
]
N_DAYS = 365          # first year of life, day 0 = date of birth
BIN = 2               # minutes per image column

# ---- palette (dataviz reference palette; blue/yellow validated all-pairs) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SLEEP = "#2a78d6"     # categorical slot 1
AWAKE = "#eda100"     # categorical slot 4
NO_RECORD = HAIRLINE

for f in glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 9})


def load_sessions(code: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / code / "glow_sleep.csv", names=["begin", "end"], header=0)
    for c in ("begin", "end"):
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y %I:%M:%S %p")
    df = df.drop_duplicates().sort_values("begin").reset_index(drop=True)
    df["duration_min"] = (df["end"] - df["begin"]).dt.total_seconds() / 60
    return df


def minute_states(df: pd.DataFrame, bday: pd.Timestamp, n_days: int) -> np.ndarray:
    """0 = no record, 1 = awake (not logged as asleep), 2 = asleep. Shape (n_days, 1440)."""
    n = n_days * 1440
    t0 = bday.value // 60_000_000_000
    to_min = lambda ts: ts.value // 60_000_000_000 - t0
    state = np.ones(n, dtype=np.int8)
    first, last = to_min(df["begin"].min()), to_min(df["end"].max())
    state[: max(first, 0)] = 0
    if last < n:
        state[max(last, 0):] = 0
    for b, e in zip(df["begin"], df["end"]):
        i, j = max(to_min(b), 0), min(to_min(e), n)
        if j > i:
            state[i:j] = 2
    return state.reshape(n_days, 1440)


def to_bins(states: np.ndarray, width: int) -> np.ndarray:
    """Collapse minutes into bins: asleep if any minute asleep, else awake if any awake."""
    s = states.reshape(states.shape[0], -1, width)
    return np.where((s == 2).any(-1), 2, np.where((s == 1).any(-1), 1, 0))


def rgb(hexstr: str) -> np.ndarray:
    return np.array(matplotlib.colors.to_rgb(hexstr))


lut = np.stack([rgb(NO_RECORD), rgb(AWAKE), rgb(SLEEP)])

# ---- data ----
panels, tidy = [], []
for code, label, bday in BABIES:
    bday = pd.Timestamp(bday)
    df = load_sessions(code)
    prev_end = df["end"].cummax().shift()
    tidy.append(pd.DataFrame({
        "baby": label,
        "sleep_start": df["begin"].dt.strftime("%Y-%m-%d %H:%M"),
        "sleep_end": df["end"].dt.strftime("%Y-%m-%d %H:%M"),
        "duration_min": df["duration_min"].astype(int),
        "age_days_at_start": (df["begin"].dt.normalize() - bday).dt.days,
        "short_le_2min": df["duration_min"] <= 2,
        "overlaps_previous": (df["begin"] < prev_end).fillna(False),
    }))
    states = minute_states(df, bday, N_DAYS)
    np.save(OUT / f"states_{code}.npy", states)  # for verification
    misc = pd.read_csv(DATA / code / "misc.csv", parse_dates=["Date"])
    daycare_day = (misc.loc[misc["Daycare"] == 1, "Date"].min() - bday).days
    panels.append((label, bday, lut[to_bins(states, BIN)], daycare_day))

pd.concat(tidy).to_csv(OUT / "agenoria_sleep_sessions.csv", index=False)

# ---- figure geometry (inches) ----
DPI = 220
L, PANEL_W, GAP, R = 1.0, 3.3, 0.62, 0.5
TOP, PANEL_H, BOTTOM = 2.5, 8.3, 1.05
W = L + 2 * PANEL_W + GAP + R
H = TOP + PANEL_H + BOTTOM
fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)

month_starts = [int(round(k * 365.25 / 12)) for k in range(12)]

for idx, (label, bday, img, daycare_day) in enumerate(panels):
    x0 = L + idx * (PANEL_W + GAP)
    ax = fig.add_axes([x0 / W, BOTTOM / H, PANEL_W / W, PANEL_H / H])
    ax.imshow(img, aspect="auto", interpolation="nearest", extent=(0, 24, N_DAYS, 0))
    ax.set_xlim(0, 24)
    ax.set_ylim(N_DAYS, 0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])
    ax.set_xticks(np.arange(0, 25, 2), minor=True)
    ax.tick_params(axis="x", which="both", top=True, labeltop=True, colors=MUTED,
                   labelcolor=MUTED, labelsize=8, length=3, width=0.6, pad=3)
    ax.tick_params(axis="x", which="minor", length=1.8)
    ax.set_yticks(month_starts + [N_DAYS])
    ax.tick_params(axis="y", which="both", left=True, labelleft=False, colors=AXIS,
                   length=3, width=0.6)

    # panel title
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(0, 24),
                textcoords="offset points", fontsize=12, fontweight="semibold",
                color=INK, va="bottom", ha="left")
    ax.annotate(f"born {bday:%b %Y}", xy=(1, 1), xycoords="axes fraction",
                xytext=(0, 24.5), textcoords="offset points", fontsize=9,
                color=INK_2, va="bottom", ha="right")

    # first day of daycare marker, just outside the right edge
    ax.plot([24.35], [daycare_day + 0.5], marker="<", markersize=5.5, color=INK_2,
            clip_on=False, zorder=5)

    if idx == 0:  # month labels, left of the first panel
        for k, d in enumerate(month_starts):
            ax.annotate(f"MONTH {k + 1}", xy=(0, d), xycoords=("axes fraction", "data"),
                        xytext=(-8, -1), textcoords="offset points", ha="right", va="top",
                        fontsize=8, fontweight="semibold", color=INK)
            ax.annotate(f"Day {d + 1}", xy=(0, d), xycoords=("axes fraction", "data"),
                        xytext=(-8, -11.5), textcoords="offset points", ha="right",
                        va="top", fontsize=7.5, color=MUTED)
        ax.annotate(f"Day {N_DAYS}", xy=(0, N_DAYS), xycoords=("axes fraction", "data"),
                    xytext=(-8, 0), textcoords="offset points", ha="right",
                    va="bottom", fontsize=7.5, color=MUTED)

# ---- header ----
headline(fig, L / W, 1 - 0.38 / H, "A year of sleep, one day at a time", fontsize=21,
         fontweight="bold", color=INK, ha="left", va="top")
headline(fig, L / W, 1 - 0.88 / H,
         "Two babies' sleep, logged by their parent in the Glow Baby app. Each row is one day "
         "of life,\nmidnight to midnight in local clock time; anything not logged as sleep "
         "counts as awake.",
         fontsize=9.5, color=INK_2, ha="left", va="top", linespacing=1.45)
handles = [
    Patch(facecolor=SLEEP, edgecolor="none"),
    Patch(facecolor=AWAKE, edgecolor="none"),
    Patch(facecolor=NO_RECORD, edgecolor="none"),
    Line2D([], [], marker="<", color=INK_2, linestyle="none", markersize=5.5),
]
fig.legend(handles, ["Asleep", "Awake", "No record yet", "First day of daycare"],
           loc="upper left", bbox_to_anchor=(L / W - 0.006, 1 - 1.5 / H), ncol=4,
           frameon=False, fontsize=9.5, handlelength=1.1, handleheight=1.1,
           handletextpad=0.5, columnspacing=1.8, labelcolor=INK)

# ---- footer ----
fig.text((L + PANEL_W + GAP / 2) / W, (BOTTOM - 0.4) / H, "Time of day", fontsize=9.5,
         color=INK_2, ha="center", va="top")
headline(fig, L / W, 0.22 / H,
         "Data: Jiuguang Wang, Agenoria (github.com/jiuguangw/Agenoria), CC BY-NC-SA 4.0. "
         "This chart is shared under the same licence.",
         fontsize=7.5, color=MUTED, ha="left", va="bottom")

path = save(fig, OUT / "agenoria_sleep_first_year.png", DPI, "FigureS2", SURFACE)
print("saved", path, f"{W * DPI:.0f}x{H * DPI:.0f}px")
