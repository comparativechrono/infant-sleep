"""Verify the sleep raster against the raw Glow sessions, independently of the build code.

Usage: python3 -I verify_agenoria_chart.py <agenoria_data_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BABIES = [("zyw", "Baby 1", "2018-11-21"), ("zlw", "Baby 2", "2022-02-21")]
N_DAYS = 365
# geometry mirrored from the build script (inches) and colours
DPI, L, PANEL_W, GAP, TOP, PANEL_H = 220, 1.0, 3.3, 0.62, 2.5, 8.3
COLORS = {0: (0xE1, 0xE0, 0xD9), 1: (0xED, 0xA1, 0x00), 2: (0x2A, 0x78, 0xD6)}

img = np.asarray(Image.open(OUT / "agenoria_sleep_first_year.png").convert("RGB")).astype(int)
tidy = pd.read_csv(OUT / "agenoria_sleep_sessions.csv")
rng = np.random.default_rng(7)
all_ok = True

for p, (code, label, bday) in enumerate(BABIES):
    bday = pd.Timestamp(bday)
    raw = pd.read_csv(DATA / code / "glow_sleep.csv", names=["b", "e"], header=0)
    raw["b"] = pd.to_datetime(raw["b"], format="%m/%d/%Y %I:%M:%S %p")
    raw["e"] = pd.to_datetime(raw["e"], format="%m/%d/%Y %I:%M:%S %p")
    raw = raw.drop_duplicates()
    raw_min = raw.assign(b=raw["b"].dt.floor("min"), e=raw["e"].dt.floor("min"))  # chart resolution
    states = np.load(OUT / f"states_{code}.npy")

    # 1) independent interval union -> asleep minutes per age-day, vs. the matrix
    iv = raw_min.sort_values("b")[["b", "e"]].to_numpy()
    merged = []
    for b, e in iv:
        if merged and b <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([b, e])
    per_day = np.zeros(N_DAYS)
    for b, e in merged:
        t = pd.Timestamp(b)
        e = pd.Timestamp(e)
        while t < e:
            day_end = t.normalize() + pd.Timedelta(days=1)
            seg_end = min(e, day_end)
            d = (t.normalize() - bday).days
            if 0 <= d < N_DAYS:
                per_day[d] += (seg_end - t).total_seconds() / 60
            t = seg_end
    matrix_per_day = (states == 2).sum(1)
    diff = np.abs(per_day - matrix_per_day)
    print(f"{label}: daily asleep minutes, union vs matrix: max |diff| = {diff.max():.0f} min")
    all_ok &= diff.max() == 0

    # 2) the tidy CSV holds every unique raw session
    n_tidy = (tidy["baby"] == label).sum()
    print(f"{label}: tidy rows {n_tidy} vs unique raw sessions {len(raw)}")
    all_ok &= n_tidy == len(raw)

    # 3) no-record region: before the first log and on the birth date
    first = raw["b"].min()
    first_min = int((first - bday).total_seconds() // 60)
    flat = states.reshape(-1)
    print(f"{label}: first log {first}, minutes before it all 'no record': "
          f"{bool((flat[:first_min] == 0).all())}, none after it: {bool((flat[first_min:] != 0).all())}")
    all_ok &= bool((flat[:first_min] == 0).all()) and bool((flat[first_min:] != 0).all())

    # 4) rendered pixels match the state at sampled (day, time) points
    left = (L + p * (PANEL_W + GAP)) * DPI
    top = TOP * DPI
    pw, ph = PANEL_W * DPI, PANEL_H * DPI
    checked = bad = 0
    while checked < 400:
        d = int(rng.integers(0, N_DAYS))
        m = int(rng.integers(6, 1434))
        window = states[d, m - 6:m + 6]
        if not (window == window[0]).all():
            continue  # skip cells near a transition, where a pixel can straddle two states
        x = int(left + (m + 0.5) / 1440 * pw)
        y = int(top + (d + 0.5) / N_DAYS * ph)
        px = img[y, x]
        if np.abs(px - COLORS[int(window[0])]).max() > 12:
            bad += 1
            if bad <= 3:
                print(f"   mismatch day {d} min {m}: pixel {px}, expected {COLORS[int(window[0])]}")
        checked += 1
    print(f"{label}: pixel spot-checks {checked}, mismatches {bad}")
    all_ok &= bad == 0

    # 5) a few descriptive numbers for the write-up
    s = raw.copy()
    s["dur_h"] = (s["e"] - s["b"]).dt.total_seconds() / 3600
    s["age_d"] = (s["b"].dt.normalize() - bday).dt.days
    night = s[(s["b"].dt.hour >= 18) | (s["b"].dt.hour < 6)]
    longest = night.groupby("age_d")["dur_h"].max()
    by_month = longest.groupby((longest.index // 30.4375).astype(int)).median()
    print(f"{label}: median longest night-time session (h) by month of age:",
          {int(k) + 1: round(v, 1) for k, v in by_month.items() if k < 12})
    asleep = states == 2
    night_cols = np.r_[0:7 * 60, 19 * 60:1440]
    frac = asleep[:, night_cols].sum(1) / np.maximum(asleep.sum(1), 1)
    mfrac = pd.Series(frac).groupby((np.arange(N_DAYS) // 30.4375).astype(int)).mean()
    print(f"{label}: share of daily sleep between 19:00 and 07:00 by month:",
          {int(k) + 1: round(v, 2) for k, v in mfrac.items()})
    print(f"{label}: mean daily logged sleep (h), months 1 / 6 / 12:",
          [round(asleep[int(a):int(b)].sum(1).mean() / 60, 1) for a, b in
           [(0, 30), (152, 183), (335, 365)]])

print("\nALL CHECKS PASS" if all_ok else "\nSOME CHECKS FAILED")
