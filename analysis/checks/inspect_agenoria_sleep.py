"""Inspect Agenoria Glow Baby sleep exports for quality issues.

Usage: python3 -I inspect_agenoria_sleep.py <data_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
BABIES = {"zyw": "2018-11-21", "zlw": "2022-02-21"}  # birthdays from the repo configs

for code, bday in BABIES.items():
    bday = pd.Timestamp(bday)
    df = pd.read_csv(DATA / code / "glow_sleep.csv")
    df.columns = ["begin", "end"]
    n_raw = len(df)
    df["begin"] = pd.to_datetime(df["begin"], format="%m/%d/%Y %I:%M:%S %p")
    df["end"] = pd.to_datetime(df["end"], format="%m/%d/%Y %I:%M:%S %p")
    dupes = df.duplicated().sum()
    df = df.drop_duplicates().sort_values("begin").reset_index(drop=True)
    df["dur_min"] = (df["end"] - df["begin"]).dt.total_seconds() / 60

    print(f"\n===== {code}  (born {bday.date()}) =====")
    print(f"rows: {n_raw}, exact duplicates: {dupes}, unique sessions: {len(df)}")
    print(f"range: {df.begin.min()}  ->  {df.end.max()}")
    print(f"age at first/last record (days): {(df.begin.min() - bday).days} / {(df.end.max() - bday).days}")
    print("duration (min) quantiles:", df.dur_min.quantile([0, .01, .05, .5, .95, .99, 1]).round(1).to_dict())
    print(f"non-positive durations: {(df.dur_min <= 0).sum()}")
    print(f"<=2 min sessions: {(df.dur_min <= 2).sum()}  | >14 h sessions: {(df.dur_min > 14*60).sum()}")
    print(df[df.dur_min > 14 * 60].to_string() if (df.dur_min > 14 * 60).any() else "")

    # overlaps: a session starting before the previous one ended
    prev_end = df["end"].cummax().shift()
    ov = df[df["begin"] < prev_end]
    print(f"overlapping sessions: {len(ov)}")
    if len(ov):
        overlap_min = (prev_end[ov.index] - ov["begin"]).dt.total_seconds() / 60
        print("  overlap minutes quantiles:", overlap_min.quantile([.5, .9, 1]).round(1).to_dict())

    # gaps between consecutive sessions (awake or unlogged)
    gaps = (df["begin"] - prev_end).dt.total_seconds() / 3600
    big = gaps[gaps > 8]
    print(f"gaps > 8 h: {len(big)}  | > 12 h: {(gaps > 12).sum()}  | > 24 h: {(gaps > 24).sum()}")
    if len(big):
        show = pd.DataFrame({"gap_start": prev_end[big.index], "gap_h": big.round(1),
                             "age_d": (prev_end[big.index] - bday).dt.days})
        print(show.head(40).to_string())

    # minute-level coverage per age-day for year one
    start = bday
    n_days = 366
    asleep = np.zeros(n_days * 1440, dtype=bool)
    t0 = start.value // 60_000_000_000
    for b, e in zip(df["begin"], df["end"]):
        i = b.value // 60_000_000_000 - t0
        j = e.value // 60_000_000_000 - t0
        i, j = max(i, 0), min(j, n_days * 1440)
        if j > i:
            asleep[i:j] = True
    daily = asleep.reshape(n_days, 1440).sum(1) / 60
    print("daily logged sleep (h) in year one, quantiles:", np.round(np.quantile(daily, [0, .05, .5, .95, 1]), 1))
    low = np.where(daily < 8)[0]
    print(f"days with < 8 h logged sleep: {len(low)} -> ages {low.tolist()[:60]}")

    misc = pd.read_csv(DATA / code / "misc.csv")
    misc["Date"] = pd.to_datetime(misc["Date"])
    dc = misc[misc["Daycare"] == 1]
    if len(dc):
        print(f"daycare first day: {dc.Date.min().date()} (age {(dc.Date.min() - bday).days} d), days flagged: {len(dc)}")
