"""Nightly sleep timing, daily sleep phase and phase-jump screening for the NYU infant-sleep data.

Usage: python3 -I nyu_sleep_timing.py <infant_sleep_preprocessed.mat> <out_dir>
Outputs in <out_dir>:
  nyu_timing_arrays.npz          per infant x day: night bedtime/wake/mid-sleep/length, daily phase, R
  nyu_phase_jumps.csv            one row per detected jump episode
  nyu_sleep_timing_weekly.csv    one row per infant-week (jump days excluded)
  nyu_infant_timing_summary.csv  one row per infant: timing at 6/12/18/24 months and number of jumps

Definitions (row 0 of the data = day 1 of age; clock times are as logged)
  asleep bin    share asleep in the 10-min bin >= 0.5
  night n       12:00 on day n to 12:00 on day n+1; both days must be logged
  night period  the longest sleep run in that window after merging bouts separated by wakes of
                <= 60 min, if 4-16 h long with its midpoint between 18:00 and 09:00
                (the "sleep period" from bedtime to final morning waking, night feeds included)
  bedtime, wake time, mid-sleep   start, end and midpoint of the night period
  daily phase   circular mean clock time of all sleep in a calendar day, weighted by share asleep;
                R (0-1) is how concentrated sleep is around it
  jump          daily phase >= 4 h from the circular mean of the previous 14 days (>= 7 usable days,
                usable = R >= 0.15 and age >= 42 days) on >= 3 usable days in a row with the same sign
                and within 4 h of each other (gaps of <= 3 days allowed); it lasts while each day's
                phase is nearer the episode's running mean phase than the baseline, and ends when
                2 usable days in a row are nearer the baseline ('round trip') or at the end of the
                record ('one-way'). shift_h = episode mean phase minus baseline, wrapped to +-12 h
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MERGE_BINS = 6            # wakes of <= 60 min are bridged
MIN_LEN_H, MAX_LEN_H = 4, 16
MID_LO, MID_HI = 18, 33   # midpoint between 18:00 and 09:00 next day, in hours after day-n midnight
R_MIN, AGE_MIN = 0.15, 42
BASE_DAYS, MIN_BASE, THR_H, RET_H, MIN_RUN, MAX_GAP = 14, 7, 4.0, 3.0, 3, 3
ANG = 2 * np.pi * (np.arange(144) + 0.5) / 144
WINDOWS = {"6mo": (152, 213), "12mo": (335, 396), "18mo": (517, 578), "24mo": (669, 730)}


def wrap(h):
    """Wrap hour differences to (-12, 12]."""
    return (np.asarray(h) + 12) % 24 - 12


def circmean_h(h):
    a = np.asarray(h) * 2 * np.pi / 24
    return (np.arctan2(np.sin(a).mean(), np.cos(a).mean()) % (2 * np.pi)) * 24 / (2 * np.pi)


def night_period(x144):
    """x144: share asleep in 144 bins from 12:00. Returns (bedtime, wake, mid, length) in hours
    after midnight of day n (bedtime 19.5 = 19:30, wake 30.5 = 06:30 next day), or None."""
    s = np.asarray(x144) >= 0.5
    if not s.any():
        return None
    edges = np.flatnonzero(np.diff(np.r_[0, s.astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2]
    runs, cs, ce = [], starts[0], ends[0]
    for a, b in zip(starts[1:], ends[1:]):
        if a - ce <= MERGE_BINS:
            ce = b
        else:
            runs.append((cs, ce))
            cs, ce = a, b
    runs.append((cs, ce))
    best = None
    for cs, ce in runs:
        length, mid = (ce - cs) / 6, 12 + (cs + ce) / 12
        if MIN_LEN_H <= length <= MAX_LEN_H and MID_LO <= mid <= MID_HI:
            if best is None or length > best[3]:
                best = (12 + cs / 6, 12 + ce / 6, mid, length)
    return best


def daily_phase(x):
    """x: (days, 144). Returns phase (hours, 0-24) and R per day (NaN if unlogged)."""
    w = np.nan_to_num(x)
    C, S, W = w @ np.cos(ANG), w @ np.sin(ANG), w.sum(1)
    logged = ~np.isnan(x).all(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        phase = np.where(logged & (W > 0), (np.arctan2(S, C) % (2 * np.pi)) * 24 / (2 * np.pi), np.nan)
        R = np.where(logged & (W > 0), np.hypot(C, S) / W, np.nan)
    return phase, R


def detect_jumps(phase, R):
    """Phase-jump episodes; see the module docstring. Circular throughout: a day belongs to an
    episode while its phase is closer to the episode's own (running) mean phase than to the
    pre-jump baseline, which copes with ~12-h shifts and with gradual developmental drift."""
    n = phase.size
    usable = ~np.isnan(phase) & (R >= R_MIN) & (np.arange(n) + 1 >= AGE_MIN)
    days = np.flatnonzero(usable)
    dist = lambda a, b: np.abs(wrap(a - b))
    episodes, i, blocked_until = [], 0, -1
    while i < days.size:
        d = days[i]
        prev = days[(days >= d - BASE_DAYS) & (days < d) & (days > blocked_until)]
        if prev.size < MIN_BASE:
            i += 1
            continue
        base = circmean_h(phase[prev])
        if dist(phase[d], base) < THR_H:
            i += 1
            continue
        run, j = [d], i + 1
        while j < days.size and len(run) < MIN_RUN:
            dj = days[j]
            if (dj - run[-1] > MAX_GAP or dist(phase[dj], base) < THR_H
                    or dist(phase[dj], circmean_h(phase[run])) >= THR_H):
                break
            run.append(dj)
            j += 1
        if len(run) < MIN_RUN:
            i += 1
            continue
        ep, k, returned = list(run), j, False
        nearer_jump = lambda day: dist(phase[day], circmean_h(phase[ep])) < dist(phase[day], base)
        while k < days.size:
            if nearer_jump(days[k]):
                ep.append(days[k])
            elif k + 1 < days.size and not nearer_jump(days[k + 1]):
                returned = True
                break
            k += 1
        last = ep[-1]
        target = circmean_h(phase[ep])
        gap_before = int(np.sum(np.isnan(phase[max(d - 3, 0):d])))
        episodes.append(dict(start_day=int(d) + 1, end_day=int(last) + 1,
                             span_days=int(last - d) + 1,
                             shift_h=round(float(wrap(target - base)), 2),
                             type="round trip" if returned else "one-way",
                             baseline_phase_h=round(float(base), 2),
                             unlogged_days_in_3_before=gap_before))
        blocked_until = last
        i = int(np.searchsorted(days, last, side="right"))
    return episodes


def analyse_infant(x):
    """x: (days, 144). Returns per-day arrays and jump episodes."""
    n = x.shape[0]
    night = np.full((n, 4), np.nan)                  # bedtime, wake, mid, length (hours)
    logged = ~np.isnan(x).all(1)
    for d in range(n - 1):
        if logged[d] and logged[d + 1]:
            r = night_period(np.r_[x[d, 72:], x[d + 1, :72]])
            if r is not None:
                night[d] = r
    phase, R = daily_phase(x)
    return night, phase, R, detect_jumps(phase, R)


def main(mat_path, out_dir):
    from scipy.io import loadmat
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = loadmat(mat_path)["data"]
    n_days, _, n_inf = data.shape
    nights = np.full((n_inf, n_days, 4), np.nan, dtype=np.float32)
    phases = np.full((n_inf, n_days), np.nan, dtype=np.float32)
    Rs = np.full((n_inf, n_days), np.nan, dtype=np.float32)
    jump_mask = np.zeros((n_inf, n_days), dtype=bool)
    jumps = []
    for i in range(n_inf):
        night, phase, R, eps = analyse_infant(data[:, :, i])
        nights[i], phases[i], Rs[i] = night, phase, R
        for e in eps:
            jumps.append(dict(infant=i + 1, **e))
            jump_mask[i, e["start_day"] - 1:e["end_day"]] = True
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} infants done", flush=True)
    np.savez_compressed(out / "nyu_timing_arrays.npz", nights=nights, phase=phases, R=Rs,
                        jump_mask=jump_mask)
    jd = pd.DataFrame(jumps, columns=["infant", "start_day", "end_day", "span_days", "shift_h", "type",
                                      "baseline_phase_h", "unlogged_days_in_3_before"])
    jd.to_csv(out / "nyu_phase_jumps.csv", index=False)

    clean = nights.copy()
    clean[jump_mask] = np.nan                          # timing summaries exclude jump days
    weekly = []
    for i in range(n_inf):
        for w in range(n_days // 7):
            seg = clean[i, 7 * w:7 * w + 7]
            ok = ~np.isnan(seg[:, 0])
            if ok.sum() >= 3:
                med = np.median(seg[ok], axis=0)
                weekly.append((i + 1, w + 1, int(ok.sum()), *(np.round(med[:3] % 24, 3)),
                               round(float(med[3]), 3)))
    pd.DataFrame(weekly, columns=["infant", "week_of_age", "n_nights", "bedtime_h", "wake_time_h",
                                  "mid_sleep_h", "night_period_h"]).to_csv(
        out / "nyu_sleep_timing_weekly.csv", index=False)

    rows = []
    for i in range(n_inf):
        rec = dict(infant=i + 1, n_jumps=int((jd.infant == i + 1).sum()))
        for lab, (a, b) in WINDOWS.items():
            seg = clean[i, a - 1:b]
            ok = ~np.isnan(seg[:, 0])
            for k, name in enumerate(("bedtime", "wake_time", "mid_sleep", "night_period")):
                v = np.median(seg[ok, k]) if ok.sum() >= 10 else np.nan
                rec[f"{name}_{lab}"] = round(float(v % 24 if k < 3 else v), 3) if not np.isnan(v) else np.nan
            rec[f"n_nights_{lab}"] = int(ok.sum())
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out / "nyu_infant_timing_summary.csv", index=False)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
