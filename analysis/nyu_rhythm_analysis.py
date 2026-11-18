"""Per-infant sliding-window Lomb-Scargle analysis of the NYU infant-sleep data (Liu et al. 2019).

Usage: python3 -I nyu_rhythm_analysis.py <infant_sleep_preprocessed.mat> <out_dir>
Needs rhythm.py next to this script. Outputs in <out_dir>:
  nyu_rhythm_windows.npz          per infant x day: 24-h strength, circadian peak, ultradian peak
  nyu_infant_rhythm_summary.csv   one row per infant: coverage, plateau, emergence ages, free-running
  nyu_infant_weekly_rhythm.csv    one row per infant-week: mean 24-h strength and circadian peak period

Definitions
  window      14 days of 10-min bins centred on each day of age; used if >= 10 days were logged
  S24         Lomb-Scargle share of variance explained by a 24.0-h sinusoid in that window
  plateau     median S24 over windows centred on days 120-240 (needs >= 28 such windows)
  t50         first day of age at which S24 reaches half the infant's own plateau and stays there:
              >= 80% of the valid windows in the next 14 days also at or above it (>= 7 windows)
  t_abs       the same rule with an absolute threshold (ABS_THRESHOLD)
  censoring   'left' if the first usable window already meets the rule (emergence at or before it),
              'right' if it is never met by day 365
  eligible    infants whose first usable window is centred on day <= 21 and whose plateau
              is >= MIN_PLATEAU (half of a near-zero plateau would be meaningless)
  free-run    >= 14 consecutive windows whose 20-28 h peak is >= 0.75 h from 24 h, explains >= 5%
              of variance and is >= 1.5x the power at 24 h, all on the same side of 24 h
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from rhythm import lomb_scargle_windows, ls_basis

PER_DAY, WIN, MIN_DAYS = 144, 14, 10
HALF = WIN // 2
PERIODS = np.unique(np.r_[np.geomspace(2, 30, 200), np.arange(20, 28.001, 0.1), 24.0])
I24 = int(np.argmin(np.abs(PERIODS - 24.0)))
CIRC = np.flatnonzero((PERIODS >= 20) & (PERIODS <= 28))
ULTRA = np.flatnonzero((PERIODS >= 2) & (PERIODS <= 8))
T_H = (np.arange(WIN * PER_DAY) + 0.5) / 6
BASIS = ls_basis(T_H, PERIODS)
PLATEAU_ROWS = slice(119, 240)          # windows centred on days 120..240 (row 0 = day 1)
ABS_THRESHOLD = 0.20                    # set after inspecting the plateau distribution
ELIGIBLE_MAX_FIRST_DAY = 21
NOTES = {446: "sleep timing shifted ~10-14 h for ~2 weeks (days ~76-92) after a 2-day logging gap, likely travel logged on home time; produces a spurious ~25-h periodogram peak"}
MIN_PLATEAU = 0.10                      # t50 is meaningless without a real plateau


def analyse_infant(day_by_bin: np.ndarray) -> dict:
    """day_by_bin: (n_days, 144) share asleep per bin, NaN rows = unlogged days.
    Returns per-day arrays (index = row of the window centre), NaN where no usable window."""
    n_days = day_by_bin.shape[0]
    series = day_by_bin.reshape(-1)
    W = sliding_window_view(series, WIN * PER_DAY)[::PER_DAY]          # (n_days - 13, 2016)
    days_logged = (~np.isnan(W)).sum(1) / PER_DAY
    ok = days_logged >= MIN_DAYS
    out = {k: np.full(n_days, np.nan) for k in
           ("s24", "circ_period", "circ_power", "ultra_period", "ultra_power")}
    if ok.any():
        P = lomb_scargle_windows(W[ok], T_H, PERIODS, basis=BASIS)
        rows = np.flatnonzero(ok) + HALF
        ic = CIRC[np.nanargmax(P[:, CIRC], axis=1)]
        iu = ULTRA[np.nanargmax(P[:, ULTRA], axis=1)]
        r = np.arange(len(rows))
        out["s24"][rows] = P[:, I24]
        out["circ_period"][rows], out["circ_power"][rows] = PERIODS[ic], P[r, ic]
        out["ultra_period"][rows], out["ultra_power"][rows] = PERIODS[iu], P[r, iu]
    return out


def crossing(s: np.ndarray, threshold: float, start: int, stop: int = 365):
    """First row >= start where s >= threshold and stays there (rule in module docstring)."""
    for a in range(start, min(stop, s.size)):
        if np.isnan(s[a]) or s[a] < threshold:
            continue
        nxt = s[a:a + WIN]
        nxt = nxt[~np.isnan(nxt)]
        if nxt.size >= 7 and np.mean(nxt >= threshold) >= 0.8:
            return a
    return None


def emergence(s24: np.ndarray) -> dict:
    valid = np.flatnonzero(~np.isnan(s24))
    res = dict(first_window_day=np.nan, plateau_s24=np.nan, eligible=False,
               t50_day=np.nan, t50_censoring="", t_abs_day=np.nan, t_abs_censoring="")
    if valid.size == 0:
        return res
    a0 = int(valid[0])
    res["first_window_day"] = a0 + 1
    plat = s24[PLATEAU_ROWS]
    if np.sum(~np.isnan(plat)) >= 28:
        res["plateau_s24"] = float(np.nanmedian(plat))
    res["eligible"] = bool(a0 + 1 <= ELIGIBLE_MAX_FIRST_DAY
                           and res["plateau_s24"] >= MIN_PLATEAU)
    if not res["eligible"]:
        return res
    for key, thr in (("t50", 0.5 * res["plateau_s24"]), ("t_abs", ABS_THRESHOLD)):
        a = crossing(s24, thr, a0)
        if a is None:
            res[f"{key}_censoring"] = "right"
        else:
            res[f"{key}_day"] = a + 1
            res[f"{key}_censoring"] = "left" if a == a0 else "none"
    return res


def free_running(out: dict, min_len: int = 14) -> dict:
    p, pw, p24 = out["circ_period"], out["circ_power"], out["s24"]
    with np.errstate(invalid="ignore"):
        flag = (np.abs(p - 24) >= 0.75) & (pw >= 0.05) & (pw >= 1.5 * p24)
    side = np.sign(p - 24)
    best = dict(fr_episode=False, fr_start_day=np.nan, fr_end_day=np.nan, fr_period_h=np.nan,
                fr_power=np.nan)
    a, n = 0, flag.size
    while a < n:
        if not flag[a]:
            a += 1
            continue
        b = a
        while b + 1 < n and flag[b + 1] and side[b + 1] == side[a]:
            b += 1
        if b - a + 1 >= min_len and (b - a + 1) > (best["fr_end_day"] - best["fr_start_day"] + 1
                                                    if best["fr_episode"] else 0):
            best = dict(fr_episode=True, fr_start_day=a + 1, fr_end_day=b + 1,
                        fr_period_h=float(np.median(p[a:b + 1])),
                        fr_power=float(np.median(pw[a:b + 1])))
        a = b + 1
    return best


def main(mat_path: str, out_dir: str):
    from scipy.io import loadmat
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = loadmat(mat_path)["data"]                                      # (730, 144, 700)
    n_days, _, n_inf = d.shape
    keys = ("s24", "circ_period", "circ_power", "ultra_period", "ultra_power")
    saved = out_dir / "nyu_rhythm_windows.npz"
    reuse = len(sys.argv) > 3 and sys.argv[3] == "reuse" and saved.exists()
    if reuse:
        z = np.load(saved)
        arrays = {k: z[k] for k in keys}
    else:
        arrays = {k: np.full((n_inf, n_days), np.nan, dtype=np.float32) for k in keys}
    rows = []
    for i in range(n_inf):
        x = d[:, :, i]
        out = {k: arrays[k][i].astype(float) for k in keys} if reuse else analyse_infant(x)
        for k in arrays:
            arrays[k][i] = out[k]
        logged = np.flatnonzero(~np.isnan(x).all(1))
        rec = dict(infant=i + 1, days_logged=logged.size,
                   first_logged_day=int(logged[0]) + 1 if logged.size else np.nan,
                   last_logged_day=int(logged[-1]) + 1 if logged.size else np.nan)
        rec.update(emergence(out["s24"]))
        for wk in (2, 4, 8, 12, 26, 52):
            rec[f"s24_week{wk}"] = out["s24"][wk * 7 - 1]
        rec.update(free_running(out))
        rec["note"] = NOTES.get(i + 1, "")
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} infants done", flush=True)

    np.savez_compressed(out_dir / "nyu_rhythm_windows.npz", periods=PERIODS, **arrays)
    summ = pd.DataFrame(rows)
    summ.round(4).to_csv(out_dir / "nyu_infant_rhythm_summary.csv", index=False)

    weekly = []
    for i in range(n_inf):
        s, cp = arrays["s24"][i], arrays["circ_period"][i]
        for w in range(n_days // 7):
            seg = slice(7 * w, 7 * w + 7)
            k = np.sum(~np.isnan(s[seg]))
            if k >= 4:
                weekly.append((i + 1, w + 1, round(float(np.nanmean(s[seg])), 4),
                               round(float(np.nanmedian(cp[seg])), 2), int(k)))
    pd.DataFrame(weekly, columns=["infant", "week_of_age", "s24_mean", "circadian_peak_period_h",
                                  "n_windows"]).to_csv(out_dir / "nyu_infant_weekly_rhythm.csv",
                                                       index=False)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
