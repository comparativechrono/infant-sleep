"""Day-by-day profile of each sudden shift in sleep timing, and summary measures of re-alignment.

Usage: python3 -I jetlag_profiles.py <infant_sleep_preprocessed.mat> <results_dir> [threshold_h]

Reads results/jetlag/episodes_<T>h.csv (jetlag_screen.py), results/nyu/nyu_timing_arrays.npz and
results/nyu/nyu_infant_rhythm_summary.csv. Writes to results/jetlag/:
  episode_metrics_<T>h.csv   one row per episode found at threshold T, with the measures below
  trajectories_<T>h.csv      per episode and day: deviation from the baseline trend, aligned on the
                        first day of the episode (onset) and on its last day (end)

Main set: episodes starting at 90 days of age or later (when the 24-hour rhythm is established in
most infants and the daily sleep phase is well defined) and shifts below 11 h (a ~12-h shift has no
direction).

Deviation. The day's sleep phase (circular mean clock time of the day's sleep, from
nyu_sleep_timing.py) minus a baseline trend, multiplied by the sign of the shift so that the shift is
positive. The baseline trend is a straight line fitted to the deviations from the detector's
baseline phase on usable days 28 to 1 before onset and 29 to 56 after the episode's last day, so
that slow developmental drift is not mistaken for an after-effect. Only usable days (logged,
R >= 0.15) count.
  shift_h        mean deviation on the episode's first 2 usable days
  prev_h         deviation on the last usable day within 3 days before onset
  abrupt_onset   prev_h is below half of shift_h (the shift appeared from one logged day to the next)
  last_h         mean deviation on the episode's last 2 usable days
  change_h       last_h - shift_h (negative: the shift shrank while it lasted)
  frac_left      last_h / shift_h, the share of the shift still present at the episode's end
  tau_d          time constant of an exponential decline from shift_h to last_h over the days
                 between the first 2 and last 2 usable episode days (inf: no decline)
  rate_h_per_day (shift_h - last_h) / those days
  next_h         deviation on the first usable day within 5 days after the episode
  abrupt_end     last_h - next_h is at least half of shift_h
  resid_<a>_<b>  mean deviation on days a to b after the episode's last day
  pre_sd_h       SD of the deviation on the 14 days before onset (ordinary day-to-day scatter)
  unlogged_before / unlogged_after   unlogged days among the 3 days before onset / after the end
  sleep_pre_h, sleep_first3_h        hours asleep per logged day: 14 days before; first 3 days
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyu_sleep_timing as st  # noqa: E402

MIN_AGE, MAX_SHIFT = 90, 11.0
ONSET_WIN = range(-14, 22)          # days relative to the first episode day
END_WIN = range(-7, 29)             # days relative to the last episode day
RESID = [(1, 3), (4, 10), (11, 17), (18, 28)]


def deviation(phase, R, start_day, end_day, shift_h, baseline_h):
    """Sign-normalised deviation from the baseline trend for every day, and the trend fit itself."""
    a, b = start_day - 1, end_day - 1
    n = phase.size
    use = ~np.isnan(phase) & (R >= st.R_MIN)
    raw = np.where(use, np.sign(shift_h) * st.wrap(phase - baseline_h), np.nan)
    x = np.r_[np.arange(a - 28, a), np.arange(b + 29, b + 57)]
    x = x[(x >= 0) & (x < n)]
    ok = ~np.isnan(raw[x])
    if ok.sum() >= 10 and (~np.isnan(raw[b + 29:b + 57])).sum() >= 5:
        coef = np.polyfit(x[ok], raw[x][ok], 1)
    else:                                   # too little data after: flat baseline from before
        pre = raw[max(a - 28, 0):a]
        coef = np.array([0.0, np.nanmean(pre) if (~np.isnan(pre)).any() else 0.0])
    return raw - np.polyval(coef, np.arange(n)), coef


def measures(dev, start_day, end_day):
    a, b = start_day - 1, end_day - 1
    n = dev.size
    ok = lambda k: 0 <= k < n and not np.isnan(dev[k])
    ep = [k for k in range(a, b + 1) if ok(k)]
    prev = [k for k in range(a - 3, a) if ok(k)]
    nxt = next((k for k in range(b + 1, b + 6) if ok(k)), None)
    shift = float(np.mean(dev[ep[:2]]))
    last = float(np.mean(dev[ep[-2:]]))
    seg = lambda lo, hi: (float(np.nanmean(dev[b + lo:b + hi + 1]))
                          if any(ok(k) for k in range(b + lo, b + hi + 1)) else np.nan)
    pre = [dev[k] for k in range(a - 14, a) if ok(k)]
    dt = float(np.mean(ep[-2:]) - np.mean(ep[:2]))
    ratio = last / shift
    tau = np.inf if ratio >= 1 else (dt / -np.log(ratio) if ratio > 0 else 0.0)
    m = dict(shift_h=shift, prev_h=float(dev[prev[-1]]) if prev else np.nan,
             last_h=last, change_h=last - shift, frac_left=ratio, tau_d=tau,
             rate_h_per_day=(shift - last) / dt if dt > 0 else np.nan,
             next_h=float(dev[nxt]) if nxt is not None else np.nan,
             pre_sd_h=float(np.std(pre)) if len(pre) > 2 else np.nan)
    m["abrupt_onset"] = bool(m["prev_h"] < shift / 2) if prev else None
    m["abrupt_end"] = bool(last - m["next_h"] >= shift / 2) if nxt is not None else None
    for lo, hi in RESID:
        m[f"resid_{lo}_{hi}"] = seg(lo, hi)
    return m


def main(mat, res_dir, thr=3.0):
    from scipy.io import loadmat
    res = Path(res_dir)
    out = res / "jetlag"
    eps = pd.read_csv(out / f"episodes_{thr:g}h.csv").sort_values(["infant", "start_day"]).reset_index(drop=True)
    z = np.load(res / "nyu" / "nyu_timing_arrays.npz")
    phase, R = z["phase"].astype(float), z["R"].astype(float)
    data = loadmat(mat)["data"]
    rsum = pd.read_csv(res / "nyu" / "nyu_infant_rhythm_summary.csv").set_index("infant")
    rows, trajs = [], []
    for k, e in eps.iterrows():
        i = int(e["infant"]) - 1
        dev, coef = deviation(phase[i], R[i], int(e["start_day"]), int(e["end_day"]), e["shift_h"],
                              e["baseline_phase_h"])
        m = measures(dev, int(e["start_day"]), int(e["end_day"]))
        a, b = int(e["start_day"]) - 1, int(e["end_day"]) - 1
        x = data[:, :, i]
        logged = ~np.isnan(x).all(1)
        sleep_h = np.nansum(x, axis=1) * 10 / 60
        pre_idx = [d for d in range(max(a - 14, 0), a) if logged[d]]
        ep3 = [d for d in range(a, min(a + 3, 730)) if logged[d]]
        main_set = (e["start_day"] >= MIN_AGE) and abs(e["shift_h"]) < MAX_SHIFT
        rows.append({"episode": k + 1, "infant": int(e["infant"]), "start_day": int(e["start_day"]),
                     "end_day": int(e["end_day"]), "span_days": int(e["span_days"]),
                     "detector_shift_h": e["shift_h"], "direction": "later" if e["shift_h"] > 0 else "earlier",
                     "main_set": main_set, "trend_slope_h_per_day": round(float(coef[0]), 4),
                     **{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in m.items()},
                     "unlogged_before": int((~logged[max(a - 3, 0):a]).sum()),
                     "unlogged_after": int((~logged[b + 1:b + 4]).sum()),
                     "sleep_pre_h": round(float(np.mean(sleep_h[pre_idx])), 2) if pre_idx else np.nan,
                     "sleep_first3_h": round(float(np.mean(sleep_h[ep3])), 2) if ep3 else np.nan,
                     "plateau_s24": rsum.loc[int(e["infant"]), "plateau_s24"],
                     "t50_day": rsum.loc[int(e["infant"]), "t50_day"]})
        for t in ONSET_WIN:
            d = a + t
            trajs.append({"episode": k + 1, "align": "onset", "day": t,
                          "deviation_h": round(float(dev[d]), 3) if 0 <= d < 730 and not np.isnan(dev[d]) else np.nan})
        for t in END_WIN:
            d = b + t
            trajs.append({"episode": k + 1, "align": "end", "day": t,
                          "deviation_h": round(float(dev[d]), 3) if 0 <= d < 730 and not np.isnan(dev[d]) else np.nan})
    M = pd.DataFrame(rows)
    M.to_csv(out / f"episode_metrics_{thr:g}h.csv", index=False)
    pd.DataFrame(trajs).to_csv(out / f"trajectories_{thr:g}h.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["episode", "infant", "start_day", "span_days", "detector_shift_h", "main_set", "shift_h", "prev_h",
            "abrupt_onset", "change_h", "abrupt_end", "resid_1_3", "resid_4_10", "resid_11_17", "resid_18_28",
            "pre_sd_h", "unlogged_before", "unlogged_after", "sleep_pre_h", "sleep_first3_h"]
    print(M[cols].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3.0)
