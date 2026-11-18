"""Screen the NYU records for sudden shifts in sleep timing at several thresholds, and estimate how
many such episodes each infant's ordinary day-to-day variation would produce on its own.

Usage: python3 -I jetlag_screen.py <results_dir> <out_dir> [n_surrogates]

Reads <results_dir>/nyu/nyu_timing_arrays.npz (daily sleep phase and its concentration R, from
nyu_sleep_timing.py). Writes to <out_dir>:
  episodes_<T>h.csv     the episodes found at threshold T hours (same detector as the main paper)
  screen_counts.csv     per threshold: episodes and infants found, and the count expected from
                        ordinary day-to-day variation (mean and 95% range over surrogate cohorts),
                        for all ages and for episodes starting at 90 days of age or later

Null model (block bootstrap). For each infant, days in or next to an episode found at the threshold
being tested (2 days before to 7 days after) are set aside. The remaining usable days give a smooth
trend (circular mean of phase within +-7 days, widened when data are sparse) and residuals around it.
A surrogate record keeps the infant's trend, logged days and R, and replaces each day's residual by
residuals resampled in blocks of 7 consecutive days, so that runs of unusual days are kept as they
occur. The detector is then run on the surrogate. Episodes found in surrogates are what ordinary
variation alone produces at that threshold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyu_sleep_timing as st  # noqa: E402

THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0]
BLOCK = 7
MIN_AGE = 90
SEED = 20261118


def detect(phase, R, thr):
    st.THR_H = thr
    return st.detect_jumps(phase, R)


def usable_days(phase, R):
    return ~np.isnan(phase) & (R >= st.R_MIN) & (np.arange(phase.size) + 1 >= st.AGE_MIN)


def trend(phase, clean, days):
    """Circular mean of clean-day phases within +-7 days of each day in `days` (widened if sparse)."""
    idx = np.flatnonzero(clean)
    out = np.full(phase.size, np.nan)
    if idx.size == 0:
        return out
    a = phase[idx] * 2 * np.pi / 24
    c, s = np.cos(a), np.sin(a)
    for d in days:
        for half in (7, 14, 28, 60, 730):
            m = np.abs(idx - d) <= half
            if m.sum() >= 4 or half == 730:
                out[d] = (np.arctan2(s[m].mean(), c[m].mean()) % (2 * np.pi)) * 24 / (2 * np.pi)
                break
    return out


def surrogate_counts(phase, R, thr, eps, n_sur, rng):
    """Episodes found in n_sur block-bootstrap surrogates of one infant's record: counts for all
    ages and for episodes starting at MIN_AGE days or later."""
    use = usable_days(phase, R)
    days = np.flatnonzero(use)
    none = np.zeros(n_sur, dtype=int)
    if days.size < 30:
        return none, none.copy()
    clean = use.copy()
    for e in eps:
        clean[max(e["start_day"] - 1 - 2, 0):e["end_day"] - 1 + 8] = False
    if clean.sum() < 2 * BLOCK:
        return none, none.copy()
    m = trend(phase, clean, days)
    pool = st.wrap(phase[clean] - m[clean])
    counts, counts_old = np.zeros(n_sur, dtype=int), np.zeros(n_sur, dtype=int)
    for k in range(n_sur):
        res = []
        while len(res) < days.size:
            j = rng.integers(0, pool.size - BLOCK + 1)
            res.extend(pool[j:j + BLOCK])
        sur = np.full(phase.size, np.nan)
        sur[days] = (m[days] + np.asarray(res[:days.size])) % 24
        found = detect(sur, R, thr)
        counts[k] = len(found)
        counts_old[k] = sum(e["start_day"] >= MIN_AGE for e in found)
    return counts, counts_old


def main(res_dir, out_dir, n_sur=20):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(Path(res_dir) / "nyu" / "nyu_timing_arrays.npz")
    phase, R = z["phase"].astype(float), z["R"].astype(float)
    rng = np.random.default_rng(SEED)
    rows = []
    for thr in THRESHOLDS:
        eps_all, null, null_old = [], np.zeros(n_sur, dtype=int), np.zeros(n_sur, dtype=int)
        for i in range(phase.shape[0]):
            eps = detect(phase[i], R[i], thr)
            for e in eps:
                eps_all.append({"infant": i + 1, **e})
            c, c_old = surrogate_counts(phase[i], R[i], thr, eps, n_sur, rng)
            null += c
            null_old += c_old
        df = pd.DataFrame(eps_all)
        df.to_csv(out / f"episodes_{thr:g}h.csv", index=False)
        old = df[df["start_day"] >= MIN_AGE]
        rows.append({"threshold_h": thr, "episodes": len(df), "infants": df["infant"].nunique(),
                     "positive": int((df["shift_h"] > 0).sum()), "negative": int((df["shift_h"] < 0).sum()),
                     "null_mean": round(float(null.mean()), 1),
                     "null_lo95": int(np.percentile(null, 2.5)), "null_hi95": int(np.percentile(null, 97.5)),
                     "episodes_from90": len(old), "infants_from90": old["infant"].nunique(),
                     "positive_from90": int((old["shift_h"] > 0).sum()),
                     "negative_from90": int((old["shift_h"] < 0).sum()),
                     "null_from90_mean": round(float(null_old.mean()), 2),
                     "null_from90_lo95": int(np.percentile(null_old, 2.5)),
                     "null_from90_hi95": int(np.percentile(null_old, 97.5)),
                     "n_surrogates": n_sur})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "screen_counts.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
