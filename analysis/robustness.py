"""Robustness checks for the emergence analysis of the NYU cohort.

Usage: python3 -I robustness.py <infant_sleep_preprocessed.mat> <nyu_results_dir> <out_dir>
Needs nyu_infant_rhythm_summary.csv and nyu_rhythm_windows.npz from nyu_rhythm_analysis.py.

1. Early 24-h specificity. In each eligible infant's first usable window: power at 24 h against
   the median power at 20-22 h and 26-28 h, and the period of the strongest 20-28 h peak.
2. Surrogate test. Is the 24-h component of a window stronger than when every logged day is
   rotated by its own random clock offset? Rotation keeps each day's amount of sleep and its bout
   structure but removes any phase shared across days. 199 surrogates per window, for the first
   usable window and the windows centred on days 14, 28, 56 and 84.
3. Definition sensitivity. Half-plateau age with thresholds of 40, 50 and 60% of the plateau and
   plateau windows of days 90-180, 120-240 and 150-300 (from the saved 24-h strengths), and with
   10-day instead of 14-day periodogram windows (periodograms recomputed).
4. Free-running detection power. From day 35 the infant's own record is time-stretched so that its
   24-h pattern repeats every tau hours for D days; the record is then re-analysed with the
   unchanged free-running screen.

Outputs in <out_dir>: first_window_specificity.csv, surrogate_24h.csv,
definition_sensitivity.csv, window_length_t50.csv, free_running_power.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

import nyu_rhythm_analysis as nra
from rhythm import lomb_scargle_windows, ls_basis

SEED = 20261118
N_SURR = 199
SURR_WINDOWS = {"first": None, "day 14": 14, "day 28": 28, "day 56": 56, "day 84": 84}
FLANK = np.flatnonzero(((nra.PERIODS >= 20) & (nra.PERIODS <= 22)) |
                       ((nra.PERIODS >= 26) & (nra.PERIODS <= 28)))
P24 = np.array([24.0])
B24 = ls_basis(nra.T_H, P24)
FR_START, FR_DAYS = 35, 140
TAUS, DURATIONS = (24.5, 25.0, 25.5, 26.0), (14, 28, 42, 56)


def window(x, centre_day):
    """(14, 144) rows of the window centred on `centre_day` (convention of analyse_infant)."""
    r = centre_day - 1 - nra.HALF
    if r < 0 or r + nra.WIN > x.shape[0]:
        return None
    w = x[r:r + nra.WIN]
    return w if (~np.isnan(w)).sum() / nra.PER_DAY >= nra.MIN_DAYS else None


def s24(rows):
    return lomb_scargle_windows(rows, nra.T_H, P24, basis=B24)[:, 0]


def surrogate_p(w, rng):
    """Observed 24-h power and its rank-based p value against day-rotation surrogates."""
    shifts = rng.integers(0, nra.PER_DAY, size=(N_SURR, nra.WIN))
    idx = (np.arange(nra.PER_DAY)[None, None, :] - shifts[:, :, None]) % nra.PER_DAY
    sur = w[np.arange(nra.WIN)[None, :, None], idx].reshape(N_SURR, -1)
    obs = s24(w.reshape(1, -1))[0]
    null = s24(sur)
    return obs, (1 + np.sum(null >= obs)) / (N_SURR + 1), float(np.median(null))


def stretch(x, start_day, n_days, tau):
    """Time-stretch the record from `start_day` so its 24-h pattern repeats every `tau` hours."""
    flat = x.reshape(-1)
    s = flat.copy()
    a, n = (start_day - 1) * nra.PER_DAY, n_days * nra.PER_DAY
    s[a:a + n] = flat[a + np.floor(np.arange(n) * 24.0 / tau).astype(int)]
    return s.reshape(x.shape)


def summarise(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return dict(n=v.size, median=np.median(v), q1=np.quantile(v, 0.25), q3=np.quantile(v, 0.75))


def main(mat_path, nyu_dir, out_dir):
    nyu_dir, out_dir = Path(nyu_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = loadmat(mat_path)["data"]                                  # (730, 144, 700)
    summ = pd.read_csv(nyu_dir / "nyu_infant_rhythm_summary.csv")
    elig = summ[summ.eligible].copy()
    S24 = np.load(nyu_dir / "nyu_rhythm_windows.npz")["s24"]          # (700, 730)
    rng = np.random.default_rng(SEED)

    # 1-2. early specificity and surrogate test
    spec, surr = [], []
    for _, r in elig.iterrows():
        i, x = int(r.infant), data[:, :, int(r.infant) - 1]
        w = window(x, int(r.first_window_day))
        P = lomb_scargle_windows(w.reshape(1, -1), nra.T_H, nra.PERIODS, basis=nra.BASIS)[0]
        spec.append(dict(infant=i, first_window_day=int(r.first_window_day), s24=P[nra.I24],
                         flank_ratio=P[nra.I24] / np.median(P[FLANK]),
                         circ_peak_h=nra.PERIODS[nra.CIRC[np.nanargmax(P[nra.CIRC])]]))
        for label, day in SURR_WINDOWS.items():
            day = int(r.first_window_day) if day is None else day
            w = window(x, day)
            if w is None:
                continue
            obs, p, null_med = surrogate_p(w, rng)
            surr.append(dict(infant=i, window=label, centre_day=day, s24=obs,
                             surrogate_median=null_med, p=p))
    spec, surr = pd.DataFrame(spec), pd.DataFrame(surr)
    spec.round(5).to_csv(out_dir / "first_window_specificity.csv", index=False)
    surr.round(5).to_csv(out_dir / "surrogate_24h.csv", index=False)
    print("first window: flank ratio", summarise(spec.flank_ratio),
          f"share > 2: {np.mean(spec.flank_ratio > 2):.3f}",
          f"peak within 0.5 h of 24: {np.mean(np.abs(spec.circ_peak_h - 24) <= 0.5 + 1e-9):.3f}")
    print(surr.groupby("window", sort=False).agg(n=("p", "size"), share_p05=("p", lambda p: np.mean(p < 0.05)),
                                                 median_s24=("s24", "median")).round(3))

    # 3a. threshold and plateau-window sensitivity from the saved 24-h strengths
    rows = []
    for (pa, pb) in ((90, 180), (120, 240), (150, 300)):
        for frac in (0.4, 0.5, 0.6):
            t = []
            for i in range(S24.shape[0]):
                s = S24[i]
                valid = np.flatnonzero(~np.isnan(s))
                plat = s[pa - 1:pb]
                if valid.size == 0 or np.sum(~np.isnan(plat)) < 28:
                    continue
                plat = float(np.nanmedian(plat))
                if valid[0] + 1 > nra.ELIGIBLE_MAX_FIRST_DAY or plat < nra.MIN_PLATEAU:
                    continue
                a = nra.crossing(s, frac * plat, int(valid[0]))
                t.append(np.nan if a is None else a + 1)
            t = np.array(t, float)
            rows.append(dict(plateau_days=f"{pa}-{pb}", threshold=frac, n_eligible=t.size,
                             n_reached=int(np.sum(~np.isnan(t))), **{k: v for k, v in summarise(t).items() if k != "n"}))
    sens = pd.DataFrame(rows)
    sens.round(2).to_csv(out_dir / "definition_sensitivity.csv", index=False)
    print(sens.round(1).to_string(index=False))

    # 3b. 10-day windows (only the window length changes; the hold rule stays 14 windows)
    saved = (nra.WIN, nra.HALF, nra.MIN_DAYS, nra.T_H, nra.BASIS)
    nra.WIN, nra.HALF, nra.MIN_DAYS = 10, 5, 7
    nra.T_H = (np.arange(nra.WIN * nra.PER_DAY) + 0.5) / 6
    nra.BASIS = ls_basis(nra.T_H, nra.PERIODS)
    s10 = {int(r.infant): nra.analyse_infant(data[:400, :, int(r.infant) - 1])["s24"]
           for _, r in elig.iterrows()}
    nra.WIN, nra.HALF, nra.MIN_DAYS, nra.T_H, nra.BASIS = saved
    wl = []
    for _, r in elig.iterrows():
        s = s10[int(r.infant)]
        valid = np.flatnonzero(~np.isnan(s))
        plat = float(np.nanmedian(s[nra.PLATEAU_ROWS]))
        a = nra.crossing(s, 0.5 * plat, int(valid[0]))
        wl.append(dict(infant=int(r.infant), t50_14d=r.t50_day, t50_10d=np.nan if a is None else a + 1,
                       plateau_14d=r.plateau_s24, plateau_10d=plat))
    wl = pd.DataFrame(wl)
    wl.round(4).to_csv(out_dir / "window_length_t50.csv", index=False)
    ok = wl[["t50_14d", "t50_10d"]].notna().all(axis=1)
    print("10-day windows:", summarise(wl.t50_10d), "rho with 14-day:",
          round(stats.spearmanr(wl.t50_14d[ok], wl.t50_10d[ok])[0], 3),
          "median paired difference:", np.median(wl.t50_10d[ok] - wl.t50_14d[ok]))

    # 4. free-running detection power
    det = []
    for _, r in elig.iterrows():
        i, x = int(r.infant), data[:FR_DAYS, :, int(r.infant) - 1]
        base = nra.free_running(nra.analyse_infant(x))
        det.append(dict(infant=i, tau_h=24.0, duration_days=0, detected=base["fr_episode"],
                        detected_period_h=base["fr_period_h"]))
        for tau in TAUS:
            for d in DURATIONS:
                fr = nra.free_running(nra.analyse_infant(stretch(x, FR_START, d, tau)))
                hit = bool(fr["fr_episode"] and fr["fr_period_h"] > 24
                           and fr["fr_start_day"] <= FR_START + d and fr["fr_end_day"] >= FR_START)
                det.append(dict(infant=i, tau_h=tau, duration_days=d, detected=hit,
                                detected_period_h=fr["fr_period_h"] if hit else np.nan))
    det = pd.DataFrame(det)
    power = det.groupby(["tau_h", "duration_days"]).agg(
        n_infants=("detected", "size"), n_detected=("detected", "sum"),
        sensitivity=("detected", "mean"), median_detected_period_h=("detected_period_h", "median")).reset_index()
    power.round(3).to_csv(out_dir / "free_running_power.csv", index=False)
    print(power.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
