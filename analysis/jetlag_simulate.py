"""Test the shift detector and the re-alignment measures on simulated shifts of known form.

Usage: python3 -I jetlag_simulate.py <infant_sleep_preprocessed.mat> <results_dir> [reps]

A simulated shift is written into a real record: from a chosen day, every 10-minute bin of the
logged sleep is moved later (or earlier) in clock time by s(t) hours, where t is days since onset:
  during the shift (t < L days)     s(t) = D * exp(-t / tau)      (tau = inf: no adjustment)
  afterwards (t >= L)               s(t) = rho * D * exp(-(t - L) / 7)
so D is the size of the shift, L how long it lasts, tau how fast sleep re-aligns while it lasts,
and rho the size of a residual offset that fades after it ends. Host records are infants with at
least 80% usable days from 28 days before onset to 56 days after the shift ends, onset at day 120
or later, and no shift of 2 h or more found anywhere in that span. The detector (3-h threshold) and
the measures of jetlag_profiles.py are then applied exactly as to the real data.

Writes results/jetlag/simulation_runs.csv (one row per simulated shift) and
results/jetlag/simulation_summary.csv (one row per condition).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyu_sleep_timing as st  # noqa: E402
from jetlag_profiles import deviation, measures  # noqa: E402

THR = 3.0
SIZES = [3.0, 4.0, 5.0, 6.0]
LENGTHS = [4, 7, 14]
TAUS = [np.inf, 10.0, 4.0]
RHOS = [0.0, 0.25]
SEED = 20261119


def shifted(x, t0, s_by_day):
    """Copy of x (days x 144) with day t0 + t moved by s_by_day[t] hours; a day whose source bins
    are not all logged becomes unlogged."""
    flat = x.reshape(-1)
    y = x.copy()
    for t, s in enumerate(s_by_day):
        d = t0 + t
        if d >= x.shape[0]:
            break
        k = int(round(s * 6))
        src = np.arange(d * 144, (d + 1) * 144) - k
        if src.min() < 0 or src.max() >= flat.size or np.isnan(flat[src]).any():
            y[d] = np.nan
        else:
            y[d] = flat[src]
    return y


def main(mat, res_dir, reps=30):
    from scipy.io import loadmat
    res = Path(res_dir)
    out = res / "jetlag"
    data = loadmat(mat)["data"]
    z = np.load(res / "nyu" / "nyu_timing_arrays.npz")
    phase0, R0 = z["phase"].astype(float), z["R"].astype(float)
    use0 = ~np.isnan(phase0) & (R0 >= st.R_MIN)
    eps2 = pd.read_csv(out / "episodes_2h.csv")
    busy = np.zeros(use0.shape, dtype=bool)
    for _, e in eps2.iterrows():
        busy[int(e["infant"]) - 1, max(int(e["start_day"]) - 1 - 7, 0):int(e["end_day"]) + 7] = True
    rng = np.random.default_rng(SEED)
    st.THR_H = THR
    rows = []
    for D in SIZES:
        for L in LENGTHS:
            for tau in TAUS:
                for rho in RHOS:
                    for rep in range(reps):
                        for _ in range(1000):                       # find a host record and onset
                            i = int(rng.integers(0, data.shape[2]))
                            t0 = int(rng.integers(119, 730 - L - 57))
                            lo, hi = t0 - 28, t0 + L + 56
                            if use0[i, lo:hi].mean() >= 0.8 and not busy[i, lo:hi].any():
                                break
                        sign = 1.0 if rng.random() < 0.5 else -1.0
                        t = np.arange(L + 56)
                        s = np.where(t < L, D * np.exp(-t / tau), rho * D * np.exp(-(t - L) / 7.0))
                        x = shifted(data[:, :, i], t0, sign * s)
                        ph, R = st.daily_phase(x)
                        found = [e for e in st.detect_jumps(ph, R)
                                 if t0 <= e["start_day"] - 1 <= t0 + 2 and np.sign(e["shift_h"]) == sign]
                        row = dict(size_h=D, length_d=L, true_tau_d=tau, rho=rho, infant=i + 1, onset_day=t0 + 1,
                                   detected=bool(found))
                        if found:
                            e = found[0]
                            dev, _ = deviation(ph, R, e["start_day"], e["end_day"], e["shift_h"],
                                               e["baseline_phase_h"])
                            row.update(span_days=e["span_days"], **measures(dev, e["start_day"], e["end_day"]))
                        rows.append(row)
                    print(f"D={D} L={L} tau={tau} rho={rho}: detected "
                          f"{np.mean([r['detected'] for r in rows[-reps:]]):.2f}", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "simulation_runs.csv", index=False)
    det = runs[runs["detected"]]
    g = runs.groupby(["size_h", "length_d", "true_tau_d", "rho"])
    summ = g["detected"].mean().rename("detected_share").to_frame()
    gd = det.groupby(["size_h", "length_d", "true_tau_d", "rho"])
    for c in ["shift_h", "change_h", "frac_left", "resid_1_3", "resid_4_10", "resid_11_17", "resid_18_28", "span_days"]:
        summ[f"{c}_mean"] = gd[c].mean()
    summ["abrupt_onset_share"] = gd["abrupt_onset"].apply(lambda v: np.mean([bool(x) for x in v.dropna()]))
    summ["abrupt_end_share"] = gd["abrupt_end"].apply(lambda v: np.mean([bool(x) for x in v.dropna()]))
    summ = summ.round(3).reset_index()
    summ.to_csv(out / "simulation_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
