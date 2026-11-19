"""The infant screen, null model, episode measures and simulations, applied to adult sleep records.

Usage: python3 -I adult_screen.py <results_dir> [n_surrogates] [reps]

Reads <results_dir>/adult/adult_records.npz (adult_data.py) and, for the comparison table,
<results_dir>/jetlag/episode_metrics_3h.csv (the infants). Writes to <results_dir>/adult/:
  adult_phase.npz          per adult: nightly sleep phase and R (same definitions as the infants)
  episodes_<T>h.csv        episodes found at threshold T (same detector; no age restriction)
  screen_counts.csv        per threshold: episodes found and expected from day-to-day variation
                           (block bootstrap, as jetlag_screen.py)
  episode_metrics_3h.csv   the measures of jetlag_profiles.py for each 3-h episode
  simulation_runs.csv, simulation_summary.csv
                           the simulated shifts of jetlag_simulate.py written into adult records
  Adults are labelled A01-A30 in the order of adult_records.npz; outputs carry no file ids or dates.
  trips.csv                Oura records, where each night's UTC offset is known: every change of
                           the clock by 2.5 h or more, with what the screen found and how much of
                           the change appeared in local-time sleep timing
  trip_trajectories.csv    for each trip, the night-by-night deviation of local-time sleep phase
                           from its pre-trip baseline, signed so that a lag behind the new clock
                           is positive

Rows are nights from 12:00 to 12:00, so phases are computed after rolling each row to start at
00:00; nothing else differs from the infant analysis except that there is no minimum age.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyu_sleep_timing as st  # noqa: E402
import jetlag_screen as js  # noqa: E402
from jetlag_profiles import deviation, measures  # noqa: E402
from jetlag_simulate import shifted, SIZES, LENGTHS, TAUS, RHOS  # noqa: E402

st.AGE_MIN = 1                     # adults: every night counts
SEED_NULL, SEED_SIM = 20261120, 20261121
MAX_SHIFT = 11.0
TRIP_MIN_H = 2.5


def phase_of(x):
    """Nightly sleep phase (clock hours) and R for noon-anchored rows."""
    return st.daily_phase(np.roll(x, 72, axis=1))


def load(res):
    z = np.load(res / "adult" / "adult_records.npz")
    ids = [f"A{k + 1:02d}" for k in range(len(z["ids"]))]   # pseudonyms in all outputs
    X = [z[f"x_{k}"].astype(float) for k in range(len(ids))]
    first = [str(z[f"first_{k}"]) for k in range(len(ids))]
    off = [z[f"offset_{k}"].astype(float) if f"offset_{k}" in z.files else None for k in range(len(ids))]
    return ids, X, first, off


def screen(ids, X, out, n_sur):
    PH, RR = [], []
    for x in X:
        p, r = phase_of(x)
        PH.append(p)
        RR.append(r)
    np.savez_compressed(out / "adult_phase.npz", **{f"phase_{k}": p for k, p in enumerate(PH)},
                        **{f"R_{k}": r for k, r in enumerate(RR)}, ids=np.array(ids))
    usable = sum(int(js.usable_days(p, r).sum()) for p, r in zip(PH, RR))
    rng = np.random.default_rng(SEED_NULL)
    rows, eps3 = [], None
    for thr in js.THRESHOLDS:
        found, null = [], np.zeros(n_sur, dtype=int)
        for k, (p, r) in enumerate(zip(PH, RR)):
            eps = js.detect(p, r, thr)
            found += [{"adult": ids[k], "adult_index": k, **e} for e in eps]
            c, _ = js.surrogate_counts(p, r, thr, eps, n_sur, rng)
            null += c
        df = pd.DataFrame(found)
        df.to_csv(out / f"episodes_{thr:g}h.csv", index=False)
        main = df[df["shift_h"].abs() < MAX_SHIFT] if len(df) else df
        rows.append({"threshold_h": thr, "episodes": len(df), "adults": df["adult"].nunique() if len(df) else 0,
                     "episodes_main": len(main),
                     "later": int((main["shift_h"] > 0).sum()) if len(main) else 0,
                     "earlier": int((main["shift_h"] < 0).sum()) if len(main) else 0,
                     "null_mean": round(float(null.mean()), 2), "null_lo95": int(np.percentile(null, 2.5)),
                     "null_hi95": int(np.percentile(null, 97.5)), "usable_nights": usable,
                     "person_years": round(usable / 365.25, 1), "n_surrogates": n_sur})
        print(rows[-1], flush=True)
        if thr == 3.0:
            eps3 = df
    pd.DataFrame(rows).to_csv(out / "screen_counts.csv", index=False)
    return PH, RR, eps3


def episode_metrics(X, PH, RR, eps, out):
    rows = []
    for j, e in eps.reset_index(drop=True).iterrows():
        k = int(e["adult_index"])
        dev, coef = deviation(PH[k], RR[k], int(e["start_day"]), int(e["end_day"]), e["shift_h"],
                              e["baseline_phase_h"])
        m = measures(dev, int(e["start_day"]), int(e["end_day"]))
        a, b = int(e["start_day"]) - 1, int(e["end_day"]) - 1
        x = X[k]
        logged = ~np.isnan(x).all(1)
        sleep_h = np.nansum(x, axis=1) * 10 / 60
        pre = [d for d in range(max(a - 14, 0), a) if logged[d]]
        ep3 = [d for d in range(a, min(a + 3, x.shape[0])) if logged[d]]
        rows.append({"episode": j + 1, "adult": e["adult"], "start_day": a + 1, "end_day": b + 1,
                     "span_days": int(e["span_days"]), "detector_shift_h": e["shift_h"],
                     "main_set": abs(e["shift_h"]) < MAX_SHIFT,
                     **{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in m.items()},
                     "unlogged_before": int((~logged[max(a - 3, 0):a]).sum()),
                     "unlogged_after": int((~logged[b + 1:b + 4]).sum()),
                     "sleep_pre_h": round(float(np.mean(sleep_h[pre])), 2) if pre else np.nan,
                     "sleep_first3_h": round(float(np.mean(sleep_h[ep3])), 2) if ep3 else np.nan})
    M = pd.DataFrame(rows)
    M.to_csv(out / "episode_metrics_3h.csv", index=False)
    return M


def simulate(X, PH, RR, eps2, out, reps):
    use = [~np.isnan(p) & (r >= st.R_MIN) for p, r in zip(PH, RR)]
    busy = [np.zeros(p.size, dtype=bool) for p in PH]
    for _, e in eps2.iterrows():
        k = int(e["adult_index"])
        busy[k][max(int(e["start_day"]) - 1 - 7, 0):int(e["end_day"]) + 7] = True
    rng = np.random.default_rng(SEED_SIM)
    st.THR_H = 3.0
    rows = []
    for D in SIZES:
        for L in LENGTHS:
            for tau in TAUS:
                for rho in RHOS:
                    for rep in range(reps):
                        for _ in range(5000):                       # host record and onset
                            k = int(rng.integers(0, len(X)))
                            n = X[k].shape[0]
                            if n < L + 120:
                                continue
                            t0 = int(rng.integers(28, n - L - 57))
                            lo, hi = t0 - 28, t0 + L + 56
                            if use[k][lo:hi].mean() >= 0.8 and not busy[k][lo:hi].any():
                                break
                        sign = 1.0 if rng.random() < 0.5 else -1.0
                        t = np.arange(L + 56)
                        s = np.where(t < L, D * np.exp(-t / tau), rho * D * np.exp(-(t - L) / 7.0))
                        ph, R = phase_of(shifted(X[k], t0, sign * s))
                        found = [e for e in st.detect_jumps(ph, R)
                                 if t0 <= e["start_day"] - 1 <= t0 + 2 and np.sign(e["shift_h"]) == sign]
                        row = dict(size_h=D, length_d=L, true_tau_d=tau, rho=rho, adult=k, onset_day=t0 + 1,
                                   detected=bool(found))
                        if found:
                            e = found[0]
                            dev, _ = deviation(ph, R, e["start_day"], e["end_day"], e["shift_h"], e["baseline_phase_h"])
                            row.update(span_days=e["span_days"], **measures(dev, e["start_day"], e["end_day"]))
                        rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "simulation_runs.csv", index=False)
    g = runs.groupby(["size_h", "true_tau_d"])["detected"].mean().unstack().round(3)
    g.to_csv(out / "simulation_summary.csv")
    print(g, flush=True)
    return runs


def trips(ids, X, PH, RR, first, off, eps_by_thr, out):
    """Clock changes in records whose UTC offset is known (Oura). Home is the most common offset and
    any offset within 1 h of it (daylight saving). A trip is a run of nights on an offset more than
    TRIP_MIN_H from home; it gives two clock changes, on arrival and on return. Both are measured
    against the same baseline, the circular mean local-time phase of the 14 usable home nights
    before departure, because people aim to sleep at their usual local time. The lag on later
    nights is the local-time phase minus that baseline, signed by the direction of the clock change,
    so that sleep left on the previous clock gives a lag equal to the change and sleep fully moved to
    the new clock gives zero. The lag on arrival is the median of the first 3 usable nights.
    Changes are found between consecutive nights with a known offset, so the 1-h steps of daylight
    saving time never count; each change away from home is paired with the next change back."""
    rows, traj = [], []
    for k, o in enumerate(off):
        if o is None:
            continue
        p, r = PH[k], RR[k]
        use = ~np.isnan(p) & (r >= st.R_MIN)
        known = np.flatnonzero(~np.isnan(o))
        steps = [(b, o[b] - o[a]) for a, b in zip(known, known[1:]) if abs(o[b] - o[a]) >= TRIP_MIN_H]
        legs, last_return = [], 0
        for (s, d_out), (e, d_back) in zip(steps[::2], steps[1::2]):   # out-and-back pairs
            base_n = [m for m in range(max(s - 21, last_return), s) if use[m]][-14:]
            last_return = e
            if len(base_n) < 5:
                continue
            legs.append((s, e, d_out, d_back, st.circmean_h(p[base_n])))
        for s, e, d_out, d_back, base in legs:
            for leg, n, change, span in (("arrival", s, d_out, e - s), ("return", e, d_back, 14)):
                sgn = np.sign(change)
                lag = np.where(use, sgn * st.wrap(p - base), np.nan)
                first3 = [m for m in range(n, min(n + 5, p.size)) if use[m]][:3]
                lag0 = float(np.median(lag[first3])) if first3 else np.nan
                found = {}
                for thr, eps in eps_by_thr.items():
                    hit = eps[(eps["adult_index"] == k) & (eps["start_day"] - 1 >= n - 2) & (eps["start_day"] - 1 <= n + 2)]
                    found[f"found_{thr:g}h"] = bool(len(hit))
                rows.append({"adult": ids[k], "leg": leg, "night_index": int(n),
                             "clock_change_h": round(float(change), 1),
                             "nights_on_new_clock": int(span) if leg == "arrival" else None,
                             "lag_on_arrival_h": round(lag0, 2),
                             "share_moved_at_once": round(1 - lag0 / abs(change), 3) if first3 else np.nan,
                             **found})
                for t in range(-7, 15):
                    m = n + t
                    traj.append({"change": len(rows), "leg": leg, "night": t, "on_new_clock": 0 <= t < span,
                                 "lag_h": round(float(lag[m]), 3) if 0 <= m < p.size and not np.isnan(lag[m]) else np.nan})
    T = pd.DataFrame(rows)
    T.to_csv(out / "trips.csv", index=False)
    pd.DataFrame(traj).to_csv(out / "trip_trajectories.csv", index=False)
    print(T.to_string(index=False), flush=True)
    return T


def scatter(res, ids, PH, RR, out):
    """Night-to-night scatter of sleep timing: SD of the phase around each person's own smooth trend
    (the trend of the null model), on usable days outside 3-h episodes (2 days before to 7 after);
    infants from day 90, adults throughout. People with fewer than 60 such days are left out."""
    def one(phase, R, eps, age_min):
        use = ~np.isnan(phase) & (R >= st.R_MIN) & (np.arange(phase.size) + 1 >= age_min)
        clean = use.copy()
        for e in eps:
            clean[max(e["start_day"] - 3, 0):e["end_day"] + 7] = False
        if clean.sum() < 60:
            return None
        r = st.wrap(phase[clean] - js.trend(phase, clean, np.flatnonzero(clean))[clean])
        return round(float(np.std(r)), 3)
    rows = []
    z = np.load(res / "nyu" / "nyu_timing_arrays.npz")
    P, R = z["phase"].astype(float), z["R"].astype(float)
    ep = pd.read_csv(res / "jetlag" / "episodes_3h.csv")
    for i in range(P.shape[0]):
        v = one(P[i], R[i], ep[ep["infant"] == i + 1].to_dict("records"), js.MIN_AGE)
        if v is not None:
            rows.append({"group": "infant", "id": i + 1, "sd_h": v})
    aep = pd.read_csv(out / "episodes_3h.csv")
    for k in range(len(ids)):
        v = one(PH[k], RR[k], aep[aep["adult_index"] == k].to_dict("records"), 1)
        if v is not None:
            rows.append({"group": "adult", "id": ids[k], "sd_h": v})
    D = pd.DataFrame(rows)
    D.to_csv(out / "day_to_day_scatter.csv", index=False)
    print(D.groupby("group")["sd_h"].describe(), flush=True)


def main(res_dir, n_sur=20, reps=30):
    res = Path(res_dir)
    out = res / "adult"
    ids, X, first, off = load(res)
    PH, RR, eps3 = screen(ids, X, out, n_sur)
    scatter(res, ids, PH, RR, out)
    M = episode_metrics(X, PH, RR, eps3, out)
    pd.set_option("display.width", 250)
    print(M[["episode", "adult", "start_day", "span_days", "detector_shift_h", "shift_h", "prev_h", "abrupt_onset",
             "change_h", "frac_left", "tau_d", "abrupt_end", "unlogged_before", "pre_sd_h"]].to_string(index=False))
    eps_by_thr = {thr: pd.read_csv(out / f"episodes_{thr:g}h.csv") for thr in js.THRESHOLDS}
    trips(ids, X, PH, RR, first, off, eps_by_thr, out)
    simulate(X, PH, RR, eps_by_thr[2.0], out, reps)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 20, int(a[3]) if len(a) > 3 else 30)
