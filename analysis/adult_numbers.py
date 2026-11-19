"""Every number about the adults quoted in the Letter, from the outputs of adult_screen.py.

Usage: python3 -I adult_numbers.py <results_dir>  (prints Markdown; run_all.sh saves it as
results/adult/adult_numbers.md)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RES = Path(sys.argv[1])
A, J = RES / "adult", RES / "jetlag"
z = np.load(A / "adult_records.npz")
ids = [str(s) for s in z["ids"]]          # source-fileid; outputs use the labels A01-A30
sc = pd.read_csv(A / "screen_counts.csv").set_index("threshold_h")
inf = pd.read_csv(J / "screen_counts.csv").set_index("threshold_h")
eps = pd.read_csv(A / "episodes_3h.csv")
scat = pd.read_csv(A / "day_to_day_scatter.csv")
sim = pd.read_csv(A / "simulation_runs.csv")
sim_i = pd.read_csv(J / "simulation_runs.csv")
T = pd.read_csv(A / "trips.csv")
tt = pd.read_csv(A / "trip_trajectories.csv")
M = pd.read_csv(J / "episode_metrics_3h.csv")
S = M[M["main_set"]]

nights = [int((~np.isnan(z[f"x_{k}"]).all(1)).sum()) for k in range(len(ids))]
print("# Adult comparison: numbers quoted in the Letter\n")
print("## Records")
print(f"- adults: {len(ids)} ({sum(i.startswith('fitbit') for i in ids)} Fitbit, {sum(i.startswith('oura') for i in ids)} Oura)")
print(f"- nights with a sleep record: {sum(nights):,} (median {int(np.median(nights))} per adult, range {min(nights)}-{max(nights)})")
print(f"- usable nights: {int(sc.loc[3.0, 'usable_nights']):,} = {sc.loc[3.0, 'person_years']} person-years")

print("\n## Screen (Figure S2A)")
for thr in sc.index:
    r = sc.loc[thr]
    print(f"- {thr:g} h: {int(r.episodes)} episodes in {int(r.adults)} adults; expected by chance {r.null_mean} "
          f"(95% range {int(r.null_lo95)}-{int(r.null_hi95)}); per 100 person-years found "
          f"{100 * r.episodes / r.person_years:.1f}, expected {100 * r.null_mean / r.person_years:.1f}")
top = eps["adult"].value_counts()
print(f"- 3 h: episodes per adult {top.to_dict()}")
r = inf.loc[3.0]
print(f"- infants from day 90, 3 h: {int(r.episodes_from90)} found vs {r.null_from90_mean} expected "
      f"({int(r.null_from90_lo95)}-{int(r.null_from90_hi95)})")

print("\n## Night-to-night scatter of sleep timing (Figure 2C)")
for g in ("infant", "adult"):
    v = scat[scat.group == g]["sd_h"]
    q = np.percentile(v, [25, 50, 75])
    print(f"- {g}s: n = {len(v)}, median SD {q[1]:.2f} h (IQR {q[0]:.2f}-{q[2]:.2f})")
u = mannwhitneyu(scat[scat.group == "infant"]["sd_h"], scat[scat.group == "adult"]["sd_h"])
print(f"- Mann-Whitney U = {u.statistic:.0f}, p = {u.pvalue:.1e}")
mx = scat[scat.group == "adult"].sort_values("sd_h").iloc[-1]
print(f"- most variable adult: {mx.id} SD {mx.sd_h:.2f} h")

print("\n## Simulated shifts, 3-h threshold (Figure S2B): detected %, adults vs infants")
for s in (3.0, 4.0, 5.0, 6.0):
    parts = []
    for tau in (np.inf, 10.0, 4.0):
        a = 100 * sim[(sim.size_h == s) & (sim.true_tau_d == tau)].detected.mean()
        i = 100 * sim_i[(sim_i.size_h == s) & (sim_i.true_tau_d == tau)].detected.mean()
        parts.append(f"tau {tau:g}: {a:.0f} vs {i:.0f}")
    print(f"- {s:g} h: " + "; ".join(parts))

print("\n## Clock changes with recorded time zone (Figure 2A, B)")
print(f"- adults with a change of >= 2.5 h: {T.adult.nunique()}; trips: {int((T.leg == 'arrival').sum())}; "
      f"clock changes: {len(T)}; sizes: {sorted(T.clock_change_h.abs().tolist())}")
print(f"- nights on the trips: {T[T.leg == 'arrival'].nights_on_new_clock.astype(int).tolist()}")
fcols = [c for c in T.columns if c.startswith("found_")]
print(f"- found by the screen at any threshold: {int(T[fcols].any(axis=1).sum())} of {len(T)}")
q = T.lag_on_arrival_h
print(f"- lag on arrival (median of first 3 nights): median {q.median():.2f} h, range {q.min():.2f}-{q.max():.2f}")
q = T.share_moved_at_once
print(f"- share of the change moved at once: median {100 * q.median():.0f}%, range {100 * q.min():.0f}%-{100 * q.max():.0f}%")
on = tt[tt.on_new_clock & (tt.night >= 0)]
med = on.groupby("night")["lag_h"].median()
print("- median lag by night: " + ", ".join(f"{n}: {v:.2f}" for n, v in med.items() if n <= 10))
first_zero = int(med[(med.index >= 1) & (med <= 0.25)].index.min())
print(f"- first night with median lag <= 0.25 h: night {first_zero} (night 0 = first night on the new clock)")

print("\n## Infants for comparison (Figure 2B)")
print(f"- shift on days 1-2: median {S.shift_h.median():.1f} h; share left at the end: median "
      f"{100 * S.frac_left.median():.0f}%; duration median {S.span_days.median():g} days")
