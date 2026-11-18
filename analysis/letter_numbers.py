"""Every number quoted in the Letter on sudden shifts in clock time, from the result files.

Usage: python3 -I letter_numbers.py <results_dir> > results/jetlag/letter_numbers.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(sys.argv[1])
J = RES / "jetlag"
z = np.load(RES / "nyu" / "nyu_timing_arrays.npz")
phase, R = z["phase"].astype(float), z["R"].astype(float)
use = ~np.isnan(phase) & (R >= 0.15)
logged = ~np.isnan(phase)
screen = pd.read_csv(J / "screen_counts.csv")
M3 = pd.read_csv(J / "episode_metrics_3h.csv")
S = M3[M3["main_set"]].copy()
M25 = pd.read_csv(J / "episode_metrics_2.5h.csv")
S25 = M25[M25["main_set"]].copy()
sim = pd.read_csv(J / "simulation_runs.csv")

med = lambda v, f=".1f": f"{np.nanmedian(v):{f}} (IQR {np.nanpercentile(v, 25):{f}} to {np.nanpercentile(v, 75):{f}})"
pct = lambda a, b: f"{a}/{b} ({100 * a / b:.0f}%)"
print("# Numbers quoted in the Letter on sudden shifts in clock time\n")
print("Threshold 3 h; main set = episodes starting at day 90 or later with shifts below 11 h. "
      "Deviations are measured from a baseline trend (see jetlag_profiles.py).\n")

print("## Screen and null model\n")
print("| Threshold (h) | Episodes, all ages | Infants | Episodes from day 90 | Infants | Expected from ordinary variation, from day 90 (mean, 95% range) |")
print("|---|---|---|---|---|---|")
for _, r in screen.iterrows():
    print(f"| {r.threshold_h:g} | {int(r.episodes)} | {int(r.infants)} | {int(r.episodes_from90)} | {int(r.infants_from90)} | "
          f"{r.null_from90_mean:.1f} ({int(r.null_from90_lo95)}–{int(r.null_from90_hi95)}) |")
r3 = screen[screen.threshold_h == 3.0].iloc[0]
print(f"\n- At 3 h, all ages: {int(r3.episodes)} episodes in {int(r3.infants)} infants; {int(r3.episodes - r3.episodes_from90)} "
      f"started before day 90 (excluded from the main set).")
young = M3[M3["start_day"] < 90]
from90_days = int(use[:, 89:].sum())
young_days = int(use[:, 41:89].sum())
print(f"- Episodes per 1,000 usable infant-days: days 42–89 {1000 * len(young) / young_days:.2f} "
      f"({len(young)}/{young_days:,}); from day 90 {1000 * len(S) / from90_days:.2f} ({len(S)}/{from90_days:,}).")
n_inf = int((use[:, 89:].sum(1) >= 30).sum())
print(f"- Infants with at least 30 usable days from day 90: {n_inf}; with a main-set episode: "
      f"{S.infant.nunique()} ({100 * S.infant.nunique() / n_inf:.1f}%); with more than one: "
      f"{int((S.infant.value_counts() > 1).sum())}.")
print(f"- One episode per {from90_days / len(S) / 365.25:.1f} infant-years of logging from day 90.")
print(f"- Expected false episodes from day 90: {r3.null_from90_mean:.2f} of {len(S)} "
      f"({100 * r3.null_from90_mean / len(S):.0f}%).\n")

print("## Main set: size, direction, timing\n")
later = int((S.direction == "later").sum())
print(f"- {len(S)} episodes in {S.infant.nunique()} infants; sleep later than before in {later}, earlier in "
      f"{len(S) - later} (two-sided binomial p = {stats.binomtest(later, len(S), 0.5).pvalue:.2f}).")
print(f"- Shift on the first 2 days: median {med(S.shift_h)} h; range {S.shift_h.min():.1f}–{S.shift_h.max():.1f}.")
print(f"- Detector's estimate of the shift (mean over the episode): median {med(S.detector_shift_h.abs())} h.")
print(f"- Duration: median {med(S.span_days)} days; range {S.span_days.min()}–{S.span_days.max()}.")
print(f"- Age at onset: median {med(S.start_day, '.0f')} days; range {S.start_day.min()}–{S.start_day.max()}.")
ab = S.abrupt_onset.dropna().astype(bool)
print(f"- Abrupt onset (day before < half the shift): {pct(int(ab.sum()), len(ab))}; "
      f"{len(S) - len(ab)} had no logged day in the 3 days before.")
bg = [((~logged[i, d - 3:d]).any()) for i in S.infant.unique() - 1 for d in range(92, 730) if use[i, d]]
bg_after = [((~logged[i, d + 1:d + 4]).any()) for i in S.infant.unique() - 1 for d in range(92, 726) if use[i, d]]
k_b = int((S.unlogged_before >= 1).sum())
k_a = int((S.unlogged_after >= 1).sum())
print(f"- At least one unlogged day in the 3 days before onset: {pct(k_b, len(S))}; background in the same "
      f"infants from day 90: {100 * np.mean(bg):.1f}% of usable days (binomial p = "
      f"{stats.binomtest(k_b, len(S), np.mean(bg), alternative='greater').pvalue:.1g}).")
print(f"- At least one unlogged day in the 3 days after the end: {pct(k_a, len(S))}; background "
      f"{100 * np.mean(bg_after):.1f}% (p = {stats.binomtest(k_a, len(S), np.mean(bg_after), alternative='greater').pvalue:.2g}).")
print(f"- Day-to-day scatter of the deviation in the 14 days before onset (SD): median {med(S.pre_sd_h, '.2f')} h.\n")

print("## Course while shifted\n")
shr = int((S.change_h <= -0.25 * S.shift_h).sum())
fin = S.tau_d[np.isfinite(S.tau_d)]
print(f"- Change from the first 2 to the last 2 days: median {med(S.change_h, '.2f')} h; "
      f"share of the shift left at the end: median {med(S.frac_left, '.2f')}.")
print(f"- Shrank by at least a quarter: {pct(shr, len(S))}; did not shrink at all: {int((S.change_h >= 0).sum())}.")
print(f"- Time constant where the shift shrank ({len(fin)} episodes): median {med(fin)} days; "
      f"rate of re-alignment, all episodes: median {med(S.rate_h_per_day, '.2f')} h/day.")
w = stats.wilcoxon(S.change_h)
print(f"- Change different from 0: Wilcoxon p = {w.pvalue:.1g}.")
ae = S.abrupt_end.dropna().astype(bool)
print(f"- Abrupt end (drop of at least half the shift from the last episode day to the next): "
      f"{pct(int(ae.sum()), len(ae))}; median duration {S[S.abrupt_end == True].span_days.median():.0f} days "
      f"(abrupt end) and {S[S.abrupt_end == False].span_days.median():.0f} days (gradual).")
d_sleep = S.sleep_first3_h - S.sleep_pre_h
print(f"- Hours asleep per logged day: before {med(S.sleep_pre_h)}; first 3 days {med(S.sleep_first3_h)}; "
      f"paired difference median {np.median(d_sleep):+.2f} h (Wilcoxon p = {stats.wilcoxon(d_sleep).pvalue:.2f}).\n")

print("## What the speed of re-alignment relates to (Spearman, share of the shift lost while it lasted)\n")
S["frac_change"] = -S.change_h / S.shift_h          # share of the initial shift lost by the episode's end
for col, lab in [("start_day", "age at onset"), ("shift_h", "size of the shift"), ("span_days", "duration"),
                 ("plateau_s24", "24-h rhythm strength (plateau, paper 1)")]:
    ok = S[col].notna()
    rho, p = stats.spearmanr(S.loc[ok, col], S.loc[ok, "frac_change"])
    print(f"- {lab}: rho = {rho:+.2f}, p = {p:.2f}, n = {ok.sum()}")
lat, ear = S[S.direction == "later"].frac_change, S[S.direction == "earlier"].frac_change
print(f"- Later vs earlier shifts: median share lost {lat.median():.2f} vs {ear.median():.2f} "
      f"(Mann–Whitney p = {stats.mannwhitneyu(lat, ear).pvalue:.2f}).\n")

print("## Simulated shifts (detector at 3 h; 30 per condition)\n")
print("Detection, averaged over durations (4, 7, 14 days) and residual offsets:\n")
print("| Size (h) | No re-alignment | Time constant 10 days | Time constant 4 days |")
print("|---|---|---|---|")
for D in sorted(sim.size_h.unique()):
    cells = [f"{100 * sim[(sim.size_h == D) & (sim.true_tau_d == t)].detected.mean():.0f}%" for t in (np.inf, 10.0, 4.0)]
    print(f"| {D:g} | " + " | ".join(cells) + " |")
det = sim[sim.detected & (sim.size_h >= 4)]
for t, lab in [(np.inf, "no re-alignment"), (10.0, "time constant 10 days"), (4.0, "time constant 4 days")]:
    d = det[det.true_tau_d == t]
    print(f"\n- Shifts of 4–6 h, {lab}: measured change median {d.change_h.median():+.2f} h "
          f"(IQR {d.change_h.quantile(.25):+.2f} to {d.change_h.quantile(.75):+.2f}); abrupt end "
          f"{100 * d.abrupt_end.astype(bool).mean():.0f}%; abrupt onset {100 * d.abrupt_onset.dropna().astype(bool).mean():.0f}%; "
          f"share of the shift left at the end median {d.frac_left.median():.2f}; "
          f"shift on first 2 days {d.shift_h.median():.2f} h vs true {d.size_h.median():g} h (median).")
for t, lab in [(10.0, "10"), (4.0, "4")]:
    x = det[det.true_tau_d == t].tau_d
    print(f"- True time constant {lab} days: estimated median {x.median():.1f} days "
          f"(IQR {x.quantile(.25):.1f} to {x.quantile(.75):.1f}).")
x = det[det.true_tau_d == np.inf].tau_d
print(f"- No re-alignment: estimated time constant infinite in {100 * np.mean(np.isinf(x)):.0f}%, "
      f"otherwise median {x[np.isfinite(x)].median():.0f} days.")
print()
print("## Sensitivity: threshold 2.5 h\n")
ab25 = S25.abrupt_onset.dropna().astype(bool)
ae25 = S25.abrupt_end.dropna().astype(bool)
r25 = screen[screen.threshold_h == 2.5].iloc[0]
print(f"- From day 90: {len(S25)} episodes in {S25.infant.nunique()} infants; expected from ordinary variation "
      f"{r25.null_from90_mean:.1f}.")
print(f"- Abrupt onset {pct(int(ab25.sum()), len(ab25))}; change median {med(S25.change_h, '.2f')} h; "
      f"shrank by at least a quarter {pct(int((S25.change_h <= -0.25 * S25.shift_h).sum()), len(S25))}; "
      f"abrupt end {pct(int(ae25.sum()), len(ae25))}; unlogged day before onset "
      f"{pct(int((S25.unlogged_before >= 1).sum()), len(S25))}.")
