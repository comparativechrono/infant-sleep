"""Do infants differ in how fast their sleep re-aligns after a sudden shift, beyond measurement error?

Usage: python3 -I jetlag_individual.py <infant_sleep_preprocessed.mat> <results_dir> [reps]

Reads results/jetlag/episode_metrics_3h.csv (main set: onset from day 90, |shift| < 11 h) and
results/nyu/nyu_timing_arrays.npz. Three checks:

1. Measurement error alone. Each observed episode is re-created `reps` times in other infants'
   records (hosts as in jetlag_simulate.py) with its own measured size (mean shift on its first 2
   days) and duration, all re-aligning with one common time constant tau0, and measured with the same
   detector and measures. tau0 is taken from a grid as the value whose simulated episodes match the
   observed median share of the shift left at the end. The spread of that share across the observed
   episodes is then compared with its spread across sets of simulated episodes, one replicate per
   observed episode per set (2,000 sets), at the best-matching and at every other time constant: if
   one speed for all infants plus measurement error explained the data, the observed spread would be
   typical of these sets.
2. Repeat episodes. In infants with more than one episode, are an infant's episodes more alike than
   episodes of different infants? Mean absolute within-infant difference in the share left,
   against the same statistic with episodes shuffled between infants (10,000 permutations).
3. Baseline traits. Spearman correlations of the share of the shift lost with the infant's
   day-to-day variability of sleep timing before onset (SD of the deviation over the 14 days before)
   and chronotype (baseline sleep phase relative to the median of all infants at the same age,
   +/- 7 days, signed in the direction of the shift).

Writes to results/jetlag/: individual_sim_runs.csv (every simulated episode), individual_spread.csv
(the spread test at each common time constant), individual_best_sets.npy (simulated sets at the
best-matching time constant, for Figure S1D), individual_traits.csv and individual_summary.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nyu_sleep_timing as st  # noqa: E402
from jetlag_profiles import deviation, measures  # noqa: E402
from jetlag_simulate import shifted  # noqa: E402

TAU_GRID = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, np.inf]
SEED = 20261120
N_SETS, N_PERM = 2000, 10000


def simulate(data, use0, busy, S, reps, rng):
    rows = []
    for tau in TAU_GRID:
        for _, e in S.iterrows():
            D, L = abs(float(e["shift_h"])), int(e["span_days"])
            for rep in range(reps):
                for _ in range(2000):                                   # host record and onset
                    i = int(rng.integers(0, data.shape[2]))
                    t0 = int(rng.integers(119, 730 - L - 57))
                    lo, hi = t0 - 28, t0 + L + 56
                    if i != int(e["infant"]) - 1 and use0[i, lo:hi].mean() >= 0.8 and not busy[i, lo:hi].any():
                        break
                sign = 1.0 if rng.random() < 0.5 else -1.0
                t = np.arange(L + 56)
                s = np.where(t < L, D * np.exp(-t / tau), 0.0)
                x = shifted(data[:, :, i], t0, sign * s)
                ph, R = st.daily_phase(x)
                found = [f for f in st.detect_jumps(ph, R)
                         if t0 <= f["start_day"] - 1 <= t0 + 2 and np.sign(f["shift_h"]) == sign]
                row = dict(tau0=tau, episode=int(e["episode"]), rep=rep, detected=bool(found))
                if found:
                    f = found[0]
                    dev, _ = deviation(ph, R, f["start_day"], f["end_day"], f["shift_h"], f["baseline_phase_h"])
                    m = measures(dev, f["start_day"], f["end_day"])
                    row.update(span_days=f["span_days"], frac_left=m["frac_left"], change_h=m["change_h"])
                rows.append(row)
        print(f"tau0 = {tau}: {np.mean([r['detected'] for r in rows if r['tau0'] == tau]):.2f} detected", flush=True)
    return pd.DataFrame(rows)


def spread(v):
    v = np.asarray(v, float)
    q1, q3 = np.percentile(v, [25, 75])
    return {"sd": float(np.std(v, ddof=1)), "iqr": float(q3 - q1),
            "none": float(np.mean(v >= 0.9)), "half": float(np.mean(v <= 0.5))}


def sets_from(runs, S, rng):
    """Sets of simulated episodes, one detected replicate per observed episode."""
    det = runs[runs["detected"] & runs["frac_left"].notna()]
    pools = {ep: g["frac_left"].to_numpy() for ep, g in det.groupby("episode")}
    eps = [ep for ep in S["episode"] if ep in pools and pools[ep].size]
    return np.array([[rng.choice(pools[ep]) for ep in eps] for _ in range(N_SETS)]), len(eps)


def main(mat, res_dir, reps=80):
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
    M = pd.read_csv(out / "episode_metrics_3h.csv")
    S = M[M["main_set"]].copy()
    st.THR_H = 3.0
    rng = np.random.default_rng(SEED)

    runs = simulate(data, use0, busy, S, reps, rng)
    runs.to_csv(out / "individual_sim_runs.csv", index=False)

    lines = ["# Do infants differ in how fast their sleep re-aligns?\n"]
    obs = spread(S["frac_left"])
    med_obs = float(S["frac_left"].median())
    lines.append(f"Observed ({len(S)} episodes): median share left {med_obs:.2f}; SD {obs['sd']:.2f}; IQR {obs['iqr']:.2f}; "
                 f"little or none lost (share left >= 0.9) {100 * obs['none']:.0f}%; at least half lost {100 * obs['half']:.0f}%\n")
    lines.append("## One common time constant plus measurement error\n")
    lines.append("| tau0 (days) | detected | median share left (mean over sets) | SD (median, 95% range) | IQR (median) |")
    lines.append("|---|---|---|---|---|")
    best, best_gap, best_sets, rows = None, np.inf, None, []
    for tau in TAU_GRID:
        r = runs[runs["tau0"] == tau]
        sets, n_eps = sets_from(r, S, np.random.default_rng(SEED + 1))
        med = float(np.mean(np.median(sets, axis=1)))
        sd = np.std(sets, axis=1, ddof=1)
        iqr = np.subtract(*np.percentile(sets, [75, 25], axis=1))
        lines.append(f"| {tau:g} | {r['detected'].mean():.2f} | {med:.2f} | {np.median(sd):.2f} "
                     f"({np.percentile(sd, 2.5):.2f}-{np.percentile(sd, 97.5):.2f}) | {np.median(iqr):.2f} |")
        rows.append({"tau0": tau, "detected": round(float(r["detected"].mean()), 3), "median_share_left": round(med, 3),
                     "sd_median": round(float(np.median(sd)), 3), "sd_lo95": round(float(np.percentile(sd, 2.5)), 3),
                     "sd_hi95": round(float(np.percentile(sd, 97.5)), 3),
                     "p_sd_ge_observed": float(np.mean(sd >= obs["sd"]))})
        if abs(med - med_obs) < best_gap:
            best, best_gap, best_sets = tau, abs(med - med_obs), sets
    pd.DataFrame(rows).to_csv(out / "individual_spread.csv", index=False)
    np.save(out / "individual_best_sets.npy", best_sets)
    hi = max(r["sd_hi95"] for r in rows)
    lines.append(f"\nAt every common time constant the simulated SD was below the observed {obs['sd']:.2f}: highest 97.5th "
                 f"percentile {hi:.2f}; largest P(simulated >= observed) {max(r['p_sd_ge_observed'] for r in rows):.4f} "
                 f"({N_SETS} sets each).")
    sd = np.std(best_sets, axis=1, ddof=1)
    iqr = np.subtract(*np.percentile(best_sets, [75, 25], axis=1))
    none = np.mean(best_sets >= 0.9, axis=1)
    half = np.mean(best_sets <= 0.5, axis=1)
    lines.append(f"\nBest-matching common time constant: {best:g} days (median share left in simulated sets "
                 f"{np.mean(np.median(best_sets, axis=1)):.2f}; observed {med_obs:.2f}).")
    lines.append(f"- SD of share left: observed {obs['sd']:.2f}; simulated median {np.median(sd):.2f} "
                 f"(95% range {np.percentile(sd, 2.5):.2f}-{np.percentile(sd, 97.5):.2f}); P(simulated >= observed) = "
                 f"{np.mean(sd >= obs['sd']):.4f}")
    lines.append(f"- IQR of share left: observed {obs['iqr']:.2f}; simulated median {np.median(iqr):.2f} "
                 f"(95% range {np.percentile(iqr, 2.5):.2f}-{np.percentile(iqr, 97.5):.2f}); P(simulated >= observed) = "
                 f"{np.mean(iqr >= obs['iqr']):.4f}")
    lines.append(f"- Episodes losing little or none (share left >= 0.9): observed {100 * obs['none']:.0f}%; simulated median "
                 f"{100 * np.median(none):.0f}% (95% range {100 * np.percentile(none, 2.5):.0f}-{100 * np.percentile(none, 97.5):.0f}%)")
    lines.append(f"- Episodes losing at least half (share left <= 0.5): observed {100 * obs['half']:.0f}%; simulated median "
                 f"{100 * np.median(half):.0f}% (95% range {100 * np.percentile(half, 2.5):.0f}-{100 * np.percentile(half, 97.5):.0f}%)")

    # 1b. two kinds of episode: a share p that do not re-align while they last, the rest with time constant tau

    # 2. repeat episodes
    lines.append("\n## Infants with more than one episode\n")
    rep_inf = S["infant"].value_counts()
    rep_inf = rep_inf[rep_inf > 1].index
    R_ = S[S["infant"].isin(rep_inf)].sort_values(["infant", "start_day"])
    for inf, g in R_.groupby("infant"):
        lines.append(f"- infant {inf}: " + "; ".join(f"day {int(r.start_day)}, {int(r.span_days)} days, share left "
                                                    f"{r.frac_left:.2f}" for r in g.itertuples()))

    def within(df):
        d = []
        for _, g in df.groupby("infant"):
            v = g["frac_left"].to_numpy()
            d += [abs(a - b) for k, a in enumerate(v) for b in v[k + 1:]]
        return float(np.mean(d))
    w_obs = within(R_)
    prng = np.random.default_rng(SEED + 2)
    perm = np.empty(N_PERM)
    vals, labs = S["frac_left"].to_numpy(), R_["infant"].to_numpy()
    for k in range(N_PERM):                      # draw the same number of episodes per infant from all 32
        pick = prng.choice(vals, size=labs.size, replace=False)
        perm[k] = within(pd.DataFrame({"infant": labs, "frac_left": pick}))
    lines.append(f"\nMean within-infant difference in share left {w_obs:.2f}; random pairings {np.median(perm):.2f} "
                 f"(P(random <= observed) = {np.mean(perm <= w_obs):.3f}; {len(rep_inf)} infants, {len(R_)} episodes)")

    # 3. baseline traits
    lines.append("\n## Baseline traits (Spearman with the share of the shift lost)\n")
    ep = pd.read_csv(out / "episodes_3h.csv")
    S = S.merge(ep[["infant", "start_day", "baseline_phase_h"]], on=["infant", "start_day"], how="left")
    S["lost"] = 1 - S["frac_left"]
    chrono = []
    for _, e in S.iterrows():
        a = int(e["start_day"]) - 1
        win = phase0[:, max(a - 21, 0):a - 7]                                   # all infants, same ages as baseline
        med = st.circmean_h(win[~np.isnan(win)])
        sgn = np.sign(e["shift_h"]) if e["direction"] == "later" else -np.sign(e["shift_h"])
        chrono.append(float(st.wrap(e["baseline_phase_h"] - med)) * (1 if e["direction"] == "later" else -1))
    S["chronotype_h"] = chrono
    for col, name in (("pre_sd_h", "day-to-day variability before onset (h)"),
                      ("chronotype_h", "baseline sleep phase vs infants of the same age, in the shift's direction (h)"),
                      ("start_day", "age at onset (days)"), ("shift_h", "size of the shift (h)"),
                      ("span_days", "duration (days)")):
        ok = S[col].notna() & S["lost"].notna()
        rho, p = stats.spearmanr(S.loc[ok, col], S.loc[ok, "lost"])
        lines.append(f"- {name}: rho = {rho:.2f}, p = {p:.2f} (n = {int(ok.sum())})")
    S[["episode", "infant", "start_day", "span_days", "shift_h", "frac_left", "pre_sd_h", "chronotype_h"]].to_csv(
        out / "individual_traits.csv", index=False)
    text = "\n".join(lines)
    (out / "individual_summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 80)
