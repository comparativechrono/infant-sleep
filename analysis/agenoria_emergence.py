"""Apply the NYU emergence pipeline to the two Agenoria infants (external check).

Usage: python3 -I agenoria_emergence.py <agenoria_data_dir> <nyu_out_dir> <out_dir>
Needs rhythm.py and nyu_rhythm_analysis.py next to this script, and the NYU outputs
(nyu_infant_rhythm_summary.csv, nyu_rhythm_windows.npz) in <nyu_out_dir>.
Data: Jiuguang Wang, Agenoria (github.com/jiuguangw/Agenoria), CC BY-NC-SA 4.0.

Each infant's Glow Baby sleep log is put into the NYU format: one row per day of age
(row 0 = date of birth), 144 ten-minute bins holding the share of the bin logged as asleep,
gaps within a logged day read as awake, and NaN rows for days with no logged sleep.
The rows then go through the same functions as the 700 NYU infants (analyse_infant,
emergence, free_running) with the same constants.

Outputs in <out_dir>:
  agenoria_emergence_summary.csv   one row per infant: plateau, t50, t_abs, early 24-h specificity,
                                   free-running screen, and where t50 and plateau fall among NYU infants
  agenoria_s24_by_day.csv          24-h strength by day of age for both infants and the NYU
                                   eligible infants (median, quartiles)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import nyu_rhythm_analysis as nra
from rhythm import load_sessions, lomb_scargle_windows, minute_states

BABIES = [("zyw", "Baby 1", "2018-11-21"), ("zlw", "Baby 2", "2022-02-21")]
N_DAYS = 372              # a week past the first birthday, so the week-52 window is complete
FLANK = np.flatnonzero(((nra.PERIODS >= 20) & (nra.PERIODS <= 22)) |
                       ((nra.PERIODS >= 26) & (nra.PERIODS <= 28)))


def nyu_format(data_dir: Path, code: str, bday: str) -> np.ndarray:
    """(N_DAYS, 144) share of each 10-min bin logged as asleep; NaN rows = no sleep logged that day."""
    st = minute_states(load_sessions(data_dir, code), pd.Timestamp(bday), N_DAYS)
    asleep = st == 2
    x = asleep.reshape(N_DAYS, 144, 10).mean(-1)
    x[~asleep.any(1)] = np.nan
    return x


def first_window_specificity(x: np.ndarray, first_day: int) -> dict:
    """Power at 24 h relative to the median power at 20-22 h and 26-28 h, in the first usable window."""
    W = sliding_window_view(x.reshape(-1), nra.WIN * nra.PER_DAY)[::nra.PER_DAY]
    w = W[first_day - 1 - nra.HALF][None, :]
    P = lomb_scargle_windows(w, nra.T_H, nra.PERIODS, basis=nra.BASIS)[0]
    circ = nra.CIRC[np.nanargmax(P[nra.CIRC])]
    return dict(first_window_s24=float(P[nra.I24]),
                first_window_flank_ratio=float(P[nra.I24] / np.median(P[FLANK])),
                first_window_circ_peak_h=float(nra.PERIODS[circ]))


def percentile(values: np.ndarray, v: float) -> float:
    """Share of values below v, counting ties as half (mid-rank percentile, 0-100)."""
    return float(100 * (np.mean(values < v) + 0.5 * np.mean(values == v)))


def main(data_dir: str, nyu_dir: str, out_dir: str):
    data_dir, nyu_dir, out_dir = Path(data_dir), Path(nyu_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nyu = pd.read_csv(nyu_dir / "nyu_infant_rhythm_summary.csv")
    elig = nyu[nyu.eligible]
    nyu_t50 = elig.t50_day.dropna().to_numpy()
    nyu_tabs = elig.t_abs_day.dropna().to_numpy()
    nyu_plat = elig.plateau_s24.to_numpy()
    s24_nyu = np.load(nyu_dir / "nyu_rhythm_windows.npz")["s24"][elig.infant.to_numpy() - 1, :N_DAYS]

    rows, by_day = [], {"day_of_age": np.arange(1, N_DAYS + 1)}
    for code, label, bday in BABIES:
        x = nyu_format(data_dir, code, bday)
        out = nra.analyse_infant(x)
        rec = dict(infant=label, birth_date=bday, days_logged_first_year=int((~np.isnan(x[:365]).all(1)).sum()))
        rec.update(nra.emergence(out["s24"]))
        rec.update(first_window_specificity(x, rec["first_window_day"]))
        for wk in (2, 4, 8, 12, 26, 52):
            rec[f"s24_week{wk}"] = out["s24"][wk * 7 - 1]
        rec.update(nra.free_running(out))
        circ = out["circ_period"][~np.isnan(out["circ_period"])]
        rec["circ_peak_h_median"] = float(np.median(circ))
        rec["windows_peak_within_0.5h_of_24"] = float(np.mean(np.abs(circ - 24) <= 0.5))
        rec["t50_nyu_percentile"] = percentile(nyu_t50, rec["t50_day"])
        rec["t_abs_nyu_percentile"] = percentile(nyu_tabs, rec["t_abs_day"])
        rec["plateau_nyu_percentile"] = percentile(nyu_plat, rec["plateau_s24"])
        rows.append(rec)
        by_day[f"{label.lower().replace(' ', '')}_s24"] = out["s24"]

    with np.errstate(invalid="ignore"):
        q = np.nanpercentile(s24_nyu, [25, 50, 75], axis=0)
    by_day.update(nyu_q1=q[0], nyu_median=q[1], nyu_q3=q[2],
                  nyu_n=(~np.isnan(s24_nyu)).sum(0))
    summ = pd.DataFrame(rows)
    summ.round(4).to_csv(out_dir / "agenoria_emergence_summary.csv", index=False)
    pd.DataFrame(by_day).round(4).to_csv(out_dir / "agenoria_s24_by_day.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summ.T)
    print(f"NYU eligible: n={len(elig)}, t50 median {np.median(nyu_t50)}, "
          f"plateau median {np.median(nyu_plat):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
