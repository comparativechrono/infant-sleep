"""Synthetic infants with known answers, run through nyu_rhythm_analysis's functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import nyu_rhythm_analysis as nra

rng = np.random.default_rng(42)
N_DAYS = 730
MIN = N_DAYS * 1440


def random_bouts(start_min, end_min, minutes):
    """Alternating sleep/wake bouts with no daily structure."""
    t, asleep = start_min, rng.random() < 0.5
    while t < end_min:
        dur = rng.uniform(60, 210) if asleep else rng.uniform(30, 120)
        if asleep:
            minutes[int(t):int(min(t + dur, end_min))] = 1
        t += dur
        asleep = not asleep


def schedule(start_day, end_day, minutes, period_h=24.0):
    """Night sleep ~10.5 h plus two naps, repeating every period_h hours (with jitter)."""
    cyc = period_h * 60
    k0 = int(np.floor(start_day * 1440 / cyc))
    k = k0
    while True:
        base = k * cyc
        if base > end_day * 1440:
            break
        for on, length in ((20 * 60, 10.5 * 60), (33.5 * 60, 60), (37.5 * 60, 90)):  # night, 09:30, 13:30 next day
            s = base + on + rng.normal(0, 20)
            e = s + length + rng.normal(0, 15)
            lo, hi = max(s, start_day * 1440), min(e, end_day * 1440)
            if hi > lo:
                minutes[int(lo):int(hi)] = 1
        k += 1


def to_bins(minutes):
    d = minutes.reshape(N_DAYS, 144, 10).mean(-1)
    missing = rng.random(N_DAYS) < 0.10
    d[missing] = np.nan
    return d


cases = {}
m = np.zeros(MIN); random_bouts(0, 60 * 1440, m); schedule(60, N_DAYS, m); cases["A: rhythm switches on at day 60"] = m
m = np.zeros(MIN); schedule(0, 50, m, period_h=25.0); schedule(50, N_DAYS, m); cases["B: 25-h free-run until day 50"] = m
m = np.zeros(MIN); schedule(0, N_DAYS, m); cases["C: entrained from birth"] = m
m = np.zeros(MIN); random_bouts(0, N_DAYS * 1440, m); cases["D: no rhythm at all"] = m

for name, minutes in cases.items():
    x = to_bins(minutes)
    out = nra.analyse_infant(x)
    em = nra.emergence(out["s24"])
    fr = nra.free_running(out)
    s = out["s24"]
    print(f"{name}\n   S24 at days 30/50/60/66/90/150: "
          + " ".join(f"{s[d - 1]:.2f}" for d in (30, 50, 60, 66, 90, 150))
          + f"\n   plateau {em['plateau_s24']:.2f}, first window day {em['first_window_day']}, "
            f"t50 day {em['t50_day']} ({em['t50_censoring'] or '-'}), t_abs day {em['t_abs_day']} "
            f"({em['t_abs_censoring'] or '-'})"
          + f"\n   free-run: {fr['fr_episode']} days {fr['fr_start_day']}-{fr['fr_end_day']}, "
            f"period {fr['fr_period_h']}")
