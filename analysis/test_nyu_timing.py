"""Synthetic checks for nyu_sleep_timing: known bedtimes and injected time-zone jumps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import nyu_sleep_timing as st

rng = np.random.default_rng(3)
N = 730


def infant(shifts=(), fragmented_until=0, missing=0.10):
    """Night 20:00-06:30 with a 20-min feed at ~02:00, naps 09:30-10:30 and 13:30-15:00, +-20 min
    jitter. shifts: (first_day, last_day, hours) clock offsets applied to those days' schedule."""
    m = np.zeros(N * 1440)
    off = np.zeros(N + 2)
    for a, b, h in shifts:
        off[a:b + 1] = h
    for day in range(N + 1):
        if day < fragmented_until:
            continue
        base = day * 1440 + off[day] * 60
        for on, length in ((20 * 60, 6 * 60), (26 * 60 + 20, 4 * 60 + 10), (33.5 * 60, 60),
                           (37.5 * 60, 90)):
            s = base + on + rng.normal(0, 20)
            e = s + length + rng.normal(0, 10)
            lo, hi = max(int(s), 0), min(int(e), N * 1440)
            if hi > lo:
                m[lo:hi] = 1
    t = 0
    while t < fragmented_until * 1440:                      # newborn-like random bouts
        dur = rng.uniform(60, 200)
        m[int(t):int(min(t + dur, fragmented_until * 1440))] = 1
        t += dur + rng.uniform(40, 120)
    x = m.reshape(N, 144, 10).mean(-1)
    x[rng.random(N) < missing] = np.nan
    for a, b, h in shifts:                                   # travel day unlogged
        x[max(a - 2, 0):a] = np.nan
    return x


cases = {
    "plain schedule": infant(),
    "+10 h trip, days 100-113": infant(shifts=[(100, 113, 10)]),
    "-6 h move from day 300": infant(shifts=[(300, N + 1, -6)]),
    "newborn chaos to day 60": infant(fragmented_until=60),
}
for name, x in cases.items():
    night, phase, R, eps = st.analyse_infant(x)
    ok = ~np.isnan(night[:, 0])
    med = np.median(night[ok], axis=0)
    print(f"{name}: nights {ok.sum()}, bedtime {med[0] % 24:.2f}, wake {med[1] % 24:.2f}, "
          f"mid {med[2] % 24:.2f}, length {med[3]:.2f} h")
    for e in eps:
        print("   jump:", e)
    if not eps:
        print("   no jumps")
