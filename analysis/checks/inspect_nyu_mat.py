"""Inspect the NYU infant-sleep .mat file. Usage: python3 -I inspect_nyu_mat.py <file.mat>"""
import sys

import numpy as np
from scipy.io import loadmat, whosmat

path = sys.argv[1]
print("variables:", whosmat(path))
m = loadmat(path)
for k, v in m.items():
    if k.startswith("__"):
        print(k, v if not isinstance(v, bytes) else v[:120])
        continue
    print(f"\n{k}: type {type(v).__name__}, dtype {getattr(v, 'dtype', None)}, shape {getattr(v, 'shape', None)}")

d = m["data"].astype(float)
n_days, n_bins, n_inf = d.shape
print(f"\ninfants {n_inf}, days {n_days}, bins per day {n_bins} ({1440 / n_bins:g} min each)")
vals = d[~np.isnan(d)]
u = np.unique(vals)
print("distinct non-NaN values:", u[:20], "..." if u.size > 20 else "", f"({u.size} total)")
print("value quantiles:", np.round(np.quantile(vals, [0, .01, .25, .5, .75, .99, 1]), 3))

# missingness: whole days missing vs partial
day_nan = np.isnan(d).all(1)            # (days, infants)
part_nan = np.isnan(d).any(1) & ~day_nan
print(f"\nmissing whole days: {day_nan.mean():.1%}; partially missing days: {part_nan.mean():.2%}")
present = (~day_nan).sum(0)
print("days present per infant: quantiles", np.quantile(present, [0, .1, .25, .5, .75, .9, 1]).astype(int))
print(f"infants missing >=50% of the 730 days: {(present <= n_days / 2).mean():.0%}")
first = np.array([np.argmax(~day_nan[:, i]) if (~day_nan[:, i]).any() else -1 for i in range(n_inf)])
last = np.array([n_days - 1 - np.argmax(~day_nan[::-1, i]) if (~day_nan[:, i]).any() else -1 for i in range(n_inf)])
print("first logged day (age) quantiles:", np.quantile(first, [0, .25, .5, .75, 1]).astype(int))
print("last logged day (age) quantiles:", np.quantile(last, [0, .25, .5, .75, 1]).astype(int))
print("infants with logs in each age month 1, 3, 6, 12, 18, 24:",
      [int((~day_nan[int(a * 30.4):int((a + 1) * 30.4)]).any(0).sum()) for a in (0, 2, 5, 11, 17, 23)])

# plausibility: share asleep per day, and day vs night
pres = ~day_nan
daily = np.nanmean(d, axis=1) * 24    # hours asleep per present day (if values are 0/1 or fractions)
print("\nhours asleep per logged day, quantiles:", np.round(np.nanquantile(daily[pres], [.05, .25, .5, .75, .95]), 1))
for lo, hi, lab in [(0, 30, "month 1"), (152, 183, "month 6"), (335, 365, "month 12"), (700, 730, "month 24")]:
    seg = d[lo:hi]
    night = np.nanmean(seg[:, 0:36]) if seg.size else np.nan   # 00:00-06:00 if bin 0 = midnight
    day = np.nanmean(seg[:, 72:108])                              # 12:00-18:00
    tot = np.nanmean(np.nanmean(seg, axis=1)) * 24
    print(f"{lab}: mean asleep 00-06 {night:.2f}, 12-18 {day:.2f}, total {tot:.1f} h/day")
