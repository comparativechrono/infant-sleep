"""Checks for adult_data.py and the noon-anchored phase of adult_screen.py on synthetic records."""
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adult_data as ad  # noqa: E402
import nyu_sleep_timing as st  # noqa: E402

ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


# Fitbit dates a sleep by the day it ends: 23:30 + 8 h on 2020-01-02 began on 01-01; 01:00 + 7 h began on 01-02
fit = {"sleep-start-time": {"2020": {"sleep-startTime": [{"dateTime": "2020-01-02", "value": "23:30"},
                                                         {"dateTime": "2020-01-03", "value": "01:00"}]}},
       "time-in-bed": {"2020": {"sleep-timeInBed": [{"dateTime": "2020-01-02", "value": "480"},
                                                    {"dateTime": "2020-01-03", "value": "420"}]}},
       "sleep-minutes": {"2020": {"sleep-minutesAsleep": [{"dateTime": "2020-01-02", "value": "432"},
                                                          {"dateTime": "2020-01-03", "value": "420"}]}}}
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "f.json"
    p.write_text(json.dumps(fit))
    iv = sorted(ad.fitbit_intervals(p))
check("sleep crossing midnight starts the day before its Fitbit date",
      iv[0][0] == dt.datetime(2020, 1, 1, 23, 30) and iv[0][1] == dt.datetime(2020, 1, 2, 7, 30))
check("sleep after midnight starts on its Fitbit date", iv[1][0] == dt.datetime(2020, 1, 3, 1, 0))
check("share asleep is minutes asleep / time in bed", abs(iv[0][2] - 0.9) < 1e-9)

# noon-anchored rows: a 23:00-07:00 night falls in one row, and its phase is the 03:00 midpoint
first = dt.date(2020, 1, 1)
x = np.zeros((3, 144))
ad._fill(x, first, dt.datetime(2020, 1, 1, 23, 0), dt.datetime(2020, 1, 2, 7, 0), 1.0)
check("a night sits in one noon-to-noon row", x[0].sum() == 48 and x[1:].sum() == 0)
ph, R = st.daily_phase(np.roll(x, 72, axis=1))
check("phase of the noon-anchored row is the clock midpoint", abs(st.wrap(ph[0] - 3.0)) < 0.01)
check("an empty night has no phase", np.isnan(ph[1]))
check("the night of a datetime is the noon-to-noon window", ad._night_of(dt.datetime(2020, 1, 2, 3, 0)) == first)

print("all passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
