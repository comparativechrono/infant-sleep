"""Show short (<=2 min) sleep sessions in context. Usage: python3 -I peek_short_sessions.py <data_dir>"""
import sys
from pathlib import Path
import pandas as pd

DATA = Path(sys.argv[1])
for code, bday in {"zyw": "2018-11-21", "zlw": "2022-02-21"}.items():
    bday = pd.Timestamp(bday)
    df = pd.read_csv(DATA / code / "glow_sleep.csv", names=["begin", "end"], header=0)
    for c in ("begin", "end"):
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y %I:%M:%S %p")
    df = df.drop_duplicates().sort_values("begin").reset_index(drop=True)
    df["dur"] = (df.end - df.begin).dt.total_seconds() / 60
    df["age_d"] = (df.begin - bday).dt.days
    short = df.dur <= 2
    df["gap_before"] = (df.begin - df.end.shift()).dt.total_seconds() / 60
    df["gap_after"] = (df.begin.shift(-1) - df.end).dt.total_seconds() / 60
    df["next_dur"] = df.dur.shift(-1)
    s = df[short]
    print(f"\n=== {code}: {short.sum()} short sessions ===")
    print("by age month:", (s.age_d // 30.44).astype(int).value_counts().sort_index().to_dict())
    print("by clock hour:", s.begin.dt.hour.value_counts().sort_index().to_dict())
    print("gap to next session (min) quantiles:", s.gap_after.quantile([.25, .5, .75]).round(0).to_dict())
    print("share followed by a >=30 min session within 30 min:",
          round(((s.gap_after <= 30) & (s.next_dur >= 30)).mean(), 2))
    # one illustrative day around age 60 and 200
    for age in (60, 200):
        print(f"--- age {age} d ---")
        print(df[df.age_d == age][["begin", "end", "dur"]].to_string(index=False))
