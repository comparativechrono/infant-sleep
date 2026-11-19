"""Adult sleep records from Open Humans public data, in the 10-minute format used for the infants.

Usage:
  python3 -I adult_data.py download <raw_dir>          fetch the public Fitbit and Oura files
  python3 -I adult_data.py build <raw_dir> <out.npz>   build the adult records

Source. Open Humans members can make data from their wearables public; the public data API needs
no login (https://www.openhumans.org/api/public-data/). We use the two sources that give the clock
time of each night's sleep in local time:
  Fitbit Connection (direct-sharing-102)  daily main sleep: start time, time in bed, minutes asleep
  Oura Connect (direct-sharing-184)       every sleep period with a 5-minute hypnogram; its start
                                          time carries the UTC offset, so the time zone is known
`download` lists all public files, keeps these two sources and writes manifest.csv (file id, size,
SHA-256). The listing is paged in no fixed order, so passes are repeated, 4 at a time, until a whole round adds nothing.
Members can withdraw data, so the manifest records what was analysed. `build` uses every readable
file in <raw_dir>, so build from a directory filled by `download` alone.

Format. One row per night: 144 bins of 10 minutes from 12:00 on the night's date to 12:00 the next
day, each holding the share of the bin asleep (0-1), in local clock time as recorded. A night with
no sleep record is NaN. Noon-to-noon rows keep each adult night in one row; the daily sleep phase
(circular mean clock time) does not depend on where the row starts.
  Fitbit  a night's main sleep runs from its start time for its time in bed, at the share asleep
          minutes asleep / time in bed. Fitbit dates a sleep by the day it ends, so a sleep whose
          start time plus time in bed passes midnight began on the previous day. Naps are not given.
  Oura    all sleep periods except 'rest'; a 5-minute epoch counts as asleep in stages 1-3.
Output npz: for each adult k, x_k (nights x 144, float32), first_k (ISO date of row 0), and for
Oura offset_k (UTC offset of each night's main sleep in hours, NaN if none); plus ids and sources.
"""
import csv
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

API = "https://www.openhumans.org/api/public-data/?limit=1000"
SOURCES = {"direct-sharing-102": "fitbit", "direct-sharing-184": "oura"}
MIN_NIGHTS = 30


def _get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def _wanted(f):
    kind = SOURCES.get(f["source"])
    return (kind == "fitbit" and f["basename"] == "fitbit-data.json") or \
           (kind == "oura" and f["basename"] == "oura-v2-sleep.json")


def _one_pass():
    found, url = {}, API
    while url:
        page = _get(url)
        for f in page["results"]:
            if _wanted(f):
                found[f["id"]] = f
        url = page.get("next")
    return found


def list_public(per_round=4, max_rounds=6):
    """The public listing is paged in no fixed order, so one pass can miss files. Passes are run
    4 at a time and their union kept, until a whole round adds nothing new (at least 2 rounds)."""
    from concurrent.futures import ThreadPoolExecutor
    seen = {}
    for k in range(max_rounds):
        with ThreadPoolExecutor(per_round) as ex:
            passes = list(ex.map(lambda _: _one_pass(), range(per_round)))
        new = 0
        for found in passes:
            for fid, f in found.items():
                new += fid not in seen
                seen[fid] = f                           # download links change between passes
        print(f"listing round {k + 1}: {len(seen)} files ({new} new); per pass "
              f"{[len(x) for x in passes]}", flush=True)
        if k >= 1 and new == 0:
            break
    return seen


def download(raw_dir):
    raw = Path(raw_dir)
    rows = []
    for fid, f in sorted(list_public().items()):
        kind = SOURCES[f["source"]]
        path = raw / kind / f"{fid}_{f['basename']}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            try:
                urllib.request.urlretrieve(f["download_url"], path)
            except urllib.error.HTTPError as err:          # listed, but withdrawn from storage
                rows.append({"file_id": fid, "source": kind, "bytes": 0, "json": False,
                             "sha256": f"unavailable (HTTP {err.code})"})
                continue
        data = path.read_bytes()
        ok = data[:1] in (b"{", b"[")
        rows.append({"file_id": fid, "source": kind, "bytes": len(data), "json": ok,
                     "sha256": hashlib.sha256(data).hexdigest()})
    with open(raw / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} files; {sum(r['json'] for r in rows)} readable")


def _series(d, key):
    out = {}
    for v in (d.get(key) or {}).values():
        if isinstance(v, dict):
            for lst in v.values():
                for rec in lst if isinstance(lst, list) else []:
                    if isinstance(rec, dict) and "dateTime" in rec:
                        out[rec["dateTime"]] = rec.get("value")
    return out


def _fill(x, first, start, end, weight):
    """Add [start, end) (naive local datetimes) at `weight` into noon-anchored rows of x."""
    t0 = dt.datetime.combine(first, dt.time(12))
    a, b = (start - t0).total_seconds() / 600, (end - t0).total_seconds() / 600
    for k in range(int(np.floor(a)), int(np.ceil(b))):
        r, c = divmod(k, 144)
        if 0 <= r < x.shape[0] and not np.isnan(x[r, c]):
            x[r, c] = min(1.0, x[r, c] + weight * (min(b, k + 1) - max(a, k)))


def _night_of(t):
    """Date of the noon-to-noon night containing naive datetime t."""
    return (t - dt.timedelta(hours=12)).date()


def fitbit_intervals(path):
    d = json.loads(Path(path).read_text())
    start, tib, asleep = _series(d, "sleep-start-time"), _series(d, "time-in-bed"), _series(d, "sleep-minutes")
    out = []
    for day, hhmm in start.items():
        try:
            minutes, slept = int(tib.get(day) or 0), int(asleep.get(day) or 0)
        except ValueError:
            continue
        if not hhmm or minutes <= 0:
            continue
        h, m = map(int, hhmm.split(":"))
        s0 = h * 60 + m
        begin = dt.datetime.fromisoformat(day) + dt.timedelta(minutes=s0 - (1440 if s0 + minutes >= 1440 else 0))
        out.append((begin, begin + dt.timedelta(minutes=minutes), min(1.0, slept / minutes), np.nan, True))
    return out


def oura_intervals(path):
    recs = json.loads(Path(path).read_text())
    out = []
    for r in recs if isinstance(recs, list) else recs.get("data", []):
        if r.get("type") == "rest" or not r.get("bedtime_start") or not r.get("sleep_phase_5_min"):
            continue
        t = dt.datetime.fromisoformat(r["bedtime_start"])
        off = t.utcoffset().total_seconds() / 3600
        local = t.replace(tzinfo=None)
        for j, stage in enumerate(r["sleep_phase_5_min"]):
            if stage in "123":
                b = local + dt.timedelta(minutes=5 * j)
                out.append((b, b + dt.timedelta(minutes=5), 1.0, off, r.get("type") == "long_sleep"))
    return out


def build(raw_dir, out_path):
    raw = Path(raw_dir)
    people = []
    for kind, fn in (("fitbit", fitbit_intervals), ("oura", oura_intervals)):
        for p in sorted((raw / kind).glob("*.json")):
            try:
                iv = fn(p)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue                          # file withdrawn or not JSON
            if iv:
                people.append((kind, p.name.split("_")[0], iv))
    arrays, ids, sources = {}, [], []
    for kind, fid, iv in people:
        nights = sorted({_night_of(b + (e - b) / 2) for b, e, *_ in iv})
        if len(nights) < MIN_NIGHTS:
            continue
        first, last = nights[0], nights[-1]
        n = (last - first).days + 1
        x = np.full((n, 144), np.nan, dtype=np.float32)
        off = np.full(n, np.nan, dtype=np.float32)
        for night in nights:
            x[(night - first).days] = 0.0
        for b, e, w, o, main in iv:
            _fill(x, first, b, e, w)
            r = (_night_of(b + (e - b) / 2) - first).days
            if main and not np.isnan(o):
                off[r] = o
        k = len(ids)
        arrays[f"x_{k}"], arrays[f"first_{k}"] = x, np.array(first.isoformat())
        if kind == "oura":
            arrays[f"offset_{k}"] = off
        ids.append(f"{kind}-{fid}")
        sources.append(kind)
        print(f"{ids[-1]:24s} nights {len(nights):5d} of {n:5d}  {first} .. {last}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=np.array(ids), sources=np.array(sources), **arrays)
    print(f"{len(ids)} adults written to {out_path}")


if __name__ == "__main__":
    if sys.argv[1] == "download":
        download(sys.argv[2])
    else:
        build(sys.argv[2], sys.argv[3])
