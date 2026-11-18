"""Independent checks of the actogram/periodogram figures.

Usage: python3 -I verify_agenoria_actograms.py <agenoria_data_dir> <out_dir>
"""
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BABIES = [("zyw", "Baby 1", datetime(2018, 11, 21)), ("zlw", "Baby 2", datetime(2022, 2, 21))]
N_DAYS, ACT_BIN, PG_BIN, WIN, PAD = 365, 3, 10, 14, 7
PERIODS = np.geomspace(2, 30, 400)
DPI, L, ACT_W, TOP, PANEL_H = 220, 1.0, 4.2, 2.8, 8.3
COL = {0: (0xE1, 0xE0, 0xD9), 1: (0xFC, 0xFC, 0xFB), 2: (0x0B, 0x0B, 0x0B)}
rng = np.random.default_rng(11)
ok = True

for code, label, bday in BABIES:
    tag = label.replace(" ", "")
    act = np.load(OUT / f"_act_{tag}.npy")
    heat = np.load(OUT / f"_heat_{tag}.npy")
    # raw sessions parsed with the standard library, minutes since birth-date midnight
    iv = set()
    with open(DATA / code / "glow_sleep.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            b = datetime.strptime(row["Begin time"], "%m/%d/%Y %I:%M:%S %p")
            e = datetime.strptime(row["End time"], "%m/%d/%Y %I:%M:%S %p")
            iv.add((int((b - bday).total_seconds() // 60), int((e - bday).total_seconds() // 60)))
    iv = np.array(sorted(iv))
    first, last = iv[:, 0].min(), iv[:, 1].max()

    def state_of(minute):
        if minute < first or minute >= last:
            return 0
        return 2 if ((iv[:, 0] <= minute) & (minute < iv[:, 1])).any() else 1

    # 1) random actogram cells vs raw intervals (double-plot: right half = next day)
    bad = 0
    for _ in range(1500):
        a, c = int(rng.integers(0, N_DAYS)), int(rng.integers(0, act.shape[1]))
        day = a + (c >= 480)
        m0 = day * 1440 + (c % 480) * ACT_BIN
        sts = [state_of(m) for m in range(m0, m0 + ACT_BIN)]
        exp = 2 if 2 in sts else (1 if 1 in sts else 0)
        bad += exp != act[a, c]
    print(f"{label}: actogram cells checked 1500, mismatches {bad}")
    ok &= bad == 0

    # 2) rendered pixels vs actogram cells (cells inside uniform runs only)
    img = np.asarray(Image.open(OUT / f"actogram_periodogram_{tag.lower()}.png").convert("RGB")).astype(int)
    left, top, pw, ph = L * DPI, TOP * DPI, ACT_W * DPI, PANEL_H * DPI
    bad = n = 0
    while n < 600:
        a, c = int(rng.integers(0, N_DAYS)), int(rng.integers(3, act.shape[1] - 3))
        if not (act[a, c - 3:c + 4] == act[a, c]).all():
            continue
        x = int(left + (c + 0.5) / act.shape[1] * pw)
        y = int(top + (a + 0.5) / N_DAYS * ph)
        bad += np.abs(img[y, x] - COL[int(act[a, c])]).max() > 12
        n += 1
    print(f"{label}: actogram pixels checked {n}, mismatches {bad}")
    ok &= bad == 0

    # 3) periodogram rows vs a direct least-squares sinusoid fit on the same window
    worst = 0.0
    for a in (20, 100, 250):
        lo = (a - WIN // 2) * 1440
        mins = np.arange(lo, lo + WIN * 1440)
        st = np.array([state_of(m) for m in mins]).reshape(-1, PG_BIN)
        rec, slp = (st != 0).sum(1), (st == 2).sum(1)
        x = np.where(rec > 0, slp / np.maximum(rec, 1), np.nan)
        t = (np.arange(x.size) + 0.5) * PG_BIN / 60
        v = ~np.isnan(x)
        xc = x[v] - x[v].mean()
        for k in (int(np.argmin(abs(PERIODS - p))) for p in (3.5, 8, 12, 24)):
            w = 2 * np.pi / PERIODS[k]
            X = np.column_stack([np.cos(w * t[v]), np.sin(w * t[v])])
            beta, *_ = np.linalg.lstsq(X, xc, rcond=None)
            r2 = 1 - ((xc - X @ beta) ** 2).sum() / (xc @ xc)
            worst = max(worst, abs(r2 - heat[a, k]))
    print(f"{label}: periodogram vs least-squares fit, worst |diff| = {worst:.2e}")
    ok &= worst < 1e-6

    nan_rows = np.flatnonzero(np.isnan(heat).all(1))
    print(f"{label}: rows without a full window: {nan_rows.tolist()}")

print("\nALL CHECKS PASS" if ok else "\nSOME CHECKS FAILED")
