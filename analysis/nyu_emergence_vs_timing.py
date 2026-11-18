"""Does the age at which the 24-h rhythm emerged predict later sleep timing?

Usage: python3 -I nyu_emergence_vs_timing.py <analysis_dir>
Reads nyu_infant_rhythm_summary.csv and nyu_infant_timing_summary.csv from <analysis_dir>;
writes nyu_emergence_vs_timing.csv (statistics) and nyu_emergence_vs_timing.png.

Primary analysis: eligible infants (logged from their first 3 weeks), emergence age t50 vs
12-month bedtime, wake time, mid-sleep and night-period length. Spearman rho with Holm correction
across those four; Theil-Sen slope with 95% CI as the effect size; partial Spearman controlling
for rhythm strength (plateau). Sensitivity: t_abs instead of t50, and the same outcomes at 6, 18
and 24 months.
"""
import sys
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperfig import headline, lettered, save  # noqa: E402


OUT = Path(sys.argv[1])
R = pd.read_csv(OUT / "nyu_infant_rhythm_summary.csv")
T = pd.read_csv(OUT / "nyu_infant_timing_summary.csv")
df = R.merge(T, on="infant")
df = df[df["eligible"]].copy()

OUTCOMES = {"bedtime": "Bedtime", "wake_time": "Wake time", "mid_sleep": "Mid-sleep",
            "night_period": "Night length"}
CLOCK = {"bedtime", "wake_time", "mid_sleep"}


def noon_ref(h):
    h = np.asarray(h, dtype=float)
    return np.where(h < 12, h + 24, h)


def outcome(col):
    base = col.rsplit("_", 1)[0]
    v = df[col].to_numpy(dtype=float)
    return noon_ref(v) if base in CLOCK else v


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(a) for a in (x, y, z))
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 3) / max(1 - r * r, 1e-12))
    return r, 2 * stats.t.sf(abs(t), n - 3)


rows = []
for pred in ("t50_day", "t_abs_day"):
    for age in ("6mo", "12mo", "18mo", "24mo"):
        for key in OUTCOMES:
            col = f"{key}_{age}"
            x, y, z = df[pred].to_numpy(float), outcome(col), df["plateau_s24"].to_numpy(float)
            ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
            if ok.sum() < 20:
                continue
            rho, p = stats.spearmanr(x[ok], y[ok])
            slope, _, lo, hi = stats.theilslopes(y[ok], x[ok])
            pr, pp = partial_spearman(x[ok], y[ok], z[ok])
            rows.append(dict(predictor=pred, age=age, outcome=key, n=int(ok.sum()),
                             spearman_rho=round(rho, 3), p=p,
                             theil_sen_min_per_10_days=round(slope * 600, 1),
                             ci95_low=round(lo * 600, 1), ci95_high=round(hi * 600, 1),
                             partial_rho_given_plateau=round(pr, 3), partial_p=pp))
res = pd.DataFrame(rows)
prim = (res.predictor == "t50_day") & (res.age == "12mo")
def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    adj, running = np.empty(p.size), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (p.size - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


res["p_holm_primary"] = np.nan
res.loc[prim, "p_holm_primary"] = holm(res.loc[prim, "p"])
for c in ("p", "partial_p", "p_holm_primary"):
    res[c] = res[c].map(lambda v: float(f"{v:.3g}") if pd.notna(v) else v)
for c in ("spearman_rho", "theil_sen_min_per_10_days", "ci95_low", "ci95_high",
          "partial_rho_given_plateau"):
    res[c] = res[c] + 0.0                      # turn -0.0 into 0.0
res.to_csv(OUT / "nyu_emergence_vs_timing.csv", index=False)
pd.set_option("display.width", 220)
print(res.to_string(index=False))
r_tp = stats.spearmanr(df["t50_day"], df["plateau_s24"])
print(f"\nt50 vs plateau: rho {r_tp[0]:.3f}, p {r_tp[1]:.3g}, n {len(df)}")

# ---------------- figure: the four primary panels ----------------
SURFACE, INK, INK_2, MUTED, HAIRLINE, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                              "#e1e0d9", "#c3c2b7")
BLUE = "#2a78d6"
for f in glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 9})
DPI = 220


def clock(h):
    h = float(h) % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


L, PW, GX, R_, TOP, PH, GY, BOTTOM = 1.0, 3.25, 0.85, 0.35, 2.3, 3.0, 1.55, 1.0
W, H = L + 2 * PW + GX + R_, TOP + 2 * PH + GY + BOTTOM
fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
for k, key in enumerate(OUTCOMES):
    r_, c_ = divmod(k, 2)
    ax = fig.add_axes([(L + c_ * (PW + GX)) / W, (BOTTOM + (1 - r_) * (PH + GY)) / H, PW / W, PH / H])
    col = f"{key}_12mo"
    x, y = df["t50_day"].to_numpy(float), outcome(col)
    ok = ~np.isnan(x) & ~np.isnan(y)
    ax.scatter(x[ok], y[ok], s=13, color=BLUE, alpha=0.55, linewidths=0, zorder=2)
    row = res[prim & (res.outcome == key)].iloc[0]
    slope, icpt, _, _ = stats.theilslopes(y[ok], x[ok])
    xx = np.array([x[ok].min(), x[ok].max()])
    ax.plot(xx, icpt + slope * xx, color=INK, linewidth=1.3, zorder=3)
    lo_, hi_ = np.nanquantile(y[ok], [0.005, 0.995])
    pad = (hi_ - lo_) * 0.12
    ax.set_ylim(lo_ - pad, hi_ + pad)
    if key in CLOCK:
        ticks = np.arange(np.ceil(lo_ - pad), np.floor(hi_ + pad) + 1, 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([clock(t) for t in ticks])
    else:
        ax.set_ylabel("hours", color=INK_2, fontsize=8.5)
    ax.set_xlim(0, max(175, x[ok].max() + 5))
    ax.grid(color=HAIRLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=8, length=3, width=0.6)
    num = lambda v, sign=False: (("+" if sign and v > 0 else "") + f"{v:.0f}").replace("-", "−") \
        if round(v) != 0 else "0"
    rho_s = f"{row.spearman_rho:.2f}".replace("-", "−") if abs(row.spearman_rho) >= 0.005 else "0.00"
    ax.annotate(lettered(f"{OUTCOMES[key]} at 12 months"), xy=(0, 1), xycoords="axes fraction", xytext=(0, 40),
                textcoords="offset points", fontsize=11, fontweight="semibold", color=INK, va="bottom")
    ax.annotate(f"ρ = {rho_s}, adjusted p = {row.p_holm_primary:.2g}\n"
                f"Slope {num(row.theil_sen_min_per_10_days, True)} min per 10 days later "
                f"(95% CI {num(row.ci95_low)} to {num(row.ci95_high, True)})",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                fontsize=8.5, color=INK_2, va="bottom", linespacing=1.4)
    if r_ == 1:
        ax.set_xlabel("Age the 24-h rhythm emerged (days)", color=INK_2, fontsize=9)

n12 = int(res[prim].n.iloc[0])
headline(fig, L / W, 1 - 0.36 / H, "Does early rhythm emergence predict later sleep?", fontsize=19,
         fontweight="bold", color=INK, ha="left", va="top")
headline(fig, L / W, 1 - 0.82 / H,
         f"{n12} NYU infants logged from their first 3 weeks and again at 11–13 months. Emergence = age the "
         "24-h\nsleep rhythm reached half its own plateau. Line: Theil–Sen fit. p adjusted (Holm) across "
         "the four panels.",
         fontsize=9, color=INK_2, ha="left", va="top", linespacing=1.45)
headline(fig, L / W, 0.18 / H, "Data: Liu et al., arXiv:1904.04780 (2019), infant_sleep_preprocessed.mat.",
         fontsize=7.5, color=MUTED, ha="left", va="bottom")
save(fig, OUT / "nyu_emergence_vs_timing.png", DPI, "Figure5", SURFACE)
print("figure saved")
