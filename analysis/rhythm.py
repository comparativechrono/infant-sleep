"""Shared helpers: load Glow Baby sleep exports, build minute states, chi-square periodogram."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_sessions(data_dir: Path, code: str) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / code / "glow_sleep.csv", names=["begin", "end"], header=0)
    for c in ("begin", "end"):
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y %I:%M:%S %p")
    return df.drop_duplicates().sort_values("begin").reset_index(drop=True)


def minute_states(df: pd.DataFrame, bday: pd.Timestamp, n_days: int) -> np.ndarray:
    """0 = no record, 1 = awake (not logged as asleep), 2 = asleep. Shape (n_days, 1440).
    Day 0 is the date of birth; times are local clock time as logged, floored to the minute."""
    n = n_days * 1440
    t0 = bday.value // 60_000_000_000
    to_min = lambda ts: ts.value // 60_000_000_000 - t0
    state = np.ones(n, dtype=np.int8)
    first, last = to_min(df["begin"].min()), to_min(df["end"].max())
    state[: max(first, 0)] = 0
    if last < n:
        state[max(last, 0):] = 0
    for b, e in zip(df["begin"], df["end"]):
        i, j = max(to_min(b), 0), min(to_min(e), n)
        if j > i:
            state[i:j] = 2
    return state.reshape(n_days, 1440)


def sleep_fraction(states: np.ndarray, width: int) -> np.ndarray:
    """Fraction of recorded minutes asleep in each `width`-minute bin (NaN if none recorded)."""
    s = states.reshape(states.shape[0], -1, width)
    rec = (s != 0).sum(-1)
    asleep = (s == 2).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rec > 0, asleep / rec, np.nan)


def chi2_periodogram(x: np.ndarray, periods: np.ndarray):
    """Sokolove-Bushell chi-square periodogram of a regularly sampled series (NaN = missing).

    For each trial period P (in samples) the series is folded at P and treated as a one-way
    ANOVA over phase bins. Returns (Qp, eta2_adj): Qp = N * SS_between / SS_total (chi-square,
    P - 1 df under the independence null) and the bias-adjusted share of variance explained,
    1 - (1 - eta2) * (N - 1) / (N - P), which removes the upward drift with longer periods.

    Caveat: a rhythm of period P scores (almost) equally at 2P, 3P, ..., so a regular 3-h
    ultradian rhythm also scores at 24 h. Prefer lomb_scargle() when ultradian rhythms matter.
    """
    valid = ~np.isnan(x)
    idx = np.flatnonzero(valid)
    xv = x[valid]
    n = xv.size
    qp = np.full(periods.size, np.nan)
    adj = np.full(periods.size, np.nan)
    if n == 0:
        return qp, adj
    m = xv.mean()
    ss_tot = ((xv - m) ** 2).sum()
    if ss_tot == 0:
        return qp, adj
    for k, p in enumerate(periods):
        ph = idx % p
        cnt = np.bincount(ph, minlength=p)
        sm = np.bincount(ph, weights=xv, minlength=p)
        used = cnt > 0
        means = sm[used] / cnt[used]
        eta2 = (cnt[used] * (means - m) ** 2).sum() / ss_tot
        g = used.sum()
        qp[k] = n * eta2
        adj[k] = 1 - (1 - eta2) * (n - 1) / (n - g)
    return qp, adj


def lomb_scargle(t_h: np.ndarray, x: np.ndarray, periods_h: np.ndarray) -> np.ndarray:
    """Classic Lomb-Scargle periodogram, 'standard' normalisation.

    t_h: sample times in hours; x: values (NaN = missing); periods_h: trial periods in hours.
    Returns, for each period, the share of variance explained by the best-fitting sinusoid
    (0..1). A non-sinusoidal rhythm adds weaker harmonics at P/2, P/3, ... but nothing at 2P.
    """
    valid = ~np.isnan(x)
    t, xv = t_h[valid], x[valid] - np.nanmean(x)
    ss = xv @ xv
    out = np.full(periods_h.size, np.nan)
    if xv.size < 3 or ss == 0:
        return out
    w = 2 * np.pi / periods_h
    wt = np.outer(w, t)
    tau = np.arctan2(np.sin(2 * wt).sum(1), np.cos(2 * wt).sum(1)) / (2 * w)
    arg = wt - (w * tau)[:, None]
    c, s = np.cos(arg), np.sin(arg)
    return ((c @ xv) ** 2 / (c * c).sum(1) + (s @ xv) ** 2 / (s * s).sum(1)) / ss


def ls_basis(t_h: np.ndarray, periods_h: np.ndarray):
    """Precompute the trig matrices lomb_scargle_windows() needs (reuse across many calls)."""
    wt = np.outer(t_h, 2 * np.pi / periods_h)            # (N, F)
    return np.cos(wt), np.sin(wt), np.cos(2 * wt), np.sin(2 * wt)


def lomb_scargle_windows(X: np.ndarray, t_h: np.ndarray, periods_h: np.ndarray,
                         basis=None) -> np.ndarray:
    """Vectorised lomb_scargle() for many windows sampled at the same relative times.

    X: (n_windows, N) values at times t_h (hours), NaN = missing. Returns (n_windows, n_periods),
    the share of each window's variance explained by the best-fitting sinusoid at each period.
    Uses sum identities so that each window costs four matrix products instead of a loop.
    Pass basis=ls_basis(t_h, periods_h) to skip recomputing the trig matrices.
    """
    cos1, sin1, cos2, sin2 = basis if basis is not None else ls_basis(t_h, periods_h)
    M = ~np.isnan(X)
    Mf = M.astype(float)
    n = Mf.sum(1, keepdims=True)
    mean = np.nansum(X, 1, keepdims=True) / np.maximum(n, 1)
    Xc = np.where(M, X - mean, 0.0)
    ss = (Xc * Xc).sum(1, keepdims=True)
    C2, S2 = Mf @ cos2, Mf @ sin2                         # sums of cos/sin(2wt) over valid samples
    XC, XS = Xc @ cos1, Xc @ sin1
    half = np.arctan2(S2, C2) / 2                         # w * tau
    c, s = np.cos(half), np.sin(half)
    CC, SS, CS = (n + C2) / 2, (n - C2) / 2, S2 / 2
    A, B = c * XC + s * XS, c * XS - s * XC
    Dc = c * c * CC + 2 * c * s * CS + s * s * SS
    Ds = s * s * CC - 2 * c * s * CS + c * c * SS
    with np.errstate(invalid="ignore", divide="ignore"):
        P = (A * A / Dc + B * B / Ds) / ss
    P[(n[:, 0] < 3) | (ss[:, 0] == 0)] = np.nan
    return P
