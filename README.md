# infant-sleep

Analysis code, result tables and figures for the manuscript

> **Emergence of the 24-hour sleep–wake rhythm in individual infants: periodogram analysis of
> parent-logged sleep from 700 infants.** In preparation for the *Journal of Biological Rhythms*.

The study times, infant by infant, when a 24-hour rhythm emerges in parent-logged sleep, using
sliding Lomb–Scargle periodograms of 700 infants from the NYU infant sleep dataset
(Liu et al., 2019), and checks the method on two independently logged infants (Agenoria).

## Contents

| Folder | What is in it |
| --- | --- |
| `analysis/` | All scripts (Python 3). Core analyses, figure scripts, tests, and `checks/` (data inspection and figure verification) |
| `results/nyu/` | Per-infant and per-week tables for the NYU cohort, plus the saved periodogram and timing arrays (`.npz`) |
| `results/agenoria/` | Results for the two Agenoria infants |
| `results/robustness/` | Surrogate tests, definition sensitivity and free-running detection power |
| `results/paper_numbers.md` | Every number quoted in the paper, regenerated from the tables above |
| `figures/paper/` | Journal versions of the figures: Figures 1–6 and S1–S4, as 300-dpi PNG and vector PDF |
| `figures/exploratory/` | The annotated versions of the same figures made during the analysis |
| `data/` | Not included; `data/README.md` says where to get both datasets |

## Reproducing everything

```
pip install -r requirements.txt
# put the two datasets in place as described in data/README.md, then
./run_all.sh
```

`run_all.sh` runs every step below in order and takes a few minutes on a laptop. Each script can
also be run on its own; its docstring gives its arguments.

| Step | Script | Writes |
| --- | --- | --- |
| 24-h rhythm emergence, 700 infants | `nyu_rhythm_analysis.py` | `results/nyu/nyu_infant_rhythm_summary.csv`, `nyu_infant_weekly_rhythm.csv`, `nyu_rhythm_windows.npz` |
| Nightly sleep timing and time-zone jump screen | `nyu_sleep_timing.py` | `results/nyu/nyu_infant_timing_summary.csv`, `nyu_sleep_timing_weekly.csv`, `nyu_phase_jumps.csv`, `nyu_timing_arrays.npz` |
| Emergence age against later sleep timing | `nyu_emergence_vs_timing.py` | `results/nyu/nyu_emergence_vs_timing.csv`, Figure 5 |
| External check on the Agenoria infants | `agenoria_emergence.py` | `results/agenoria/agenoria_emergence_summary.csv`, `agenoria_s24_by_day.csv` |
| Robustness checks | `robustness.py` | `results/robustness/*.csv` |
| Figures | `nyu_population_heatmap.py`, `nyu_rhythm_figures.py`, `nyu_timing_figures.py`, `agenoria_figure.py`, `agenoria_sleep_chart.py`, `agenoria_actograms.py` | `figures/` |
| Numbers for the paper | `paper_numbers.py` | `results/paper_numbers.md` |
| Tests on synthetic infants | `test_nyu_rhythm.py`, `test_nyu_timing.py` | printed checks |

Figure scripts write the annotated version by default. With `PAPER_FIGURES=<folder>` set they
write the journal version instead (see `analysis/paperfig.py`). They use the Inter typeface if it
is installed and fall back to the matplotlib default otherwise.

## Method in brief

- **Input.** For each infant, one row per day of age and 144 ten-minute bins holding the share of the
  bin logged as asleep; days without a log are missing.
- **Periodograms.** Lomb–Scargle periodograms (periods 2–30 h) in sliding 14-day windows advanced by
  one day, kept when at least 10 days in the window were logged. The 24-h strength of a window is
  the share of variance explained by a 24-h sinusoid.
- **Plateau and emergence age.** An infant's plateau is the median 24-h strength of windows centred
  on days 120–240. The half-plateau age is the first age at which the 24-h strength reaches half
  the plateau and stays there in at least 80% of the next 14 windows. It is computed for infants
  whose first usable window is centred by day 21 and whose plateau is at least 0.10 (260 infants).
- **Free-running screen.** At least 14 consecutive windows whose strongest 20–28 h peak lies at
  least 0.75 h from 24 h, explains at least 5% of variance and has at least 1.5 times the 24-h power.
- **Sleep timing.** Each night's main sleep period (sleep bouts bridged across wakes of up to 60 min,
  4–16 h long, midpoint between 18:00 and 09:00), with a circular screen for jumps in sleep phase
  of 4 h or more, such as travel across time zones.

## Data sources and licences

- **NYU infant sleep data:** Liu S, Cheng M, Brooks H, Mackey W, Heeger DJ, Tabak EG,
  Fernandez-Granda C. *Time-series analysis via low-rank matrix factorization applied to
  infant-sleep data.* arXiv:1904.04780 (2019). The data were released with the paper; please cite it.
- **Agenoria:** Jiuguang Wang, <https://github.com/jiuguangw/Agenoria>, CC BY-NC-SA 4.0. The
  tables in `results/agenoria/` and Figures 6 and S2–S4 are adaptations of these data and are
  shared under the same licence.
