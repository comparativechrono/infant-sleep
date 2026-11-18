# Numbers quoted in the Letter on sudden shifts in clock time

Threshold 3 h; main set = episodes starting at day 90 or later with shifts below 11 h. Deviations are measured from a baseline trend (see jetlag_profiles.py).

## Screen and null model

| Threshold (h) | Episodes, all ages | Infants | Episodes from day 90 | Infants | Expected from ordinary variation, from day 90 (mean, 95% range) |
|---|---|---|---|---|---|
| 2 | 181 | 136 | 121 | 94 | 28.1 (22–33) |
| 2.5 | 75 | 64 | 49 | 42 | 6.0 (2–9) |
| 3 | 44 | 38 | 32 | 26 | 1.1 (0–3) |
| 3.5 | 20 | 19 | 15 | 14 | 0.3 (0–1) |
| 4 | 11 | 10 | 9 | 8 | 0.0 (0–0) |

- At 3 h, all ages: 44 episodes in 38 infants; 12 started before day 90 (excluded from the main set).
- Episodes per 1,000 usable infant-days: days 42–89 0.59 (12/20,250); from day 90 0.14 (32/234,371).
- Infants with at least 30 usable days from day 90: 683; with a main-set episode: 26 (3.8%); with more than one: 5.
- One episode per 20.1 infant-years of logging from day 90.
- Expected false episodes from day 90: 1.15 of 32 (4%).

## Main set: size, direction, timing

- 32 episodes in 26 infants; sleep later than before in 21, earlier in 11 (two-sided binomial p = 0.11).
- Shift on the first 2 days: median 4.0 (IQR 3.4 to 4.9) h; range 2.3–6.0.
- Detector's estimate of the shift (mean over the episode): median 3.8 (IQR 3.2 to 4.6) h.
- Duration: median 6.5 (IQR 5.0 to 10.0) days; range 3–87.
- Age at onset: median 256 (IQR 147 to 418) days; range 94–714.
- Abrupt onset (day before < half the shift): 22/29 (76%); 3 had no logged day in the 3 days before.
- At least one unlogged day in the 3 days before onset: 9/32 (28%); background in the same infants from day 90: 2.6% of usable days (binomial p = 1e-07).
- At least one unlogged day in the 3 days after the end: 4/32 (12%); background 2.8% (p = 0.011).
- Day-to-day scatter of the deviation in the 14 days before onset (SD): median 0.90 (IQR 0.63 to 1.65) h.

## Course while shifted

- Change from the first 2 to the last 2 days: median -1.73 (IQR -2.12 to -0.49) h; share of the shift left at the end: median 0.62 (IQR 0.50 to 0.86).
- Shrank by at least a quarter: 22/32 (69%); did not shrink at all: 6.
- Time constant where the shift shrank (26 episodes): median 8.0 (IQR 5.6 to 16.3) days; rate of re-alignment, all episodes: median 0.28 (IQR 0.07 to 0.36) h/day.
- Change different from 0: Wilcoxon p = 0.0001.
- Abrupt end (drop of at least half the shift from the last episode day to the next): 13/32 (41%); median duration 5 days (abrupt end) and 7 days (gradual).
- Hours asleep per logged day: before 12.5 (IQR 12.0 to 13.3); first 3 days 12.5 (IQR 11.4 to 12.9); paired difference median -0.05 h (Wilcoxon p = 0.35).

## What the speed of re-alignment relates to (Spearman, share of the shift lost while it lasted)

- age at onset: rho = +0.28, p = 0.12, n = 32
- size of the shift: rho = -0.32, p = 0.07, n = 32
- duration: rho = +0.35, p = 0.05, n = 32
- 24-h rhythm strength (plateau, paper 1): rho = +0.29, p = 0.11, n = 31
- Later vs earlier shifts: median share lost 0.38 vs 0.31 (Mann–Whitney p = 0.66).

## Simulated shifts (detector at 3 h; 30 per condition)

Detection, averaged over durations (4, 7, 14 days) and residual offsets:

| Size (h) | No re-alignment | Time constant 10 days | Time constant 4 days |
|---|---|---|---|
| 3 | 18% | 7% | 2% |
| 4 | 87% | 64% | 14% |
| 5 | 98% | 89% | 41% |
| 6 | 99% | 97% | 79% |

- Shifts of 4–6 h, no re-alignment: measured change median +0.01 h (IQR -0.44 to +0.35); abrupt end 96%; abrupt onset 98%; share of the shift left at the end median 1.00; shift on first 2 days 5.11 h vs true 5 h (median).

- Shifts of 4–6 h, time constant 10 days: measured change median -1.77 h (IQR -2.44 to -1.04); abrupt end 44%; abrupt onset 100%; share of the shift left at the end median 0.63; shift on first 2 days 4.92 h vs true 5 h (median).

- Shifts of 4–6 h, time constant 4 days: measured change median -1.92 h (IQR -2.49 to -1.25); abrupt end 22%; abrupt onset 100%; share of the shift left at the end median 0.62; shift on first 2 days 4.92 h vs true 6 h (median).
- True time constant 10 days: estimated median 10.6 days (IQR 8.5 to 13.3).
- True time constant 4 days: estimated median 4.9 days (IQR 4.1 to 6.0).
- No re-alignment: estimated time constant infinite in 51%, otherwise median 69 days.

## Sensitivity: threshold 2.5 h

- From day 90: 49 episodes in 42 infants; expected from ordinary variation 6.0.
- Abrupt onset 35/45 (78%); change median -1.11 (IQR -1.80 to -0.33) h; shrank by at least a quarter 29/49 (59%); abrupt end 27/47 (57%); unlogged day before onset 14/49 (29%).
