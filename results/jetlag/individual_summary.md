# Do infants differ in how fast their sleep re-aligns?

Observed (32 episodes): median share left 0.62; SD 0.35; IQR 0.36; little or none lost (share left >= 0.9) 25%; at least half lost 28%

## One common time constant plus measurement error

| tau0 (days) | detected | median share left (mean over sets) | SD (median, 95% range) | IQR (median) |
|---|---|---|---|---|
| 3 | 0.12 | 0.75 | 0.20 (0.17-0.23) | 0.26 |
| 4 | 0.28 | 0.71 | 0.20 (0.17-0.23) | 0.32 |
| 5 | 0.35 | 0.65 | 0.23 (0.19-0.27) | 0.32 |
| 6 | 0.42 | 0.63 | 0.19 (0.15-0.22) | 0.25 |
| 7 | 0.47 | 0.66 | 0.19 (0.16-0.23) | 0.29 |
| 8 | 0.49 | 0.66 | 0.19 (0.16-0.23) | 0.28 |
| 9 | 0.52 | 0.69 | 0.19 (0.16-0.24) | 0.30 |
| 10 | 0.54 | 0.68 | 0.20 (0.16-0.25) | 0.28 |
| 12 | 0.58 | 0.71 | 0.21 (0.17-0.26) | 0.29 |
| 15 | 0.60 | 0.76 | 0.21 (0.18-0.25) | 0.30 |
| 20 | 0.64 | 0.77 | 0.19 (0.15-0.24) | 0.26 |
| inf | 0.72 | 0.98 | 0.16 (0.12-0.21) | 0.16 |

At every common time constant the simulated SD was below the observed 0.35: highest 97.5th percentile 0.27; largest P(simulated >= observed) 0.0000 (2000 sets each).

Best-matching common time constant: 6 days (median share left in simulated sets 0.63; observed 0.62).
- SD of share left: observed 0.35; simulated median 0.19 (95% range 0.15-0.22); P(simulated >= observed) = 0.0000
- IQR of share left: observed 0.36; simulated median 0.25 (95% range 0.17-0.36); P(simulated >= observed) = 0.0215
- Episodes losing little or none (share left >= 0.9): observed 25%; simulated median 10% (95% range 3-20%)
- Episodes losing at least half (share left <= 0.5): observed 28%; simulated median 23% (95% range 10-37%)

## Infants with more than one episode

- infant 220: day 569, 7 days, share left 0.71; day 589, 10 days, share left 0.36
- infant 237: day 249, 5 days, share left 0.61; day 498, 5 days, share left 0.94; day 605, 9 days, share left 0.32
- infant 387: day 309, 8 days, share left 0.35; day 589, 7 days, share left 0.55
- infant 598: day 176, 10 days, share left 0.07; day 714, 5 days, share left 0.69
- infant 631: day 147, 5 days, share left 0.52; day 356, 14 days, share left 0.81

Mean within-infant difference in share left 0.39; random pairings 0.38 (P(random <= observed) = 0.508; 5 infants, 11 episodes)

## Baseline traits (Spearman with the share of the shift lost)

- day-to-day variability before onset (h): rho = 0.21, p = 0.25 (n = 32)
- baseline sleep phase vs infants of the same age, in the shift's direction (h): rho = -0.24, p = 0.19 (n = 32)
- age at onset (days): rho = 0.28, p = 0.12 (n = 32)
- size of the shift (h): rho = -0.32, p = 0.07 (n = 32)
- duration (days): rho = 0.35, p = 0.05 (n = 32)
