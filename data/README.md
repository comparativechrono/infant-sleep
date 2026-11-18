# Data

The raw data are not redistributed here. Put them where `run_all.sh` expects them, or pass
their paths as arguments: `./run_all.sh <path/to/infant_sleep_preprocessed.mat> <path/to/Agenoria/data>`.

## NYU infant sleep data (main cohort)

- **Source:** Liu S, Cheng M, Brooks H, Mackey W, Heeger DJ, Tabak EG, Fernandez-Granda C.
  *Time-series analysis via low-rank matrix factorization applied to infant-sleep data.*
  arXiv:1904.04780 (2019). The paper gives the data and code at
  `https://cims.nyu.edu/~sl5924/infant.html`. That page is no longer online; the files used here
  are the copies archived by the Internet Archive's Wayback Machine.
- **File:** `infant_sleep_preprocessed.mat`, variable `data`: 730 days × 144 ten-minute bins ×
  700 infants. Each value is the share of the bin logged as asleep (0 to 1 in steps of 0.1);
  days that were not logged are `NaN`. Row 1 is day 1 of age; bins start at midnight.
- **Checksum of the copy used:** SHA-256
  `9d8d9ccd80377da6b85c188b7defc579495ab04c78347b6c717d5ecf563b0c33` (18,225,938 bytes).
- **Expected path:** `data/nyu/infant_sleep_preprocessed.mat`

## Agenoria (external check)

- **Source:** Jiuguang Wang, *Agenoria*, <https://github.com/jiuguangw/Agenoria>, released under
  CC BY-NC-SA 4.0. Sleep sessions of two infants logged with the Glow Baby app
  (`data/zyw`: Baby 1, born 2018-11-21; `data/zlw`: Baby 2, born 2022-02-21).
- **Version used:** commit `4c6319f26fdea9b81154288c170b9af41d4fc835`.
- **Expected path:** clone the repository to `data/agenoria`:

  ```
  git clone https://github.com/jiuguangw/Agenoria data/agenoria
  git -C data/agenoria checkout 4c6319f26fdea9b81154288c170b9af41d4fc835
  ```
