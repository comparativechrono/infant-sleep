# Data

The raw data are not redistributed here. Put them where `run_all.sh` expects them, or pass
their paths as arguments:
`./run_all.sh <path/to/infant_sleep_preprocessed.mat> <path/to/Agenoria/data> <path/to/openhumans>`.

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

## Open Humans public wearable data (adult comparison in the Letter)

- **Source:** [Open Humans](https://www.openhumans.org), where members can choose to make data from
  their wearables public. Greshake Tzovaras B, Angrist M, Arvai K, et al. *Open Humans: a platform
  for participant-centered research and personal data exploration.* GigaScience 8(6):giz076 (2019).
- **Files:** the public files of two Open Humans projects: Fitbit Connection
  (`direct-sharing-102`, `fitbit-data.json`: daily main-sleep start time, time in bed and minutes
  asleep) and Oura Connect (`direct-sharing-184`, `oura-v2-sleep.json`: every sleep period with a
  5-minute hypnogram and the UTC offset of its time stamps).
- **Getting them:** `python3 -I analysis/adult_data.py download data/openhumans` (`run_all.sh` does
  this on its first run). It lists the public files through the Public Data API, which needs no
  login, downloads these two kinds and writes `data/openhumans/manifest.csv` with each file's id,
  size and SHA-256. The listing is paged in no fixed order, so it is repeated until a whole pass adds
  nothing new; this takes about 10 minutes.
- **Version used:** the files analysed for the Letter, downloaded on 19 November 2026, are listed with
  their checksums in `results/adult/openhumans_manifest.csv`. Members can withdraw or change their
  data, so a later download may differ.
- **Terms:** Open Humans asks users of public data to cite the platform and the projects that
  produced the data, not to try to identify members, and to point to its guidelines when passing the
  data on: <https://www.openhumans.org/community-guidelines/#public-data>. The raw files are not
  redistributed here, and the outputs in `results/adult/` label adults A01, A02, ... and carry no
  calendar dates.
- **Expected path:** `data/openhumans`
