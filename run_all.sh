#!/usr/bin/env bash
# Reproduce every result table and figure in this repository.
#
# Usage:  ./run_all.sh [path/to/infant_sleep_preprocessed.mat] [path/to/Agenoria/data]
# Defaults: data/nyu/infant_sleep_preprocessed.mat and data/agenoria/data (see data/README.md).
# Takes roughly 20-40 minutes on a laptop; the periodograms of the 700 NYU infants and the
# surrogate cohorts of the shift screen dominate.
set -euo pipefail
cd "$(dirname "$0")"

MAT=${1:-data/nyu/infant_sleep_preprocessed.mat}
AGEN=${2:-data/agenoria/data}
PY="python3 -I"
A=analysis

echo "== NYU cohort: 24-h rhythm emergence"
$PY $A/nyu_rhythm_analysis.py "$MAT" results/nyu
echo "== NYU cohort: nightly sleep timing and time-zone jump screen"
$PY $A/nyu_sleep_timing.py "$MAT" results/nyu
echo "== Agenoria infants: external check"
$PY $A/agenoria_emergence.py "$AGEN" results/nyu results/agenoria
echo "== Robustness checks"
$PY $A/robustness.py "$MAT" results/nyu results/robustness

echo "== Figures, annotated versions (figures/exploratory)"
export FIG_DIR=figures/exploratory
$PY $A/nyu_emergence_vs_timing.py results/nyu        # also writes the statistics table
$PY $A/nyu_population_heatmap.py "$MAT" $FIG_DIR/nyu_700_infants_sleep_by_age.png
$PY $A/nyu_rhythm_figures.py "$MAT" results/nyu
$PY $A/nyu_timing_figures.py "$MAT" results/nyu
$PY $A/agenoria_figure.py results/agenoria results/nyu
$PY $A/agenoria_sleep_chart.py "$AGEN" results/agenoria
$PY $A/agenoria_actograms.py "$AGEN" results/agenoria
unset FIG_DIR

echo "== Figures, journal versions (figures/paper)"
$PY $A/paper_figures.py "$MAT" results figures/paper

echo "== Numbers quoted in the paper"
$PY $A/paper_numbers.py results > results/paper_numbers.md

echo "== Letter: sudden shifts in clock time"
$PY $A/jetlag_screen.py results results/jetlag 20
$PY $A/jetlag_profiles.py "$MAT" results 3
$PY $A/jetlag_profiles.py "$MAT" results 2.5
$PY $A/jetlag_simulate.py "$MAT" results 30
$PY $A/jetlag_individual.py "$MAT" results 80
$PY $A/letter_numbers.py results > results/jetlag/letter_numbers.md
$PY $A/letter_figures.py "$MAT" results figures/letter

echo "== Tests"
$PY $A/test_nyu_rhythm.py
$PY $A/test_nyu_timing.py
echo "done"
