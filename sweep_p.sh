#!/bin/bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INDEX_P INDEX_ETA " >&2
  exit 1
fi

index_p="$1"
index_eta="$2"
echo "index_p: $index_p"
echo "index_eta: $index_eta"

python -m scripts.build_input \
  --mode p-index-sweep \
  --template input/general.yaml \
  --runs-dir data \
  --p-index-values 2.01 2.50 3.00\
  --sweep-index "data/${index_p}.csv" \
  --yes

python -m scripts.build_input \
  --mode f-e-sweep \
  --template input/general.yaml \
  --runs-dir data \
  --f-e-values 0.0005 0.005 0.05 \
  --sweep-index "data/${index_eta}.csv" \
  --yes

python -m scripts.plot_lightcurve_sweep \
  --f-e-sweep-index "data/${index_eta}.csv" \
  --p-index-sweep-index "data/${index_p}.csv" \
  --minimum-chi-squared-path  minimum_chi_squared.csv\
  --model-parameter-table-path model_parameter_table.csv \
  --candidate-lightcurves-path candidate_lightcurves.csv \
  --obslc ./data/obs/obslc.csv \
  --plotconfig plotconfigs/minimum_chi_squared_lightcurves.yaml \
  --output fig/helium_powerlaw_lightcurves.pdf \
  --frequency 9
