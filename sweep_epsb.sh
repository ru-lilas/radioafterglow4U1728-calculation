#!/bin/bash

set -e

if [[ "$1" == "--full" ]]; then
  python -m scripts.build_input \
    --mode eps-b-sweep \
    --template input/general.yaml \
    --runs-dir data \
    --eps-b-values 0.0001 0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 \
    --sweep-index data/p_sweep_full.csv \
    --yes
else
  eps_b_values=$(python -c 'import numpy as np; print(*np.geomspace(1e-3, 1e-1, 8))')
  python -m scripts.build_input \
    --mode eps-b-sweep \
    --template input/general.yaml \
    --runs-dir data \
    --eps-b-values $eps_b_values \
    --sweep-index data/epsb_sweep_geom.csv \
    --yes
fi

if [[ "$1" == "--combine" ]]; then
  python -m scripts.combine_energetics \
    --sweep-index data/epsb_sweep_geom.csv \
    --output data/budget_epsb_geom.csv
fi
