#!/bin/bash

if [[ "$1" == "--full" ]]; then
  python -m scripts.build_input \
    --mode f-e-sweep \
    --template input/general.yaml \
    --runs-dir data \
    --f-e-values 0.0001 0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 \
    --sweep-index data/f_e_sweep_full.csv \
    --yes
else
  python -m scripts.build_input \
    --mode f-e-sweep \
    --template input/general.yaml \
    --runs-dir data \
    --f-e-values 0.0005 0.005 0.05 \
    --sweep-index data/f_e_sweep.csv \
    --yes
fi
