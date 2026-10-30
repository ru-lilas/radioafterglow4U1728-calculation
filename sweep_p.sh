#!/bin/bash

if [[ "$1" == "--full" ]]; then
  python -m scripts.build_input \
    --mode p-sweep \
    --template input/general.yaml \
    --runs-dir data \
    --p-values 0.0001 0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 \
    --sweep-index data/p_sweep_full.csv \
    --yes
else
  python -m scripts.build_input \
    --mode p-index-sweep \
    --template input/general.yaml \
    --runs-dir data \
    --p-index-values 2.001 2.50 3.00\
    --sweep-index data/p_sweep.csv \
    --yes
fi
