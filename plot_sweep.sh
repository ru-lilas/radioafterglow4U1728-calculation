#!/bin/bash

if [[ "$1" == "--full" ]]; then
  python -m scripts.plot_lightcurve \
    --mode sweep \
    --parameter f_e \
    --sweep-index data/f_e_sweep_full.csv \
    --minimum-chi-squared-path  minimum_chi_squared.csv\
    --model-parameter-table-path model_parameter_table.csv \
    --candidate-lightcurves-path candidate_lightcurves.csv \
    --obslc ./data/obs/obslc.csv \
    --plotconfig ./plotconfigs/minimum_chi_squared_lightcurves.yaml \
    --frequencies 5.5 9.0 \
    --output fig/f_e_lightcurves_full.pdf
else
  python -m scripts.plot_lightcurve \
    --mode sweep \
    --parameter f_e \
    --sweep-index data/f_e_sweep.csv \
    --minimum-chi-squared-path  minimum_chi_squared.csv\
    --model-parameter-table-path model_parameter_table.csv \
    --candidate-lightcurves-path candidate_lightcurves.csv \
    --obslc ./data/obs/obslc.csv \
    --plotconfig ./plotconfigs/minimum_chi_squared_lightcurves.yaml \
    --frequencies 5.5 9.0 \
    --output fig/f_e_lightcurves.pdf
fi
