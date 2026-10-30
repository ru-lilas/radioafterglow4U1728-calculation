python -m scripts.plot_lightcurve_sweep \
  --f-e-sweep-index data/f_e_sweep.csv \
  --p-index-sweep-index data/p_sweep.csv \
  --minimum-chi-squared-path  minimum_chi_squared.csv\
  --model-parameter-table-path model_parameter_table.csv \
  --candidate-lightcurves-path candidate_lightcurves.csv \
  --obslc ./data/obs/obslc.csv \
  --plotconfig plotconfigs/minimum_chi_squared_lightcurves.yaml \
  --output fig/lightcurves.pdf
