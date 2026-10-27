python -m scripts.build_input \
  --mode f-e-sweep \
  --template input/general.yaml \
  --runs-dir data \
  --f-e-values 0.0000001 0.005 \
  --sweep-index data/f_e_sweep.csv \
  --yes

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
