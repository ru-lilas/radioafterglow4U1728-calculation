#!/bin/bash

# for i in $(seq -f "%03g" 0 63); do
#     make fig/thermal_only/t64f64/beta/010/spectrum/${i}.pdf SPEC_RESO=64
# done

# for beta in $(seq -f "%03g" 10 50); do
#   beta_fmt=$(printf "%03d" $beta)
#
#   for i in $(seq -f "%03g" 0 63); do
#     make fig/thermal_only/t64f64/beta/$beta_fmt/spectrum/${i}.pdf SPEC_RESO=64
#   done
#   make SPEC_RESO=64 BETA=$beta fig/thermal_only/t64f64/beta/$beta_fmt/spectrum_peak_evolution.pdf
# done

# for beta in $(seq 10 30); do
#   beta_fmt=$(printf "%03d" $beta)
#   for i in $(seq -f "%03.0f" 0 63); do
    # make fig/thermal_only/t64f64/beta/$beta_fmt/spectrum/${i}.pdf SPEC_RESO=64 BETA=$beta
    # make data/thermal_only/t64f64/beta/$beta_fmt/spectrum/${i}.csv SPEC_RESO=64 BETA=$beta
    # make SPEC_RESO=64 BETA=$beta fig/thermal_only/t64f64/beta/$beta_fmt/spectrum_peak_evolution.pdf
#   done
# done

for i in $(seq 0 63); do
  i_fmt=$(printf "%03d" $i)
  make fig/test/test_peak_matching/spectrum/${i_fmt}.pdf
done
