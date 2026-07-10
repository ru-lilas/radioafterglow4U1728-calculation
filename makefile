INPUTDIR := input
DATADIR := data
FIGDIR := fig
INTEGRAL_TABLE := integral_table
PARAMETER_TABLE := parameter_table
PEAK_TABLE := peak_table
CONTOUR_RAW := contour_raw
CHEVALIER_DIAGRAM := chevalier_diagram
ESTIMATED_LIGHTCURVE := estimated_lightcurve
CHI2FIT_PARAMETERS := chi2fit_parameters
CHEVALIER_SCATTERS := chevalier_scatters
OBSERVATION_LIGHTCURVE := observation_lightcurve
OBSERVATION_PEAK_DATA := observation_peak_data

RESO := a0256b0256

#=== figures ===#
$(FIGDIR)/$(RESO)/$(CHEVALIER_DIAGRAM).pdf: \
	$(DATADIR)/$(RESO)/chevalier_contour.csv \
		$(DATADIR)/$(RESO)/$(CHEVALIER_SCATTERS).csv \
		plotconfigs/$(CHEVALIER_DIAGRAM).yaml \
		plot/$(CHEVALIER_DIAGRAM).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		-s $(word 2,$^) \
		-c $(word 3,$^) \
		-o $@

$(FIGDIR)/$(RESO)/$(ESTIMATED_LIGHTCURVE).pdf: \
	$(DATADIR)/$(RESO)/$(ESTIMATED_LIGHTCURVE).csv \
		$(DATADIR)/$(OBSERVATION_LIGHTCURVE).csv \
		plotconfigs/$(ESTIMATED_LIGHTCURVE).yaml \
		plot/$(ESTIMATED_LIGHTCURVE).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--observation $(word 2,$^) \
		--config $(word 3,$^) \
		--output $@

$(FIGDIR)/$(CHEVALIER_DIAGRAM)_background.pdf: \
	$(DATADIR)/$(CHEVALIER_DIAGRAM).csv \
		plotconfigs/$(CHEVALIER_DIAGRAM)_background.yaml \
		plot/$(CHEVALIER_DIAGRAM)_background.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		-c $(word 2,$^) \
		-o $@

$(FIGDIR)/$(RESO)/tau_theta_colormap.pdf: \
	$(DATADIR)/$(RESO)/parameter_table.csv \
		plotconfigs/tau_theta_colormap.yaml \
		plot/tau_theta_colormap.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		-c $(word 2,$^) \
		-o $@

$(FIGDIR)/$(RESO)/chi2_colormap.pdf: \
	$(DATADIR)/$(RESO)/chi2_colormap.csv \
		plotconfigs/chi2_colormap.yaml \
		plot/chi2_colormap.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		-c $(word 2,$^) \
		-o $@

$(FIGDIR)/$(RESO)/parameter_dependence/%.pdf: \
	$(DATADIR)/$(RESO)/$(PARAMETER_TABLE).csv \
		plotconfigs/parameter_dependence/%.yaml \
		plot/parameter_dependence/%.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		-c $(word 2,$^) \
		-o $@

#=== csv files ===#
$(DATADIR)/estimated_ejecta_property.csv: \
	$(DATADIR)/estimated_parameters.csv \
		$(INPUTDIR)/property_4u1728.yaml \
		scripts/estimate_ejecta_property.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--burster_property $(word 2,$^) \
		--output $@

.PRECIOUS: $(DATADIR)/$(RESO)/$(CHEVALIER_SCATTERS).csv
$(DATADIR)/$(RESO)/$(CHEVALIER_SCATTERS).csv: \
	$(DATADIR)/$(RESO)/$(ESTIMATED_LIGHTCURVE).csv \
		scripts/build_chi2fit_$(CHEVALIER_SCATTERS).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--output $@

$(DATADIR)/$(RESO)/$(ESTIMATED_LIGHTCURVE).csv: \
	$(DATADIR)/$(RESO)/chi2_estimated_parameters.csv \
		$(INPUTDIR)/lightcurve.yaml \
		$(DATADIR)/$(INTEGRAL_TABLE).csv \
		scripts/compute_$(ESTIMATED_LIGHTCURVE).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--config $(word 2,$^) \
		--table_integral $(word 3,$^) \
		--output $@

$(DATADIR)/estimated_parameters.csv: \
	$(DATADIR)/$(CHEVALIER_DIAGRAM).csv \
		$(DATADIR)/$(OBSERVATION_PEAK_DATA).csv \
		scripts/extract_estimated_parameters.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--scatters $(word 2,$^) \
		--output $@

$(DATADIR)/$(CHEVALIER_DIAGRAM).csv: \
	$(DATADIR)/$(CONTOUR_RAW).csv \
		$(DATADIR)/$(CONTOUR_RAW).csv \
		$(INPUTDIR)/$(CHEVALIER_DIAGRAM).yaml \
		scripts/extract_contour_data.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--config $(word 2,$^) \
		--output $@

$(DATADIR)/$(RESO)/chevalier_contour.csv: \
	$(DATADIR)/$(RESO)/$(PARAMETER_TABLE).csv \
		$(DATADIR)/$(PEAK_TABLE).csv \
		scripts/build_chevalier_contour.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--peak_table $(word 2,$^) \
		--output $@

$(DATADIR)/$(CONTOUR_RAW).csv: \
	$(DATADIR)/$(PARAMETER_TABLE).csv \
		$(DATADIR)/$(PEAK_TABLE).csv \
		scripts/build_$(CONTOUR_RAW).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--table $(word 2,$^) \
		--output $@

.PRECIOUS: $(DATADIR)/$(RESO)/chi2fit_parameters.csv
$(DATADIR)/$(RESO)/chi2fit_parameters.csv: \
	$(DATADIR)/$(RESO)/$(PARAMETER_TABLE).csv \
		$(DATADIR)/$(INTEGRAL_TABLE).csv \
		$(DATADIR)/$(OBSERVATION_LIGHTCURVE).csv \
		$(INPUTDIR)/estimate_chi2fit_parameters.yaml \
		scripts/estimate_chi2fit_parameters.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--integral_table $(word 2,$^) \
		--observation_lc $(word 3,$^) \
		--config $(word 4,$^) \
		--output $@

$(DATADIR)/$(RESO)/chi2_estimated_parameters.csv: \
	$(DATADIR)/$(RESO)/chi2.csv \
		$(DATADIR)/$(RESO)/$(PARAMETER_TABLE).csv \
		scripts/chi2_estimated_parameters.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--parameter_table $(word 2,$^) \
		--output $@

$(DATADIR)/$(RESO)/chi2_colormap.csv: \
	$(DATADIR)/$(RESO)/parameter_table.csv \
		$(DATADIR)/$(RESO)/chi2.csv \
		scripts/chi2_colormap.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--chi2_table $(word 2,$^) \
		-o $@

$(DATADIR)/$(RESO)/chi2.csv: \
	$(DATADIR)/$(RESO)/$(PARAMETER_TABLE).csv \
		$(DATADIR)/$(INTEGRAL_TABLE).csv \
		$(DATADIR)/$(OBSERVATION_LIGHTCURVE).csv \
		$(INPUTDIR)/estimate_chi2fit_parameters.yaml \
		scripts/compute_chi2.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--integral_table $(word 2,$^) \
		--observation_lc $(word 3,$^) \
		--config $(word 4,$^) \
		--output $@

#=== tabulating ===#
$(DATADIR)/$(RESO)/$(PARAMETER_TABLE).csv: \
	input/$(RESO)/$(PARAMETER_TABLE).yaml \
		$(PARAMETER_TABLE).py
	python3 $(lastword $^) \
		$< \
		--output $@ \

$(DATADIR)/$(PEAK_TABLE).csv: \
	input/$(PEAK_TABLE).yaml \
		scripts/$(PEAK_TABLE).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--table $(DATADIR)/$(INTEGRAL_TABLE).csv \
		--output $@ \

$(DATADIR)/$(INTEGRAL_TABLE).csv: \
	input/$(INTEGRAL_TABLE).yaml \
		scripts/$(INTEGRAL_TABLE).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--output $@

#=== arrange observation data ===#
$(DATADIR)/$(OBSERVATION_PEAK_DATA).csv: \
	$(DATADIR)/$(OBSERVATION_LIGHTCURVE).csv \
		$(OBSERVATION_PEAK_DATA).py
	python3 $(lastword $^) \
		$< \
		--output $@

$(DATADIR)/$(OBSERVATION_LIGHTCURVE).csv: \
	$(DATADIR)/observation_raw/4U1728_stacked_1min.dat \
		$(OBSERVATION_LIGHTCURVE).py
	python3 $(lastword $^) \
		$< \
		--output $@

#=== config files ===#
input/%.yaml:
	@echo "エラー: インプットファイルがありません $@"
	@false
plotconfigs/%.yaml:
	@echo "エラー: プロットコンフィグがありません $@"
	@false
