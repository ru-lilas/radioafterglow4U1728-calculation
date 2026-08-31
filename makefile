INPUTDIR := input
DATADIR := data
FIGDIR := fig
OBSDIR := obs
PLOTCONFIGDIR := plotconfigs
EXTENTION_CSV := csv
EXTENTION_YAML := yaml
EXTENTION_PDF := pdf
EXTENTION_SVG := svg
EXTENTION_PY := py
SUFFIX_TMP := _tmp
INTEGRAL_TABLE := integral_table
PARAMETER_TABLE := parameter_table
ESTIMATED_LIGHTCURVE := estimated_lightcurve
CHI2FIT_PARAMETERS := chi2fit_parameters
OBSERVATION_LIGHTCURVE := obslc
CHI2_ESTIMATED_PARAMETERS := chi2_estimated_parameters

RUN_ID ?=
ifndef RUN_ID
$(error RUN_IDを指定してください. 例 make RUN_ID=1)
endif
RUN_DIR := $(shell printf 'run_%06d' '$(RUN_ID)')

OUTDATADIR := $(DATADIR)/$(RESO)/$(OBS_TIMEWINDOW_TAG)
OUTDIR := $(FIGDIR)/$(RESO)/$(OBS_TIMEWINDOW_TAG)
FIGOUTDIR := $(FIGDIR)/$(SCENARIO_TAG)/$(SAMPLING_TAG)
PATH_PHYSICAL_PARAMETERS := $(DATADIR)/$(PARAMETER_DIR)/$(PHYSICAL_PARAMETERS)
FILEPATH_INPUT := $(DATADIR)/$(RUN_DIR)/input.$(EXTENTION_YAML)
FILEPATH_INTEGRAL_TABLE := $(DATADIR)/$(INTEGRAL_TABLE).csv
FILEPATH_PARAMETER_TABLE := $(DATADIR)/$(RUN_DIR)/.$(EXTENTION_YAML)
FILEPATH_CHI2TEST := $(PARAMETER_DIR)/$(CHI2TEST_SUMMARY).$(EXTENTION_CSV)
FILEPATH_LIST_CHI2ESTIMATED_PARAMETERS ?=
FILEPATH_OBSLC := $(DATADIR)/$(OBSDIR)/$(OBSERVATION_LIGHTCURVE).$(EXTENTION_CSV)
FILEPATH_OBSLC_TMP := $(DATADIR)/$(OBSDIR)/$(OBSERVATION_LIGHTCURVE)$(SUFFIX_TMP).$(EXTENTION_CSV)
FILEPATH_OBSBG := $(DATADIR)/$(OBSDIR)/bg.$(EXTENTION_CSV)

# all: $(FIGDIR)/$(SAMPLING_DIR)/chi2_colormap.pdf \
# 	$(FIGDIR)/$(SAMPLING_DIR)/estimated_lightcurve.pdf
#
# %.$(EXTENTION_SVG): %.$(EXTENTION_PDF)
# 	pdftocairo -svg $< $@

# #=== figures ===#
# $(FIGDIR)/$(PARAMETER_DIR)/$(CHI2TEST_TIMEWINDOW).$(EXTENTION_PDF): \
# 	$(DATADIR)/$(PARAMETER_DIR)/$(CHI2_ESTIMATED_SUMMARY).$(EXTENTION_CSV) \
# 		$(FILEPATH_PHYSICAL_PARAMETERS) \
# 		$(PLOTCONFIGDIR)/$(CHI2TEST_TIMEWINDOW).$(EXTENTION_YAML) \
# 		plot/$(CHI2TEST_TIMEWINDOW).py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		--input $< \
# 		-c $(PLOTCONFIGDIR)/$(CHI2TEST_TIMEWINDOW).$(EXTENTION_YAML) \
# 		--parameters $(FILEPATH_PHYSICAL_PARAMETERS) \
# 		-o $@
#
# $(FIGDIR)/$(RESO)/$(CHEVALIER_DIAGRAM).pdf: \
# 	$(DATADIR)/$(RESO)/chevalier_contour.csv \
# 		$(DATADIR)/$(RESO)/$(CHEVALIER_SCATTERS).csv \
# 		plotconfigs/$(CHEVALIER_DIAGRAM).yaml \
# 		plot/$(CHEVALIER_DIAGRAM).py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		-s $(word 2,$^) \
# 		-c $(word 3,$^) \
# 		-o $@
#
# $(FIGDIR)/$(SAMPLING_DIR)/$(ESTIMATED_LIGHTCURVE).pdf: \
# 	$(DATADIR)/$(SAMPLING_DIR)/$(ESTIMATED_LIGHTCURVE).csv \
# 		$(FILEPATH_OBSLC) \
# 		plotconfigs/$(ESTIMATED_LIGHTCURVE).yaml \
# 		plot/$(ESTIMATED_LIGHTCURVE).py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		--observation $(word 2,$^) \
# 		--config $(word 3,$^) \
# 		--output $@
#
# $(FIGDIR)/$(CHEVALIER_DIAGRAM)_background.pdf: \
# 	$(DATADIR)/$(CHEVALIER_DIAGRAM).csv \
# 		plotconfigs/$(CHEVALIER_DIAGRAM)_background.yaml \
# 		plot/$(CHEVALIER_DIAGRAM)_background.py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		-c $(word 2,$^) \
# 		-o $@
#
# $(FIGDIR)/$(RESO)/tau_theta_colormap.pdf: \
# 	$(DATADIR)/$(RESO)/parameter_table.csv \
# 		plotconfigs/tau_theta_colormap.yaml \
# 		plot/tau_theta_colormap.py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		-c $(word 2,$^) \
# 		-o $@
#
# $(FIGDIR)/$(SAMPLING_DIR)/chi2_colormap.pdf: \
# 	$(DATADIR)/$(SAMPLING_DIR)/chi2.csv \
# 		plotconfigs/chi2_colormap.yaml \
# 		plot/chi2_colormap.py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		--chi2 $< \
# 		--config $(word 2,$^) \
# 		--output $@
#
# $(FIGDIR)/$(RESO)/parameter_dependence/%.pdf: \
# 	$(DATADIR)/$(RESO)/$(PARAMETER_TABLE).csv \
# 		plotconfigs/parameter_dependence/%.yaml \
# 		plot/parameter_dependence/%.py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		-c $(word 2,$^) \
# 		-o $@
#
# #=== csv files ===#
# $(DATADIR)/$(SAMPLING_DIR)/estimated_ejecta_property.csv: \
# 	$(DATADIR)/chi2_estimated_parameters.csv \
# 		$(INPUTDIR)/property_4u1728.yaml \
# 		scripts/estimate_ejecta_property.py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		--burster_property $(word 2,$^) \
# 		--output $@
#
# .PRECIOUS: $(DATADIR)/$(RESO)/$(CHEVALIER_SCATTERS).csv
# $(DATADIR)/$(RESO)/$(CHEVALIER_SCATTERS).csv: \
# 	$(DATADIR)/$(RESO)/$(ESTIMATED_LIGHTCURVE).csv \
# 		scripts/build_chi2fit_$(CHEVALIER_SCATTERS).py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		--output $@
#
# $(DATADIR)/$(SAMPLING_DIR)/$(ESTIMATED_LIGHTCURVE).csv: \
# 	$(DATADIR)/$(SAMPLING_DIR)/chi2_estimated_parameters.csv \
# 		$(FILEPATH_PHYSICAL_PARAMETERS) \
# 		$(INPUTDIR)/lightcurve.yaml \
# 		$(FILEPATH_INTEGRAL_TABLE) \
# 		scripts/compute_$(ESTIMATED_LIGHTCURVE).py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		--estimated_parameters $< \
# 		--output $@ \
# 		--physical_parameters $(FILEPATH_PHYSICAL_PARAMETERS) \
# 		--lightcurve_config $(word 3,$^) \
# 		--table_integral $(FILEPATH_INTEGRAL_TABLE)
#
# $(DATADIR)/$(CHEVALIER_DIAGRAM).csv: \
# 	$(DATADIR)/$(CONTOUR_RAW).csv \
# 		$(DATADIR)/$(CONTOUR_RAW).csv \
# 		$(INPUTDIR)/$(CHEVALIER_DIAGRAM).yaml \
# 		scripts/extract_contour_data.py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		--config $(word 2,$^) \
# 		--output $@
#
# $(DATADIR)/$(RESO)/chevalier_contour.csv: \
# 	$(DATADIR)/$(RESO)/$(PARAMETER_TABLE).csv \
# 		$(DATADIR)/$(PEAK_TABLE).csv \
# 		scripts/build_chevalier_contour.py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		--peak_table $(word 2,$^) \
# 		--output $@
#
# $(DATADIR)/$(CONTOUR_RAW).csv: \
# 	$(DATADIR)/$(PARAMETER_TABLE).csv \
# 		$(DATADIR)/$(PEAK_TABLE).csv \
# 		scripts/build_$(CONTOUR_RAW).py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		--table $(word 2,$^) \
# 		--output $@
#
# $(DATADIR)/$(PARAMETER_DIR)/$(CHI2_ESTIMATED_SUMMARY).$(EXTENTION_CSV): \
# 	$(CHI2_ESTIMATED_LIST) \
# 		scripts/build_chi2test_timewindow.py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		--data $(CHI2_ESTIMATED_LIST) \
# 		--output $@
#
# $(DATADIR)/$(SAMPLING_DIR)/$(CHI2_ESTIMATED_PARAMETERS).$(EXTENTION_CSV): \
# 	$(DATADIR)/$(SAMPLING_DIR)/$(CHI2_ESTIMATED_PARAMETERS)_tmp.$(EXTENTION_CSV)\
# 		$(DATADIR)/$(SAMPLING_DIR)/$(SAMPLING).$(EXTENTION_YAML)\
# 		scripts/combine_chi2est_timewindow.py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		--chi2_estimated $< \
# 		--sampling_config $(DATADIR)/$(SAMPLING_DIR)/$(SAMPLING).$(EXTENTION_YAML) \
# 		--output $@
#
# $(DATADIR)/$(SAMPLING_DIR)/$(CHI2_ESTIMATED_PARAMETERS)_tmp.$(EXTENTION_CSV): \
# 	$(DATADIR)/$(SAMPLING_DIR)/chi2.csv \
# 		scripts/chi2_estimated_parameters.py
# 	python3 -m $(subst /,.,$(basename $(lastword $^))) \
# 		$< \
# 		--output $@
#
lc: $(FIGDIR)/$(RUN_DIR)/lc.$(EXTENTION_PDF)
$(FIGDIR)/$(RUN_DIR)/lc.$(EXTENTION_PDF): \
	$(DATADIR)/$(RUN_DIR)/lc_est.$(EXTENTION_CSV) \
	$(FILEPATH_OBSLC) \
	$(FILEPATH_INPUT) \
	plotconfigs/estimated_lightcurve.yaml \
	plot/plot_calculation_lightcurve.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--lc $< \
		--obslc $(FILEPATH_OBSLC) \
		--config $(FILEPATH_INPUT) \
		--plotconfig plotconfigs/estimated_lightcurve.yaml \
		--output $@
#
$(DATADIR)/$(RUN_DIR)/lc_est.$(EXTENTION_CSV): \
	$(DATADIR)/$(RUN_DIR)/chi2.csv \
	$(DATADIR)/$(RUN_DIR)/$(PARAMETER_TABLE).$(EXTENTION_CSV) \
	$(DATADIR)/$(INTEGRAL_TABLE).$(EXTENTION_CSV) \
	$(FILEPATH_OBSLC) \
	$(FILEPATH_INPUT) \
	scripts/compute_estimated_lightcurve.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--estimated $< \
		--parameter_table $(DATADIR)/$(RUN_DIR)/$(PARAMETER_TABLE).$(EXTENTION_CSV) \
		--integral_table $(DATADIR)/$(INTEGRAL_TABLE).$(EXTENTION_CSV) \
		--obs_lc $(FILEPATH_OBSLC) \
		--config $(FILEPATH_INPUT) \
		--output $@

$(DATADIR)/$(RUN_DIR)/chi2.csv: \
	$(DATADIR)/$(RUN_DIR)/$(PARAMETER_TABLE).$(EXTENTION_CSV) \
	$(DATADIR)/$(INTEGRAL_TABLE).$(EXTENTION_CSV) \
	$(FILEPATH_OBSLC) \
	$(FILEPATH_INPUT) \
	scripts/compute_chi2.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--parameter_table $< \
		--integral_table $(DATADIR)/$(INTEGRAL_TABLE).$(EXTENTION_CSV) \
		--obs_lc $(FILEPATH_OBSLC) \
		--config $(FILEPATH_INPUT) \
		--output $@

#=== tabulating ===#
$(DATADIR)/$(RUN_DIR)/$(PARAMETER_TABLE).$(EXTENTION_CSV): \
	$(FILEPATH_INPUT) \
	scripts/$(PARAMETER_TABLE).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--output $@ \

$(DATADIR)/$(INTEGRAL_TABLE).$(EXTENTION_CSV): \
	$(INPUTDIR)/$(INTEGRAL_TABLE).$(EXTENTION_YAML) \
		scripts/$(INTEGRAL_TABLE).$(EXTENTION_PY)
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

$(FILEPATH_OBSLC): \
	$(FILEPATH_OBSLC_TMP) \
	$(FILEPATH_OBSBG) \
	scripts/combine_obslc_background.$(EXTENTION_PY)
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--lc_tmp $< \
		--bg $(FILEPATH_OBSBG) \
		--output $@

$(FILEPATH_OBSBG): \
	$(FILEPATH_OBSLC_TMP) \
		scripts/estimate_background.$(EXTENTION_PY)
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--data $< \
		--output $@

$(FILEPATH_OBSLC_TMP): \
	$(DATADIR)/observation_raw/4U1728_stacked_1min.dat \
		$(INPUTDIR)/$(OBSERVATION_LIGHTCURVE).$(EXTENTION_YAML) \
		preprocess_$(OBSERVATION_LIGHTCURVE).$(EXTENTION_PY)
	python3 $(lastword $^) \
		--data $< \
		--config $(INPUTDIR)/$(OBSERVATION_LIGHTCURVE).$(EXTENTION_YAML) \
		--output $@

#=== config files ===#
plotconfigs/%.yaml:
	@echo "エラー: プロットコンフィグがありません $@"
	@false
