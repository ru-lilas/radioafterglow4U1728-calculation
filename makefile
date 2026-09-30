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
CANDIDATE_MODELS := candidate_models
CANDIDATE_LIGHTCURVES := candidate_lightcurves
CANDIDATE_BINNED_LIGHTCURVES := candidate_binned_lightcurves
MODEL_LIGHTCURVES := model_lightcurves
MODEL_PARAMETER_TABLE := model_parameter_table
SYNCHROTRON_FUNCTION_NUMERICAL_TABLE := synchrotron_function_numerical_table
THERMAL_PEAK_TABLE := thermal_peak_table
PEAK_GRID := peak_grid
MINIMUM_CHI2_LIGHTCURVE := minimum_chi2_lightcurve

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
FILEPATH_PEAK_TABLE := $(DATADIR)/peak_table.$(EXTENTION_CSV)
FILEPATH_PARAMETER_TABLE := $(DATADIR)/$(RUN_DIR)/.$(EXTENTION_YAML)
FILEPATH_CHI2TEST := $(PARAMETER_DIR)/$(CHI2TEST_SUMMARY).$(EXTENTION_CSV)
FILEPATH_LIST_CHI2ESTIMATED_PARAMETERS ?=
FILEPATH_OBSLC := $(DATADIR)/$(OBSDIR)/obslc.$(EXTENTION_CSV)
FILEPATH_OBSLC_TMP := $(DATADIR)/$(OBSDIR)/$(OBSERVATION_LIGHTCURVE)$(SUFFIX_TMP).$(EXTENTION_CSV)
FILEPATH_OBSBG := $(DATADIR)/$(OBSDIR)/bg.$(EXTENTION_CSV)
FILEPATH_LC_EST := $(DATADIR)/$(RUN_DIR)/lc_est.$(EXTENTION_CSV)
FILEPATH_LC_PDF := $(FIGDIR)/$(RUN_DIR)/lc.$(EXTENTION_PDF)
FILEPATH_CHEVALIER_PDF := $(FIGDIR)/$(RUN_DIR)/chevalier_diagram.$(EXTENTION_PDF)
FILEPATH_CHEVALIER_COLORMAP_PDF := $(FIGDIR)/$(RUN_DIR)/chevalier_colormap.$(EXTENTION_PDF)
FILEPATH_PEAK_GRID := $(DATADIR)/$(RUN_DIR)/$(PEAK_GRID).$(EXTENTION_CSV)
FILEPATH_CHI2 := $(DATADIR)/$(RUN_DIR)/chi2.$(EXTENTION_CSV)
FILEPATH_MINIMUM_CHI2_LC := $(DATADIR)/$(RUN_DIR)/$(MINIMUM_CHI2_LIGHTCURVE).$(EXTENTION_CSV)
FILEPATH_MINIMUM_CHI2_LC_PDF := $(DATADIR)/$(RUN_DIR)/$(MINIMUM_CHI2_LIGHTCURVE).$(EXTENTION_PDF)

FILEPATH_MODEL_LIGHTCURVES := $(DATADIR)/$(RUN_DIR)/$(MODEL_LIGHTCURVES).$(EXTENTION_CSV)
FILEPATH_CANDIDATE_MODELS := $(DATADIR)/$(RUN_DIR)/$(CANDIDATE_MODELS).$(EXTENTION_CSV)
FILEPATH_CANDIDATE_LIGHTCURVES := $(DATADIR)/$(RUN_DIR)/$(CANDIDATE_LIGHTCURVES).$(EXTENTION_CSV)
FILEPATH_CANDIDATE_BINNED_LIGHTCURVES := $(DATADIR)/$(RUN_DIR)/$(CANDIDATE_BINNED_LIGHTCURVES).$(EXTENTION_CSV)
FILEPATH_MODEL_PARAMETER_TABLE := $(DATADIR)/$(RUN_DIR)/$(MODEL_PARAMETER_TABLE).$(EXTENTION_CSV)

FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE := $(DATADIR)/$(SYNCHROTRON_FUNCTION_NUMERICAL_TABLE).$(EXTENTION_CSV)
FILEPATH_THERMAL_PEAK_TABLE := $(DATADIR)/$(THERMAL_PEAK_TABLE).$(EXTENTION_CSV)

FILEPATH_CANDIDATE_LIGHTCURVES_PDF := $(FIGDIR)/$(RUN_DIR)/$(CANDIDATE_LIGHTCURVES).$(EXTENTION_PDF)

param: \
	$(FILEPATH_LC_EST) \
	$(FILEPATH_INPUT) \
	scripts/estimate_system_property.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--estimated $< \
		--input $(FILEPATH_INPUT)

minchi2lcfig: $(FILEPATH_MINIMUM_CHI2_LC_PDF)
	evince $< >/dev/null 2>&1 &
$(FILEPATH_MINIMUM_CHI2_LC_PDF): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_MINIMUM_CHI2_LC) \
	$(FILEPATH_CANDIDATE_LIGHTCURVES) \
	$(FILEPATH_CANDIDATE_MODELS) \
	$(FILEPATH_OBSLC) \
	plotconfigs/estimated_lightcurve.yaml \
	scripts/plot_$(MINIMUM_CHI2_LIGHTCURVE).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--$(MINIMUM_CHI2_LIGHTCURVE) $(FILEPATH_MINIMUM_CHI2_LC) \
		--$(CANDIDATE_MODELS) $(FILEPATH_CANDIDATE_MODELS) \
		--$(CANDIDATE_LIGHTCURVES) $(FILEPATH_CANDIDATE_LIGHTCURVES) \
		--obslc $(FILEPATH_OBSLC) \
		--plotconfig plotconfigs/estimated_lightcurve.yaml \
		--output $@

lcfig: $(FILEPATH_CANDIDATE_LIGHTCURVES_PDF)
	evince $< >/dev/null 2>&1 &

$(FILEPATH_CANDIDATE_LIGHTCURVES_PDF): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES) \
	$(FILEPATH_CANDIDATE_LIGHTCURVES) \
	$(FILEPATH_CANDIDATE_MODELS) \
	$(FILEPATH_OBSLC) \
	plotconfigs/estimated_lightcurve.yaml \
	scripts/plot_lightcurve.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--$(CANDIDATE_BINNED_LIGHTCURVES) $(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES) \
		--$(CANDIDATE_MODELS) $(FILEPATH_CANDIDATE_MODELS) \
		--$(CANDIDATE_LIGHTCURVES) $(FILEPATH_CANDIDATE_LIGHTCURVES) \
		--obslc $(FILEPATH_OBSLC) \
		--plotconfig plotconfigs/estimated_lightcurve.yaml \
		--output $@

view-chevalier_cmp: $(FILEPATH_CHEVALIER_PDF)
	evince $< >/dev/null 2>&1 &

chevaliercm: $(FILEPATH_CHEVALIER_COLORMAP_PDF)
$(FILEPATH_CHEVALIER_COLORMAP_PDF): \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(FILEPATH_OBSLC) \
	plotconfigs/chevalier_colormaps.yaml \
	module/chevalier.py \
	scripts/chevalier_colormaps.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--model_table $< \
		--plotconfig plotconfigs/chevalier_colormaps.yaml \
		--obslc $(FILEPATH_OBSLC) \
		--output $@

view-chevalier: $(FILEPATH_CHEVALIER_PDF)
	evince $< >/dev/null 2>&1 &

chevalier: $(FILEPATH_CHEVALIER_PDF)

$(FILEPATH_CHEVALIER_PDF): \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(FILEPATH_OBSLC) \
	module/chevalier.py \
	plotconfigs/chevalier_diagram.yaml \
	scripts/chevalier.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--model_table $< \
		--plotconfig plotconfigs/chevalier_diagram.yaml \
		--obslc $(FILEPATH_OBSLC) \
		--output $@

#=== datafiles ===#
min_chi2: $(FILEPATH_MINIMUM_CHI2_LC)
$(FILEPATH_MINIMUM_CHI2_LC): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES) \
	scripts/find_minimum_chi2_model.$(EXTENTION_PY)
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--$(CANDIDATE_BINNED_LIGHTCURVES) $(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES) \
		--output $@

lc_bin: $(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES)
$(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_CANDIDATE_LIGHTCURVES) \
	$(FILEPATH_OBSLC) \
	scripts/compute_$(CANDIDATE_BINNED_LIGHTCURVES).$(EXTENTION_PY)
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--$(CANDIDATE_LIGHTCURVES) $(FILEPATH_CANDIDATE_LIGHTCURVES) \
		--obslc $(FILEPATH_OBSLC) \
		--output $@

lc: $(FILEPATH_CANDIDATE_LIGHTCURVES)
$(FILEPATH_CANDIDATE_LIGHTCURVES): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_INTEGRAL_TABLE) \
	$(FILEPATH_CANDIDATE_MODELS) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	scripts/compute_lightcurve_$(CANDIDATE_MODELS).$(EXTENTION_PY)
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--integral_table $(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
		--$(CANDIDATE_MODELS) $(FILEPATH_CANDIDATE_MODELS) \
		--$(MODEL_PARAMETER_TABLE) $(FILEPATH_MODEL_PARAMETER_TABLE) \
		--output $@

candidates: $(FILEPATH_CANDIDATE_MODELS)
$(FILEPATH_CANDIDATE_MODELS): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_PEAK_GRID) \
	$(FILEPATH_OBSLC) \
	scripts/select_$(CANDIDATE_MODELS).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--$(PEAK_GRID) $(FILEPATH_PEAK_GRID)\
		--obslc $(FILEPATH_OBSLC) \
		--output $@

#=== tabulating ===#
peak_grid: $(FILEPATH_PEAK_GRID)
$(FILEPATH_PEAK_GRID): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_INTEGRAL_TABLE) \
	scripts/compute_$(PEAK_GRID).$(EXTENTION_PY)
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--integral_table $(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
		--$(MODEL_PARAMETER_TABLE) $(FILEPATH_MODEL_PARAMETER_TABLE) \
		--output $@

lc_all: $(FILEPATH_MODEL_LIGHTCURVES)
$(FILEPATH_MODEL_LIGHTCURVES): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_INTEGRAL_TABLE) \
	scripts/compute_lightcurve_$(CANDIDATE_MODELS).$(EXTENTION_PY)
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--integral_table $(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
		--output $@

modeltable: $(FILEPATH_MODEL_PARAMETER_TABLE)
$(FILEPATH_MODEL_PARAMETER_TABLE): \
	$(FILEPATH_INPUT) \
	scripts/build_$(MODEL_PARAMETER_TABLE).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--output $@ \

peak_table: $(FILEPATH_THERMAL_PEAK_TABLE)
$(FILEPATH_THERMAL_PEAK_TABLE): \
	$(INPUTDIR)/$(THERMAL_PEAK_TABLE).$(EXTENTION_YAML) \
	$(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
	scripts/build_$(THERMAL_PEAK_TABLE).$(EXTENTION_PY)
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--$(SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) $(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
		--output $@

$(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE): \
	$(INPUTDIR)/$(SYNCHROTRON_FUNCTION_NUMERICAL_TABLE).$(EXTENTION_YAML) \
	scripts/build_$(SYNCHROTRON_FUNCTION_NUMERICAL_TABLE).$(EXTENTION_PY)
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
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
