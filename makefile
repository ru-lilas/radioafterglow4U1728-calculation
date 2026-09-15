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
ALLOWING_MODELS := allowing_models
MODEL_PARAMETER_TABLE := model_parameter_table

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
FILEPATH_INTEGRAL_TABLE := $(DATADIR)/$(INTEGRAL_TABLE).$(EXTENTION_CSV)
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
FILEPATH_CHI2 := $(DATADIR)/$(RUN_DIR)/chi2.$(EXTENTION_CSV)
FILEPATH_MINIMUM_CHI2_LC := $(DATADIR)/$(RUN_DIR)/minimum_chi2_lc.$(EXTENTION_CSV)
FILEPATH_ALLOWING_MODELS := $(DATADIR)/$(RUN_DIR)/$(ALLOWING_MODELS).$(EXTENTION_CSV)
FILEPATH_MODEL_PARAMETER_TABLE := $(DATADIR)/$(RUN_DIR)/$(MODEL_PARAMETER_TABLE).$(EXTENTION_CSV)

param: \
	$(FILEPATH_LC_EST) \
	$(FILEPATH_INPUT) \
	scripts/estimate_system_property.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--estimated $< \
		--input $(FILEPATH_INPUT)

lcfig: $(FILEPATH_LC_PDF)


$(FILEPATH_LC_PDF): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_LC_EST) \
	$(FILEPATH_MINIMUM_CHI2_LC) \
	$(FILEPATH_OBSLC) \
	plotconfigs/estimated_lightcurve.yaml \
	scripts/plot_lightcurve.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--calclc $(FILEPATH_LC_EST) \
		--calclc_raw $(FILEPATH_MINIMUM_CHI2_LC) \
		--obslc $(FILEPATH_OBSLC) \
		--plotconfig plotconfigs/estimated_lightcurve.yaml \
		--output $@

chevaliercm: $(FILEPATH_CHEVALIER_COLORMAP_PDF)
$(FILEPATH_CHEVALIER_COLORMAP_PDF): \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(FILEPATH_OBSLC) \
	plotconfigs/chevalier_colormaps.yaml \
	scripts/chevalier_colormaps.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--model_table $< \
		--plotconfig plotconfigs/chevalier_colormaps.yaml \
		--obslc $(FILEPATH_OBSLC) \
		--output $@

chevalier: $(FILEPATH_CHEVALIER_PDF)

$(FILEPATH_CHEVALIER_PDF): \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(FILEPATH_OBSLC) \
	plotconfigs/chevalier_diagram.yaml \
	scripts/chevalier.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--model_table $< \
		--plotconfig plotconfigs/chevalier_diagram.yaml \
		--obslc $(FILEPATH_OBSLC) \
		--output $@
#
lc: $(FILEPATH_LC_EST)

$(FILEPATH_LC_EST): \
	$(FILEPATH_INPUT) \
	$(DATADIR)/$(RUN_DIR)/estimation.csv \
	$(FILEPATH_INTEGRAL_TABLE) \
	$(FILEPATH_CHI2) \
	$(FILEPATH_OBSLC) \
	scripts/extract_minimum_chi2_binnedlightcurve.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--minchi2 $(FILEPATH_CHI2) \
		--allowing $(DATADIR)/$(RUN_DIR)/estimation.csv \
		--integral_table $(FILEPATH_INTEGRAL_TABLE) \
		--obslc $(FILEPATH_OBSLC) \
		--output $@

lc_raw: $(FILEPATH_MINIMUM_CHI2_LC)

$(FILEPATH_MINIMUM_CHI2_LC): \
	$(FILEPATH_INPUT) \
	$(DATADIR)/$(RUN_DIR)/estimation.csv \
	$(FILEPATH_INTEGRAL_TABLE) \
	$(FILEPATH_CHI2) \
	$(FILEPATH_OBSLC) \
	scripts/extract_minimum_chi2_lightcurve.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--minchi2 $(FILEPATH_CHI2) \
		--allowing $(DATADIR)/$(RUN_DIR)/estimation.csv \
		--integral_table $(FILEPATH_INTEGRAL_TABLE) \
		--obslc $(FILEPATH_OBSLC) \
		--output $@

chi2: $(FILEPATH_CHI2)

$(FILEPATH_CHI2): \
	$(FILEPATH_INPUT) \
	$(DATADIR)/$(RUN_DIR)/estimation.csv \
	$(FILEPATH_INTEGRAL_TABLE) \
	$(FILEPATH_OBSLC) \
	scripts/find_minimum_chi2_model.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--allowing $(DATADIR)/$(RUN_DIR)/estimation.csv \
		--integral_table $(FILEPATH_INTEGRAL_TABLE) \
		--obslc $(FILEPATH_OBSLC) \
		--output $@

est : $(DATADIR)/$(RUN_DIR)/estimation.csv

$(FILEPATH_ALLOWING_MODELS): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_PEAK_TABLE) \
	$(FILEPATH_OBSLC) \
	scripts/chevalier_est.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		--model_table $(FILEPATH_MODEL_PARAMETER_TABLE) \
		--peak_table $(FILEPATH_PEAK_TABLE) \
		--obslc $(FILEPATH_OBSLC) \
		--output $@

#=== tabulating ===#
modeltable: $(FILEPATH_MODEL_PARAMETER_TABLE)

$(FILEPATH_MODEL_PARAMETER_TABLE): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_PEAK_TABLE) \
	scripts/$(MODEL_PARAMETER_TABLE).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--peak_table $(FILEPATH_PEAK_TABLE) \
		--output $@ \

$(FILEPATH_INTEGRAL_TABLE): \
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
