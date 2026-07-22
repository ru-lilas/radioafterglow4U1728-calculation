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
PEAK_TABLE := peak_table
CONTOUR_RAW := contour_raw
CHEVALIER_DIAGRAM := chevalier_diagram
ESTIMATED_LIGHTCURVE := estimated_lightcurve
CHI2FIT_PARAMETERS := chi2fit_parameters
CHEVALIER_SCATTERS := chevalier_scatters
OBSERVATION_LIGHTCURVE := obslc
OBSERVATION_PEAK_DATA := observation_peak_data
CHI2_ESTIMATED_PARAMETERS := chi2_estimated_parameters
CHI2_ESTIMATED_SUMMARY := chi2_estimated_summary
CHI2TEST_SUMMARY ?=
CHI2TEST_TIMEWINDOW := chi2test_timewindow
SAMPLING := sampling

CONFIG_CHI2_SAMPLING ?=

TAG_RESO ?=
TAG_MICROPHYS ?=
TAG_SAMPLING ?=

PARAMETER_DIR := $(TAG_RESO)/$(TAG_MICROPHYS)
SAMPLING_DIR := $(PARAMETER_DIR)/$(TAG_SAMPLING)

PHYSICAL_PARAMETERS := physical_parameters

RESO := a0256b0256

OUTDATADIR := $(DATADIR)/$(RESO)/$(OBS_TIMEWINDOW_TAG)
OUTDIR := $(FIGDIR)/$(RESO)/$(OBS_TIMEWINDOW_TAG)
FIGOUTDIR := $(FIGDIR)/$(SCENARIO_TAG)/$(SAMPLING_TAG)
PATH_PHYSICAL_PARAMETERS := $(DATADIR)/$(PARAMETER_DIR)/$(PHYSICAL_PARAMETERS)
FILEPATH_INTEGRAL_TABLE := $(DATADIR)/$(INTEGRAL_TABLE).csv
FILEPATH_PHYSICAL_PARAMETERS := $(DATADIR)/$(PARAMETER_DIR)/$(PHYSICAL_PARAMETERS).$(EXTENTION_YAML)
FILEPATH_CHI2TEST := $(PARAMETER_DIR)/$(CHI2TEST_SUMMARY).$(EXTENTION_CSV)
FILEPATH_LIST_CHI2ESTIMATED_PARAMETERS ?=
FILEPATH_OBSLC := $(DATADIR)/$(OBSDIR)/$(OBSERVATION_LIGHTCURVE).$(EXTENTION_CSV)
FILEPATH_OBSLC_TMP := $(DATADIR)/$(OBSDIR)/$(OBSERVATION_LIGHTCURVE)$(SUFFIX_TMP).$(EXTENTION_CSV)
FILEPATH_OBSBG := $(DATADIR)/$(OBSDIR)/bg.$(EXTENTION_CSV)

all: $(FIGDIR)/$(SAMPLING_DIR)/chi2_colormap.pdf \
	$(FIGDIR)/$(SAMPLING_DIR)/estimated_lightcurve.pdf

%.$(EXTENTION_SVG): %.$(EXTENTION_PDF)
	pdftocairo -svg $< $@

#=== figures ===#
$(FIGDIR)/$(PARAMETER_DIR)/$(CHI2TEST_TIMEWINDOW).$(EXTENTION_PDF): \
	$(DATADIR)/$(PARAMETER_DIR)/$(CHI2_ESTIMATED_SUMMARY).$(EXTENTION_CSV) \
		$(FILEPATH_PHYSICAL_PARAMETERS) \
		$(PLOTCONFIGDIR)/$(CHI2TEST_TIMEWINDOW).$(EXTENTION_YAML) \
		plot/$(CHI2TEST_TIMEWINDOW).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--input $< \
		-c $(PLOTCONFIGDIR)/$(CHI2TEST_TIMEWINDOW).$(EXTENTION_YAML) \
		--parameters $(FILEPATH_PHYSICAL_PARAMETERS) \
		-o $@

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

$(FIGDIR)/$(SAMPLING_DIR)/$(ESTIMATED_LIGHTCURVE).pdf: \
	$(DATADIR)/$(SAMPLING_DIR)/$(ESTIMATED_LIGHTCURVE).csv \
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

$(FIGDIR)/$(SAMPLING_DIR)/chi2_colormap.pdf: \
	$(DATADIR)/$(SAMPLING_DIR)/chi2.csv \
		plotconfigs/chi2_colormap.yaml \
		plot/chi2_colormap.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--chi2 $< \
		--config $(word 2,$^) \
		--output $@

$(FIGDIR)/$(RESO)/parameter_dependence/%.pdf: \
	$(DATADIR)/$(RESO)/$(PARAMETER_TABLE).csv \
		plotconfigs/parameter_dependence/%.yaml \
		plot/parameter_dependence/%.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		-c $(word 2,$^) \
		-o $@

#=== csv files ===#
$(DATADIR)/$(SAMPLING_DIR)/estimated_ejecta_property.csv: \
	$(DATADIR)/chi2_estimated_parameters.csv \
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

$(DATADIR)/$(SAMPLING_DIR)/$(ESTIMATED_LIGHTCURVE).csv: \
	$(DATADIR)/$(SAMPLING_DIR)/chi2_estimated_parameters.csv \
		$(FILEPATH_PHYSICAL_PARAMETERS) \
		$(INPUTDIR)/lightcurve.yaml \
		$(FILEPATH_INTEGRAL_TABLE) \
		scripts/compute_$(ESTIMATED_LIGHTCURVE).py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--estimated_parameters $< \
		--output $@ \
		--physical_parameters $(FILEPATH_PHYSICAL_PARAMETERS) \
		--lightcurve_config $(word 3,$^) \
		--table_integral $(FILEPATH_INTEGRAL_TABLE)

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

$(DATADIR)/$(PARAMETER_DIR)/$(CHI2_ESTIMATED_SUMMARY).$(EXTENTION_CSV): \
	$(CHI2_ESTIMATED_LIST) \
		scripts/build_chi2test_timewindow.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--data $(CHI2_ESTIMATED_LIST) \
		--output $@

$(DATADIR)/$(SAMPLING_DIR)/$(CHI2_ESTIMATED_PARAMETERS).$(EXTENTION_CSV): \
	$(DATADIR)/$(SAMPLING_DIR)/$(CHI2_ESTIMATED_PARAMETERS)_tmp.$(EXTENTION_CSV)\
		$(DATADIR)/$(SAMPLING_DIR)/$(SAMPLING).$(EXTENTION_YAML)\
		scripts/combine_chi2est_timewindow.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		--chi2_estimated $< \
		--sampling_config $(DATADIR)/$(SAMPLING_DIR)/$(SAMPLING).$(EXTENTION_YAML) \
		--output $@

$(DATADIR)/$(SAMPLING_DIR)/$(CHI2_ESTIMATED_PARAMETERS)_tmp.$(EXTENTION_CSV): \
	$(DATADIR)/$(SAMPLING_DIR)/chi2.csv \
		scripts/chi2_estimated_parameters.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--output $@

$(DATADIR)/$(SAMPLING_DIR)/chi2.csv: \
	$(PATH_PHYSICAL_PARAMETERS).csv \
		$(DATADIR)/$(SAMPLING_DIR)/chi2_tmp.csv \
		scripts/chi2_colormap.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--chi2_table $(word 2,$^) \
		-o $@

$(DATADIR)/$(SAMPLING_DIR)/chi2_tmp.csv: \
	$(DATADIR)/$(PARAMETER_DIR)/$(PHYSICAL_PARAMETERS).csv \
		$(DATADIR)/$(INTEGRAL_TABLE).csv \
		$(DATADIR)/$(OBSERVATION_LIGHTCURVE).csv \
		$(PATH_PHYSICAL_PARAMETERS).yaml \
		$(DATADIR)/$(SAMPLING_DIR)/sampling.yaml \
		scripts/compute_chi2.py
	python3 -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		$(word 4,$^) \
		--integral_table $(word 2,$^) \
		--observation_lc $(word 3,$^) \
		--config $(word 5,$^) \
		--output $@

#=== tabulating ===#
$(PATH_PHYSICAL_PARAMETERS).csv: \
	$(PATH_PHYSICAL_PARAMETERS).yaml \
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
input/%.yaml:
	@echo "エラー: インプットファイルがありません $@"
	@false
plotconfigs/%.yaml:
	@echo "エラー: プロットコンフィグがありません $@"
	@false

# $(DATADIR)/$(SAMPLING_DIR)/sampling.yaml: $(INPUTDIR)/sampling.yaml
# 	cp $< $@
#
# $(PATH_PHYSICAL_PARAMETERS).yaml: $(INPUTDIR)/$(PHYSICAL_PARAMETERS).yaml
# 	cp $< $@
