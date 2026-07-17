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
FILEPATH_PHYSICAL_PARAMETERS := $(DATADIR)/$(PARAMETER_DIR)/$(PHYSICAL_PARAMETERS).yaml

all: $(FIGDIR)/$(SAMPLING_DIR)/chi2_colormap.pdf \
	$(FIGDIR)/$(SAMPLING_DIR)/estimated_lightcurve.pdf

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

$(DATADIR)/$(SAMPLING_DIR)/chi2_estimated_parameters.csv: \
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

# $(DATADIR)/$(SAMPLING_DIR)/sampling.yaml: $(INPUTDIR)/sampling.yaml
# 	cp $< $@
#
# $(PATH_PHYSICAL_PARAMETERS).yaml: $(INPUTDIR)/$(PHYSICAL_PARAMETERS).yaml
# 	cp $< $@
