INPUTDIR := input
DATADIR := data
FIGDIR := fig

BETA_DIR := beta
BETA ?= 10
BETA_FMT := $(shell printf "%03d" $(BETA))
TIME_RESO ?= 64
SPEC_RESO ?= 64
RESO := t$(TIME_RESO)f$(SPEC_RESO)

THERMAL_DIR := thermal_only
MODEL ?= $(THERMAL_DIR)
SPECTRUM_PEAK_EVOLUTION := spectrum_peak_evolution

TIME_YAML  := $(INPUTDIR)/template/$(RESO)/time.yaml
FREQ_YAML := $(INPUTDIR)/template/$(RESO)/frequency.yaml
VARYING_YAML := $(INPUTDIR)/template/$(RESO)/varying_beta.yaml
FIXED_YAML := $(INPUTDIR)/template/$(RESO)/$(BETA_DIR)/base.yaml
INTEGRAL_TABLE := integral_table
PARAMETER_TABLE := parameter_table
PEAK_TABLE := peak_table
CONTOUR_RAW := contour_raw
CHEVALIER_DIAGRAM := chevalier_diagram
ESTIMATED_LIGHTCURVE := estimated_lightcurve
OBSERVATION_LIGHTCURVE := observation_lightcurve
OBSERVATION_PEAK_DATA := observation_peak_data

#=== figures ===#
$(FIGDIR)/$(ESTIMATED_LIGHTCURVE).pdf: \
	$(DATADIR)/$(ESTIMATED_LIGHTCURVE).csv \
		$(DATADIR)/$(OBSERVATION_LIGHTCURVE).csv \
		plotconfigs/$(ESTIMATED_LIGHTCURVE).yaml \
		plot/$(ESTIMATED_LIGHTCURVE).py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--observation $(word 2,$^) \
		--config $(word 3,$^) \
		--output $@

$(FIGDIR)/test/chevalier_overlap.pdf: \
	$(DATADIR)/$(CHEVALIER_DIAGRAM).csv \
		$(DATADIR)/chevalier_diagram_obsscats.csv \
		$(DATADIR)/estimated_parameters.csv \
		plotconfigs/$(CHEVALIER_DIAGRAM).yaml \
		plot/test/chevalier_overlap.py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--scatters $(word 2,$^) \
		--estimated $(word 3,$^) \
		--config $(word 4,$^) \
		--output $@

$(FIGDIR)/$(CHEVALIER_DIAGRAM).pdf: \
	$(DATADIR)/$(CHEVALIER_DIAGRAM).csv \
		$(DATADIR)/chevalier_diagram_obsscats.csv \
		plotconfigs/$(CHEVALIER_DIAGRAM).yaml \
		plot/$(CHEVALIER_DIAGRAM).py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		-s $(word 2,$^) \
		-c $(word 3,$^) \
		-o $@

$(FIGDIR)/test/%.pdf: \
	$(DATADIR)/test/%.csv \
		plot/test/%.py \
		plotconfigs/test/%.yaml
	python -m plot.test.$* \
		$< \
 		-o $@ \
		-c $(lastword $^)

$(DATADIR)/test/%/spectrum/.done: \
	input/test/%.yaml \
		test/%.py \
		data/tabular/xi.csv
	python -m test.$* \
		$< \
		--outdir $(DATADIR)/test/$*/spectrum \
		--tabular $(lastword $^)
	mkdir -p $(dir $@)
	touch $@

#=== csv files ===#
$(DATADIR)/estimated_ejecta_property.csv: \
	$(DATADIR)/estimated_parameters.csv \
		$(INPUTDIR)/property_4u1728.yaml \
		scripts/estimate_ejecta_property.py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--burster_property $(word 2,$^) \
		--output $@

$(DATADIR)/$(ESTIMATED_LIGHTCURVE).csv: \
	$(DATADIR)/estimated_parameters.csv \
		$(INPUTDIR)/lightcurve.yaml \
		$(DATADIR)/$(INTEGRAL_TABLE).csv \
		scripts/compute_$(ESTIMATED_LIGHTCURVE).py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--config $(word 2,$^) \
		--table $(word 3,$^) \
		--output $@

$(DATADIR)/estimated_parameters.csv: \
	$(DATADIR)/$(CHEVALIER_DIAGRAM).csv \
		$(DATADIR)/$(OBSERVATION_PEAK_DATA).csv \
		scripts/extract_estimated_parameters.py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--scatters $(word 2,$^) \
		--output $@

$(DATADIR)/$(CHEVALIER_DIAGRAM).csv: \
	$(DATADIR)/$(CONTOUR_RAW).csv \
		$(DATADIR)/$(CONTOUR_RAW).csv \
		$(INPUTDIR)/$(CHEVALIER_DIAGRAM).yaml \
		scripts/extract_contour_data.py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--config $(word 2,$^) \
		--output $@

$(DATADIR)/test/test_contour_chevalier.csv: \
	$(DATADIR)/test/test_contour_chevalier_raw.csv \
		scripts/extract_contour_data.py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--output $@

$(DATADIR)/$(CONTOUR_RAW).csv: \
	$(DATADIR)/$(PARAMETER_TABLE).csv \
		$(DATADIR)/$(PEAK_TABLE).csv \
		scripts/build_$(CONTOUR_RAW).py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--table $(word 2,$^) \
		--output $@

$(DATADIR)/test/$(CONTOUR_RAW).csv: \
	$(DATADIR)/test/test_$(PARAMETER_TABLE).csv \
		$(DATADIR)/$(OBSERVATION_LIGHTCURVE).csv \
		scripts/build_$(CONTOUR_RAW).py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--table $(word 2,$^) \
		--output $@

#=== tabulating ===#
$(DATADIR)/$(PARAMETER_TABLE).csv: \
	input/$(PARAMETER_TABLE).yaml \
		$(PARAMETER_TABLE).py
	python $(lastword $^) \
		$< \
		--output $@ \

$(DATADIR)/test/test_$(PARAMETER_TABLE).csv: \
	input/test/test_$(PARAMETER_TABLE).yaml \
		scripts/$(PARAMETER_TABLE).py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--output $@ \

$(DATADIR)/$(PEAK_TABLE).csv: \
	input/$(PEAK_TABLE).yaml \
		scripts/$(PEAK_TABLE).py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--table $(DATADIR)/$(INTEGRAL_TABLE).csv \
		--output $@ \

$(DATADIR)/$(INTEGRAL_TABLE).csv: \
	input/$(INTEGRAL_TABLE).yaml \
		scripts/$(INTEGRAL_TABLE).py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--output $@

#=== arrange observation data ===#
$(DATADIR)/$(OBSERVATION_PEAK_DATA).csv: \
	$(DATADIR)/$(OBSERVATION_LIGHTCURVE).csv \
		$(OBSERVATION_PEAK_DATA).py
	python $(lastword $^) \
		$< \
		--output $@

$(DATADIR)/$(OBSERVATION_LIGHTCURVE).csv: \
	$(DATADIR)/observation_raw/4U1728_stacked_1min.dat \
		$(OBSERVATION_LIGHTCURVE).py
	python $(lastword $^) \
		$< \
		--output $@

#=== config files ===#
input/%.yaml:
	@echo "エラー: インプットファイルがありません $@"
	@false
plotconfigs/%.yaml:
	@echo "エラー: プロットコンフィグがありません $@"
	@false
