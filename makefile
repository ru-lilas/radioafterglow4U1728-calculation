NUMBERS := $(shell seq -f "%03g" 10 50)
TIMES := $(shell seq -f "%03g" 0 100)

CSVS := $(addsuffix .csv,$(NUMBERS))

INPUTDIR := input
DATADIR := data
FIGDIR := fig

BETA_DIR := beta
TIME_RESO ?= 64
SPEC_RESO ?= 256
RESO := t$(TIME_RESO)f$(SPEC_RESO)

THERMAL_DIR := thermal_only
MODEL ?= $(THERMAL_DIR)

TIME_YAML  := $(INPUTDIR)/template/$(RESO)/time.yaml
FREQ_YAML := $(INPUTDIR)/template/$(RESO)/frequency.yaml
VARYING_YAML := $(INPUTDIR)/template/$(RESO)/varying_beta.yaml
FIXED_YAML := $(INPUTDIR)/template/$(RESO)/$(BETA_DIR)/base.yaml

$(FIGDIR)/$(MODEL)/$(RESO)/beta/020/spectrum/%.pdf: \
	$(DATADIR)/$(MODEL)/$(RESO)/beta/020/spectrum/%.csv \
	pipelines/plot/spectrum.py \
	plotconfigs/spectrum.yaml
	python -m pipelines.plot.spectrum \
		$< \
		-o $@ \
		-c $(lastword $^)

.SECONDEXPANSION:
$(FIGDIR)/$(MODEL)/$(RESO)/beta/%/lightcurve.pdf: \
	$$(wildcard $(DATADIR)/$(MODEL)/$(RESO)/beta/$$*/spectrum/[0-9][0-9][0-9].csv) \
	$(DATADIR)/$(MODEL)/$(RESO)/beta/$$*/spectrum/.done \
	pipelines/plot/lightcurve.py \
	plotconfigs/lightcurve.yaml
	python -m pipelines.plot.lightcurve \
		$(filter %.csv,$^) \
		-o $@ \
		-c $(lastword $^)

.SECONDARY:

$(DATADIR)/$(MODEL)/$(RESO)/beta/%/spectrum/.done: \
	input/generated/$(MODEL)/$(RESO)/beta/%.yaml \
		pipelines/compute.py
	python -m pipelines.compute \
		$< \
		--outdir $(DATADIR)/$(MODEL)/$(RESO)/beta/$*/spectrum
	touch $@

input/generated/$(MODEL)/$(RESO)/beta/%.yaml: \
	$(VARYING_YAML) \
	$(FIXED_YAML) \
	$(TIME_YAML) \
	$(FREQ_YAML) \
	pipelines/build_input_parameters.py

	python -m pipelines.build_input_parameters \
		--varying $< \
		--fixed $(FIXED_YAML) \
		--time $(TIME_YAML) \
		--frequency $(FREQ_YAML) \
		--outdir input/generated/$(MODEL)/$(RESO)/beta/

plotconfigs/lightcurve.yaml:
	@echo "エラー: ファイルがありません $@"
	@false
plotconfigs/spectrum.yaml:
	@echo "エラー: ファイルがありません $@"
	@false
