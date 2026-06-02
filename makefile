INPUTDIR := input
DATADIR := data
FIGDIR := fig

BETA_DIR := beta
BETA ?= 10
BETA_FMT := $(shell printf "%03d" $(BETA))
TIME_RESO ?= 64
SPEC_RESO ?= 256
RESO := t$(TIME_RESO)f$(SPEC_RESO)

THERMAL_DIR := thermal_only
MODEL ?= $(THERMAL_DIR)
SPECTRUM_PEAK_EVOLUTION := spectrum_peak_evolution

TIME_YAML  := $(INPUTDIR)/template/$(RESO)/time.yaml
FREQ_YAML := $(INPUTDIR)/template/$(RESO)/frequency.yaml
VARYING_YAML := $(INPUTDIR)/template/$(RESO)/varying_beta.yaml
FIXED_YAML := $(INPUTDIR)/template/$(RESO)/$(BETA_DIR)/base.yaml

data/test/peak_time_for_given_nu.csv: test/test_peak_time_for_given_nu.py
	python -m test.test_peak_time_for_given_nu $@

fig/test/tau_theta__lnu_peak.pdf: \
	data/test/tau_theta_dependence.csv\
	plot/test/test_tau_theta_dependence.py\
	plotconfigs/tau_theta__lnu_peak.yaml
	python -m plot.test.test_tau_theta_dependence \
		$< \
		-o $@ \
		-c $(lastword $^)

fig/test/tau_theta__tau_peak.pdf: \
	data/test/tau_theta_dependence.csv\
	plot/test/test_tau_theta_dependence.py\
	plotconfigs/tau_theta__tau_peak.yaml
	python -m plot.test.test_tau_theta_dependence \
		$< \
		-o $@ \
		-c $(lastword $^)

fig/test/tau_theta__xi_peak.pdf: \
	data/test/tau_theta_dependence.csv\
	plot/test/test_tau_theta_dependence.py\
	plotconfigs/tau_theta__xi_peak.yaml
	python -m plot.test.test_tau_theta_dependence \
		$< \
		-o $@ \
		-c $(lastword $^)

data/test/tau_theta_dependence.csv: test/test_tau_theta__xi.py
	python -m test.test_tau_theta__xi $@

.SECONDEXPANSION:

$(FIGDIR)/$(MODEL)/$(RESO)/beta/$(BETA_FMT)/spectrum/%.pdf: \
	$(DATADIR)/$(MODEL)/$(RESO)/beta/$(BETA_FMT)/spectrum/.done \
		pipelines/plot/spectrum.py \
		plotconfigs/spectrum.yaml
	python -m pipelines.plot.spectrum \
		$(DATADIR)/$(MODEL)/$(RESO)/beta/$(BETA_FMT)/spectrum/$*.csv \
 		-o $@ \
		-c $(lastword $^)

$(FIGDIR)/$(MODEL)/$(RESO)/beta/$(BETA_FMT)/$(SPECTRUM_PEAK_EVOLUTION).pdf: \
	$(DATADIR)/$(MODEL)/$(RESO)/beta/$(BETA_FMT)/$(SPECTRUM_PEAK_EVOLUTION).csv \
	pipelines/plot/$(SPECTRUM_PEAK_EVOLUTION).py \
	plotconfigs/$(SPECTRUM_PEAK_EVOLUTION).yaml
	python -m pipelines.plot.$(SPECTRUM_PEAK_EVOLUTION) \
		$< \
		-o $@ \
		-c $(lastword $^)

$(FIGDIR)/$(MODEL)/$(RESO)/beta/$(BETA_FMT)/lightcurve.pdf: \
	$$(wildcard $(DATADIR)/$(MODEL)/$(RESO)/beta/$$*/spectrum/[0-9][0-9][0-9].csv) \
	$(DATADIR)/$(MODEL)/$(RESO)/beta/$$*/spectrum/.done \
	pipelines/plot/lightcurve.py \
	plotconfigs/lightcurve.yaml
	python -m pipelines.plot.lightcurve \
		$(filter %.csv,$^) \
		-o $@ \
		-c $(lastword $^)

.SECONDARY:
$(DATADIR)/$(MODEL)/$(RESO)/beta/$(BETA_FMT)/$(SPECTRUM_PEAK_EVOLUTION).csv: \
	$$(wildcard $(DATADIR)/$(MODEL)/$(RESO)/beta/$(BETA_FMT)/spectrum/[0-9][0-9][0-9].csv) \
	pipelines/spectrum_peak_evolution.py
	python -m pipelines.spectrum_peak_evolution \
		$(filter %.csv,$^) \
		-o $@ \

$(DATADIR)/$(MODEL)/$(RESO)/beta/$(BETA_FMT)/spectrum/.done: \
	input/generated/$(MODEL)/$(RESO)/beta/$(BETA_FMT).yaml \
		pipelines/compute.py
	python -m pipelines.compute \
		$< \
		--outdir $(DATADIR)/$(MODEL)/$(RESO)/beta/$(BETA_FMT)/spectrum
	mkdir -p $(dir $@)
	touch $@

input/generated/$(MODEL)/$(RESO)/beta/$(BETA_FMT).yaml: \
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
test/test_tau_xi_dependence.py:
	@echo "エラー: ファイルがありません $@"
	@false
plotconfigs/tau_xi_dependence.yaml:
	@echo "エラー: ファイルがありません $@"
	@false
