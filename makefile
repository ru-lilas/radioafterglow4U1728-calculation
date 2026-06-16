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
TABLE := integral_table

$(FIGDIR)/test/test_peak_matching/spectrum/%.pdf: \
	$(DATADIR)/test/test_peak_matching/spectrum/%.csv \
	$(DATADIR)/test/test_peak_matching/spectrum/.done \
		plot/test/test_peak_matching.py \
		plotconfigs/test/test_peak_matching.yaml
	python -m plot.test.test_peak_matching \
		$< \
 		-o $@ \
		-c $(lastword $^)

$(FIGDIR)/integral_ip.pdf: \
	$(DATADIR)/tabular/xi.csv \
		plot/integral_tabular.py \
		plotconfigs/integral_tabular.yaml
	python -m plot.integral_tabular \
		$< \
 		-o $@ \
		-c $(lastword $^)

$(FIGDIR)/test/%.pdf: \
	$(DATADIR)/test/%.csv \
		plot/test/%.py \
		plotconfigs/test/%.yaml
	python -m plot.test.$* \
		$< \
 		-o $@ \
		-c $(lastword $^)

$(DATADIR)/test/%.csv: \
	input/test/%.yaml \
	test/%.py \
	data/tabular/xi.csv
	python -m test.$* \
		$<\
		--output $@ \
		--tabular $(lastword $^)

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

data/test/peak_time_for_given_nu.csv: test/test_peak_time_for_given_nu.py data/tabular/xi.csv
	python -m test.test_peak_time_for_given_nu \
		--tabular $(lastword $^)

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

fig/synchrotron_functions/%.pdf: \
	data/tabular/xi.csv \
	plot/test/test_tau_theta_dependence.py \
	plotconfigs/&.yaml
	python -m plot.test.test_tau_theta_dependence \
		$< \
		-o $@ \
		-c $(lastword $^)

data/test/tau_theta_dependence.csv: test/test_tau_theta__xi.py data/tabular/xi.csv
	python -m test.test_tau_theta__xi \
		$@ \
		--tabular $(lastword $^)

$(DATADIR)/table_%.csv: \
	input/table_%.yaml \
		$(DATADIR)/$(TABLE).csv \
		scripts/table_%.py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--output $@ \
		--table $(word 2,$^)

$(DATADIR)/$(TABLE).csv: \
	input/$(TABLE).yaml \
		scripts/$(TABLE).py
	python -m $(subst /,.,$(basename $(lastword $^))) \
		$< \
		--output $@

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
$(DATADIR)/test/test_peak_deviations.csv: \
	$$(wildcard $(DATADIR)/test/test_peak_matching/spectrum/[0-9][0-9][0-9].csv) \
		$(DATADIR)/test/test_peak_matching/spectrum/.done
	python -m test.test_peak_deviations \
		$(filter %.csv,$^) \
		-o $@

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


input/%.yaml:
	@echo "エラー: インプットファイルがありません $@"
	@false
plotconfigs/%.yaml:
	@echo "エラー: ファイルがありません $@"
	@false
plotconfigs/test/%.yaml:
	@echo "エラー: プロットコンフィグファイルがありません $@"
	@false
test/test_tau_xi_dependence.py:
	@echo "エラー: ファイルがありません $@"
	@false
plotconfigs/tau_xi_dependence.yaml:
	@echo "エラー: ファイルがありません $@"
	@false
