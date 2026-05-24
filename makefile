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

# CSVDIR := $(DATADIR)/$(MODEL)/$(BETA_DIR)/$(VARYING)/spectrum

TIME_YAML  := $(INPUTDIR)/template/$(RESO)/time.yaml
FREQ_YAML := $(INPUTDIR)/template/$(RESO)/frequency.yaml
VARYING_YAML := $(INPUTDIR)/template/$(RESO)/varying_beta.yaml
FIXED_YAML := $(INPUTDIR)/template/$(RESO)/$(BETA_DIR)/base.yaml

# $(FIGDIR)/thermal_only/beta/statics_lnu_peak.pdf: $(DATADIR)/statics_lpeak.csv
# 	python -m plot.statics_lnu_peak $< $@
#
# $(FIGDIR)/thermal_only/beta/statistics_nu_peak.pdf: $(DATADIR)/statics_lpeak.csv
# 	python -m plot.statistics.nu_peak $< $@
#
# $(FIGDIR)/thermal_only/beta/spectrum/luminosity_beta_%.pdf: $(OUTDIR)/beta_%.csv
# 	python -m plot.luminosity $< $@
#
# $(DATADIR)/$(MODEL)/$(RESO)/$(BETA_DIR)/statistics_lnu_peak.csv: $(TARGETS)
# 	python -m pipelines.statistics_lnu $(CSVDIR) $@

# $(SPECTRUMS): $(YAMLDIR)/$(BETA_DIR)/beta_010.yaml $(YAMLDIR)/frequency.yaml
# 	python -m pipelines.spectrum_evolution $< $@

# $(CSVDIR): $(YAMLDIR)/$(BETA_DIR)/$(VARYING).yaml \
#               $(FIXED_YAML) \
#               $(TIME_YAML) \
#               $(FREQ_YAML) \
#               pipelines/spectrum_evolution.py
#
# 	python -m pipelines.spectrum_evolution \
#     	"$<" \
#     	"$(FIXED_YAML)" \
#     	"$(TIME_YAML)" \
#     	"$(FREQ_YAML)" \
#     	-o "$@"
#
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

# $(YAMLS) &: $(INDIR)/general/varying_beta.yaml pipelines/separate_betas.py
# 	python -m pipelines.separate_betas $< $(INDIR)/$(MODEL)/$(BETA_DIR)
