NUMBERS := $(shell seq -f "%03g" 10 50)

CSVS := $(addsuffix .csv,$(NUMBERS))

FIGDIR := fig
SCENARIODIR ?= thermal_only
VARYINGDIR ?= beta
DATADIR := data/thermal_only/beta

INDIR := input/thermal_only/beta

OUTDIR := $(DATADIR)/spectrum

TARGETS := $(addprefix $(OUTDIR)/beta_,$(CSVS))

all: $(TARGETS)

$(OUTDIR)/beta_%.csv: $(INDIR)/beta_%.yaml
	python -m pipelines.beta $< $@

$(DATADIR)/statics_lpeak.csv: $(TARGETS)
	python -m pipelines.statics_lnu $(OUTDIR) $@

$(FIGDIR)/thermal_only/beta/statics_lnu_peak.pdf: $(DATADIR)/statics_lpeak.csv
	python -m plot.statics_lnu_peak $< $@
