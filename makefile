NUMBERS := $(shell seq -f "%03g" 10 30)

CSVS := $(addsuffix .csv,$(NUMBERS))

FIGDIR := fig
SCENARIODIR ?= thermal_only
VARYINGDIR ?= beta
DATADIR := data

INDIR := input/thermal_only/beta
OUTDIR := data/thermal_only/beta

TARGETS := $(addprefix $(OUTDIR)/beta_,$(CSVS))

all: $(TARGETS)

$(OUTDIR)/beta_%.csv: $(INDIR)/beta_%.yaml
	python -m pipelines.beta $<
