NUMBERS := 010 020 030

CSVS := $(addsuffix .csv,$(NUMBERS))

INDIR := input/thermal_only/beta
OUTDIR := data/thermal_only/beta

TARGETS := $(addprefix $(OUTDIR)/beta_,$(CSVS))

all: $(TARGETS)

$(OUTDIR)/beta_%.csv: $(INDIR)/beta%.yaml $(INDIR)/base.yaml pipelines/beta.py
	python -m pipelines.beta $<
