# 実行例: make RUN_ID=1 chi2
.DEFAULT_GOAL := all
.SILENT:

PYTHON ?= python3
PDF_VIEWER ?= evince

INPUTDIR := input
DATADIR := data
FIGDIR := fig
OBSDIR := obs
PLOTCONFIGDIR := plotconfigs

ifndef RUN_ID
$(error RUN_IDを指定してください. 例 make RUN_ID=1)
endif
RUN_DIR := $(shell printf 'run_%06d' '$(RUN_ID)')
RUN_DATADIR := $(DATADIR)/$(RUN_DIR)
RUN_FIGDIR := $(FIGDIR)/$(RUN_DIR)

# パス生成関数: $(1) は拡張子を含むファイル名
run_data = $(RUN_DATADIR)/$(1)
run_fig = $(RUN_FIGDIR)/$(1)
data_file = $(DATADIR)/$(1)
obs_file = $(DATADIR)/$(OBSDIR)/$(1)

# 入力・共通テーブル・観測データ
FILEPATH_INPUT := $(call run_data,input.yaml)
FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE := $(call data_file,synchrotron_function_numerical_table.csv)
FILEPATH_THERMAL_PEAK_TABLE := $(call data_file,thermal_peak_table.csv)
FILEPATH_OBSLC := $(call obs_file,obslc.csv)
FILEPATH_OBSLC_TMP := $(call obs_file,obslc_tmp.csv)
FILEPATH_OBSBG := $(call obs_file,bg.csv)

# RUN ごとの計算結果
FILEPATH_BINNING := $(call run_data,binning.csv)
FILEPATH_MODEL_PARAMETER_TABLE := $(call run_data,model_parameter_table.csv)
FILEPATH_MODEL_PARAMETER_TABLE_POWERLAW := $(call run_data,model_parameter_table_powerlaw.csv)
FILEPATH_PEAK_GRID := $(call run_data,peak_grid.csv)
FILEPATH_MODEL_LIGHTCURVES := $(call run_data,model_lightcurves.csv)
FILEPATH_CANDIDATE_MODELS := $(call run_data,candidate_models.csv)
FILEPATH_CANDIDATE_LIGHTCURVES := $(call run_data,candidate_lightcurves.csv)
FILEPATH_BINNED_LIGHTCURVES := $(call run_data,binned_lightcurves.csv)
FILEPATH_POWERLAW_LIGHTUCURVES:= $(call run_data,powerlaw_lightcurves.csv)
FILEPATH_POWERLAW_LIGHTUCURVES_PDF:= $(call run_fig,powerlaw_lightcurves.pdf)
FILEPATH_CHI_SQUARED := $(call run_data,chi_squared.csv)
FILEPATH_MINIMUM_CHI_SQUARED := $(call run_data,minimum_chi_squared.csv)
FILEPATH_CHEVALIER_BESTFIT_POINTS := $(call run_data,chevalier_bestfit_points.csv)
FILEPATH_ENERGETICS := $(call run_data,energetics.csv)

FILEPATH_MINIMUM_CHI2_LC_PDF := $(call run_fig,minimum_chi2_lightcurve.pdf)
FILEPATH_CHEVALIER_PDF := $(call run_fig,chevalier_diagram.pdf)
FILEPATH_CHEVALIER_WITH_BESTFITS_PDF := $(call run_fig,chevalier_diagram_with_bestfits.pdf)
FILEPATH_CHEVALIER_COLORMAP_PDF := $(call run_fig,chevalier_colormap.pdf)
FILEPATH_DIMENSIONLESS_SYNCHROTRONS_PDF:= $(call run_fig,dimensionless_synchrotrons.pdf)
FILEPATH_OPTICALDEPTH_CMAPS_PDF:=$(call run_fig,opticaldepth_cmaps.pdf)
FILEPATH_THERMAL_PEAK_PDF:=$(call run_fig,thermal_peak.pdf)
FILEPATH_PEAK_CUTOFF_RATIO_PDF:=$(call run_fig,peak_cutoff_ratio.pdf)

# 共通CLI引数（$<・$@はレシピ実行時に展開するため = を使う）
ARGS_INPUT = --input $<
ARGS_OUTPUT = --output $@
ARGS_OBSLC = --obslc $(FILEPATH_OBSLC)
ARGS_INTEGRAL_TABLE = --integral_table $(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE)
ARGS_MODEL_PARAMETER_TABLE = --model_parameter_table $(FILEPATH_MODEL_PARAMETER_TABLE)
ARGS_MODEL_PARAMETER_TABLE_POWERLAW = --model_parameter_table_powerlaw $(FILEPATH_MODEL_PARAMETER_TABLE_POWERLAW)
ARGS_PEAK_GRID = --peak_grid $(FILEPATH_PEAK_GRID)
ARGS_CANDIDATE_LIGHTCURVES = --candidate_lightcurves $(FILEPATH_CANDIDATE_LIGHTCURVES)
ARGS_BINNED_LIGHTCURVES = --binned_lightcurves $(FILEPATH_BINNED_LIGHTCURVES)
ARGS_POWERLAW_LIGHTCURVES = --powerlaw_lightcurves $(FILEPATH_POWERLAW_LIGHTUCURVES)
ARGS_BINNING = --binning $(FILEPATH_BINNING)
ARGS_CHEVALIER_BESTFIT_POINTS = --chevalier_bestfit_points $(FILEPATH_CHEVALIER_BESTFIT_POINTS)
# ARGS_CONTINUOUS_LIGHTCURVE = --continuous_lightcurves $(FILEPATH_CONTINUOUS_LIGHTCURVES)

.PHONY: all view-best-lc view-chevalier view-chevalier_cmp \
    chevaliercm chevalier min_chi2 chi2 lc_bin lc candidates \
    peak_grid lc_all modeltable peak_table

all: view-best-lc view-chevalierbest view-chevaliercm

view-best-lc: $(FILEPATH_MINIMUM_CHI2_LC_PDF)
	$(PDF_VIEWER) "$<" >/dev/null 2>&1 &

view-dimless-syn: $(FILEPATH_DIMENSIONLESS_SYNCHROTRONS_PDF)
	$(PDF_VIEWER) "$<" >/dev/null 2>&1 &

view-pl-lc: $(FILEPATH_POWERLAW_LIGHTUCURVES_PDF)
	$(PDF_VIEWER) "$<" >/dev/null 2>&1 &


$(FILEPATH_DIMENSIONLESS_SYNCHROTRONS_PDF): \
	$(FILEPATH_POWERLAW_LIGHTUCURVES) \
	$(PLOTCONFIGDIR)/dimensionless_synchrotrons.yaml \
	scripts/plot_dimensionless_synchrotrons.py
	$(PYTHON) -m scripts.plot_dimensionless_synchrotrons\
		--plotconfig $(PLOTCONFIGDIR)/dimensionless_synchrotrons.yaml \
		$(ARGS_POWERLAW_LIGHTCURVES) \
		$(ARGS_MODEL_PARAMETER_TABLE_POWERLAW) \
		$(ARGS_OUTPUT)

$(FILEPATH_POWERLAW_LIGHTUCURVES_PDF): \
	$(FILEPATH_POWERLAW_LIGHTUCURVES) \
	$(FILEPATH_BINNED_LIGHTCURVES) \
	$(PLOTCONFIGDIR)/lightcurves_powerlaw_sensitivity.yaml \
	$(FILEPATH_OBSLC) \
	scripts/plot_powerlaw_lightcurves.py
	$(PYTHON) -m scripts.plot_powerlaw_lightcurves\
		--plotconfig $(PLOTCONFIGDIR)/lightcurves_powerlaw_sensitivity.yaml \
		$(ARGS_POWERLAW_LIGHTCURVES) \
		$(ARGS_BINNED_LIGHTCURVES) \
		$(ARGS_MODEL_PARAMETER_TABLE_POWERLAW) \
		$(ARGS_OBSLC) \
		$(ARGS_OUTPUT)

view-thermal-peak: $(FILEPATH_THERMAL_PEAK_PDF)
	$(PDF_VIEWER) "$<" >/dev/null 2>&1 &
$(FILEPATH_THERMAL_PEAK_PDF): \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(PLOTCONFIGDIR)/thermal_peak.yaml \
	scripts/plot_thermal_peak.py
	$(PYTHON) -m scripts.plot_thermal_peak \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		$(ARGS_INTEGRAL_TABLE) \
		--plotconfig $(PLOTCONFIGDIR)/thermal_peak.yaml \
		$(ARGS_OUTPUT)

view-opt-cmap: $(FILEPATH_OPTICALDEPTH_CMAPS_PDF)
	$(PDF_VIEWER) "$<" >/dev/null 2>&1 &
opticaldepth_cmaps: $(FILEPATH_OPTICALDEPTH_CMAPS_PDF)
$(FILEPATH_OPTICALDEPTH_CMAPS_PDF): \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(PLOTCONFIGDIR)/opticaldepth_cmaps.yaml \
	scripts/plot_opticaldepth_cmaps.py
	$(PYTHON) -m scripts.plot_opticaldepth_cmaps \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		$(ARGS_INTEGRAL_TABLE) \
		--plotconfig $(PLOTCONFIGDIR)/opticaldepth_cmaps.yaml \
		$(ARGS_OUTPUT)

$(FILEPATH_MINIMUM_CHI2_LC_PDF): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_MINIMUM_CHI_SQUARED) \
	$(FILEPATH_BINNED_LIGHTCURVES) \
	$(FILEPATH_CANDIDATE_LIGHTCURVES) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(FILEPATH_OBSLC) \
	$(PLOTCONFIGDIR)/minimum_chi_squared_lightcurves.yaml \
	scripts/plot_lightcurve.py
	$(PYTHON) -m scripts.plot_lightcurve \
		$(ARGS_INPUT) \
		--minimum_chi_squared $(FILEPATH_MINIMUM_CHI_SQUARED) \
		$(ARGS_BINNED_LIGHTCURVES) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		$(ARGS_CANDIDATE_LIGHTCURVES) \
		$(ARGS_OBSLC) \
		--plotconfig $(PLOTCONFIGDIR)/minimum_chi_squared_lightcurves.yaml \
		$(ARGS_OUTPUT)

view-chevaliercm: $(FILEPATH_CHEVALIER_COLORMAP_PDF)
	$(PDF_VIEWER) "$<" >/dev/null 2>&1 &
chevaliercm: $(FILEPATH_CHEVALIER_COLORMAP_PDF)
$(FILEPATH_CHEVALIER_COLORMAP_PDF): \
	$(FILEPATH_PEAK_GRID) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(FILEPATH_OBSLC) \
	$(PLOTCONFIGDIR)/chevalier_colormap.yaml \
	module/chevalier_plotting.py \
	scripts/plot_chevalier_colormap.py
	$(PYTHON) -m scripts.plot_chevalier_colormap \
		$(ARGS_PEAK_GRID) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		--plotconfig $(PLOTCONFIGDIR)/chevalier_colormap.yaml \
		$(ARGS_OBSLC) \
		$(ARGS_OUTPUT)

view-chevalierbest: $(FILEPATH_CHEVALIER_WITH_BESTFITS_PDF)
	$(PDF_VIEWER) "$<" >/dev/null 2>&1 &
$(FILEPATH_CHEVALIER_WITH_BESTFITS_PDF): \
	$(FILEPATH_PEAK_GRID) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(FILEPATH_OBSLC) \
	$(PLOTCONFIGDIR)/chevalier_diagram.yaml \
	module/chevalier_plotting.py \
	module/chevalier.py \
	module/plot_utils.py \
	scripts/plot_chevalier_diagram.py \
	$(FILEPATH_CHEVALIER_BESTFIT_POINTS) \
	module/chevalier.py
	$(PYTHON) -m scripts.plot_chevalier_diagram \
		$(ARGS_PEAK_GRID) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		--plotconfig $(PLOTCONFIGDIR)/chevalier_diagram.yaml \
		$(ARGS_OBSLC) \
		$(ARGS_CHEVALIER_BESTFIT_POINTS) \
		$(ARGS_OUTPUT)

view-chevalier: $(FILEPATH_CHEVALIER_PDF)
	$(PDF_VIEWER) "$<" >/dev/null 2>&1 &

chevalier: $(FILEPATH_CHEVALIER_PDF)

$(FILEPATH_CHEVALIER_PDF): \
	$(FILEPATH_PEAK_GRID) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(FILEPATH_OBSLC) \
	module/chevalier.py \
	$(PLOTCONFIGDIR)/chevalier_diagram.yaml \
	scripts/plot_chevalier_diagram.py
	$(PYTHON) -m scripts.plot_chevalier_diagram \
		$(ARGS_PEAK_GRID) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		--plotconfig $(PLOTCONFIGDIR)/chevalier_diagram.yaml \
		$(ARGS_OBSLC) \
		$(ARGS_OUTPUT)

view-peak-cutoff-ratio: $(FILEPATH_PEAK_CUTOFF_RATIO_PDF)
	$(PDF_VIEWER) "$<" >/dev/null 2>&1 &
$(FILEPATH_PEAK_CUTOFF_RATIO_PDF): \
	$(FILEPATH_PEAK_GRID) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(PLOTCONFIGDIR)/peak_cutoff_ratio.yaml \
	scripts/plot_peak_cutoff_ratio.py
	$(PYTHON) -m scripts.plot_peak_cutoff_ratio \
		$(ARGS_PEAK_GRID) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		--plotconfig $(PLOTCONFIGDIR)/peak_cutoff_ratio.yaml \
		$(ARGS_OUTPUT)

#=== datafiles ===#
energetics: $(FILEPATH_ENERGETICS)
$(FILEPATH_ENERGETICS): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_CHEVALIER_BESTFIT_POINTS) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	scripts/calculate_energetics.py
	$(PYTHON) -m scripts.calculate_energetics \
		$(ARGS_INPUT) \
		$(ARGS_CHEVALIER_BESTFIT_POINTS) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		$(ARGS_OUTPUT)

chevalier-bestfit-points: $(FILEPATH_CHEVALIER_BESTFIT_POINTS)
$(FILEPATH_CHEVALIER_BESTFIT_POINTS): \
	$(FILEPATH_MINIMUM_CHI_SQUARED) \
	$(FILEPATH_BINNED_LIGHTCURVES) \
	module/chevalier.py \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	scripts/build_chevalier_bestfit_points.py
	$(PYTHON) -m scripts.build_chevalier_bestfit_points \
		--minimum_chi_squared $(FILEPATH_MINIMUM_CHI_SQUARED) \
		$(ARGS_BINNED_LIGHTCURVES) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		$(ARGS_OUTPUT)

powerlaw_lightcurves: $(FILEPATH_POWERLAW_LIGHTUCURVES)
$(FILEPATH_POWERLAW_LIGHTUCURVES): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_MODEL_PARAMETER_TABLE_POWERLAW) \
	$(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
	$(FILEPATH_MODEL_PARAMETER_TABLE_POWERLAW) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	scripts/compute_lightcurves_powerlaw.py
	$(PYTHON) -m scripts.compute_lightcurves_powerlaw \
		$(ARGS_INPUT) \
		--integral_table $(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
		--model_parameter_table_powerlaw $(FILEPATH_MODEL_PARAMETER_TABLE_POWERLAW) \
		--model_parameter_table $(FILEPATH_MODEL_PARAMETER_TABLE) \
		$(ARGS_OUTPUT)

powerlaw_table: $(FILEPATH_MODEL_PARAMETER_TABLE_POWERLAW)
$(FILEPATH_MODEL_PARAMETER_TABLE_POWERLAW): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_MINIMUM_CHI_SQUARED) \
	scripts/model_parameter_table_powerlaw.py
	$(PYTHON) -m scripts.model_parameter_table_powerlaw \
		$(ARGS_INPUT) \
		--minimum_chi_squared $(FILEPATH_MINIMUM_CHI_SQUARED) \
		$(ARGS_OUTPUT)

min_chi2: $(FILEPATH_MINIMUM_CHI_SQUARED)
$(FILEPATH_MINIMUM_CHI_SQUARED): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_CHI_SQUARED) \
	scripts/find_minimum_chi_squared.py
	$(PYTHON) -m scripts.find_minimum_chi_squared \
		$(ARGS_INPUT) \
		--chi_squared $(FILEPATH_CHI_SQUARED) \
		$(ARGS_OUTPUT)

chi2: $(FILEPATH_CHI_SQUARED)
$(FILEPATH_CHI_SQUARED): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_BINNED_LIGHTCURVES) \
	$(FILEPATH_OBSLC) \
	scripts/calculate_chi_squared.py
	$(PYTHON) -m scripts.calculate_chi_squared \
		$(ARGS_INPUT) \
		$(ARGS_BINNED_LIGHTCURVES) \
		$(ARGS_OBSLC) \
		$(ARGS_OUTPUT)

lc_bin: $(FILEPATH_BINNED_LIGHTCURVES)
$(FILEPATH_BINNED_LIGHTCURVES): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_CANDIDATE_LIGHTCURVES) \
	$(FILEPATH_BINNING) \
	scripts/compute_binned_lightcurves.py
	$(PYTHON) -m scripts.compute_binned_lightcurves \
		$(ARGS_INPUT) \
		--continuous_lightcurves $(FILEPATH_CANDIDATE_LIGHTCURVES) \
		$(ARGS_BINNING) \
		$(ARGS_OUTPUT)

lc: $(FILEPATH_CANDIDATE_LIGHTCURVES)
$(FILEPATH_CANDIDATE_LIGHTCURVES): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
	$(FILEPATH_CANDIDATE_MODELS) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	scripts/compute_lightcurve_candidate_models.py
	$(PYTHON) -m scripts.compute_lightcurve_candidate_models \
		$(ARGS_INPUT) \
		$(ARGS_INTEGRAL_TABLE) \
		--candidate_models $(FILEPATH_CANDIDATE_MODELS) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		$(ARGS_OUTPUT)

candidates: $(FILEPATH_CANDIDATE_MODELS)
$(FILEPATH_CANDIDATE_MODELS): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_PEAK_GRID) \
	$(FILEPATH_OBSLC) \
	scripts/select_candidate_models.py
	$(PYTHON) -m scripts.select_candidate_models \
		$(ARGS_INPUT) \
		$(ARGS_PEAK_GRID) \
		$(ARGS_OBSLC) \
		$(ARGS_OUTPUT)

#=== tabulating ===#
peak_grid: $(FILEPATH_PEAK_GRID)
$(FILEPATH_PEAK_GRID): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	scripts/compute_peak_grid.py
	$(PYTHON) -m scripts.compute_peak_grid \
		$(ARGS_INPUT) \
		$(ARGS_INTEGRAL_TABLE) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		$(ARGS_OUTPUT)

lc_all: $(FILEPATH_MODEL_LIGHTCURVES)
$(FILEPATH_MODEL_LIGHTCURVES): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
	scripts/compute_lightcurve_candidate_models.py
	$(PYTHON) -m scripts.compute_lightcurve_candidate_models \
		$(ARGS_INPUT) \
		$(ARGS_INTEGRAL_TABLE) \
		$(ARGS_OUTPUT)

modeltable: $(FILEPATH_MODEL_PARAMETER_TABLE)
$(FILEPATH_MODEL_PARAMETER_TABLE): \
	$(FILEPATH_INPUT) \
	module/models.py \
	scripts/build_model_parameter_table.py
	$(PYTHON) -m scripts.build_model_parameter_table \
		$< \
		$(ARGS_OUTPUT)

peak_table: $(FILEPATH_THERMAL_PEAK_TABLE)
$(FILEPATH_THERMAL_PEAK_TABLE): \
	$(INPUTDIR)/thermal_peak_table.yaml \
	$(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
	scripts/build_thermal_peak_table.py
	$(PYTHON) -m scripts.build_thermal_peak_table \
		$(ARGS_INPUT) \
		--synchrotron_function_numerical_table $(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE) \
		$(ARGS_OUTPUT)

binning: $(FILEPATH_BINNING)
$(FILEPATH_BINNING): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_OBSLC) \
	scripts/build_binning.py
	$(PYTHON) -m scripts.build_binning \
		$(ARGS_INPUT) \
		$(ARGS_OBSLC) \
		$(ARGS_OUTPUT)

table: $(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE)
$(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE): \
	$(INPUTDIR)/synchrotron_function_numerical_table.yaml \
	scripts/build_synchrotron_function_numerical_table.py
	$(PYTHON) -m scripts.build_synchrotron_function_numerical_table \
		$(ARGS_INPUT) \
		$(ARGS_OUTPUT)

#=== arrange observation data ===#
# OBSERVATION_PEAK_DATA は必要な場合のみ外部から指定する
ifdef OBSERVATION_PEAK_DATA
$(DATADIR)/$(OBSERVATION_PEAK_DATA).csv: \
	$(DATADIR)/obslc.csv \
	$(OBSERVATION_PEAK_DATA).py
	$(PYTHON) $(OBSERVATION_PEAK_DATA).py \
		$< \
		$(ARGS_OUTPUT)
endif

$(FILEPATH_OBSLC): \
	$(FILEPATH_OBSLC_TMP) \
	$(FILEPATH_OBSBG) \
	scripts/combine_obslc_background.py
	$(PYTHON) -m scripts.combine_obslc_background \
		--lc_tmp $< \
		--bg $(FILEPATH_OBSBG) \
		$(ARGS_OUTPUT)

$(FILEPATH_OBSBG): \
	$(FILEPATH_OBSLC_TMP) \
	scripts/estimate_background.py
	$(PYTHON) -m scripts.estimate_background \
		--data $< \
		$(ARGS_OUTPUT)

$(FILEPATH_OBSLC_TMP): \
	$(DATADIR)/observation_raw/4U1728_stacked_1min.dat \
	$(INPUTDIR)/obslc.yaml \
	preprocess_obslc.py
	$(PYTHON) preprocess_obslc.py \
		--data $< \
		--config $(INPUTDIR)/obslc.yaml \
		$(ARGS_OUTPUT)

fig/distribution_factors.pdf: \
	$(PLOTCONFIGDIR)/distribution_factors.yaml \
	scripts/plot_distribution_factors.py
	$(PYTHON) -m scripts.plot_distribution_factors \
		--plotconfig $(PLOTCONFIGDIR)/distribution_factors.yaml \
		$(ARGS_OUTPUT)

#=== config files ===#
$(PLOTCONFIGDIR)/%.yaml:
	@echo "エラー: プロットコンフィグがありません $@"
	@false
