# 実行例: make RUN_ID=1 chi2
.DEFAULT_GOAL := all

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
FILEPATH_MODEL_PARAMETER_TABLE := $(call run_data,model_parameter_table.csv)
FILEPATH_PEAK_GRID := $(call run_data,peak_grid.csv)
FILEPATH_MODEL_LIGHTCURVES := $(call run_data,model_lightcurves.csv)
FILEPATH_CANDIDATE_MODELS := $(call run_data,candidate_models.csv)
FILEPATH_CANDIDATE_LIGHTCURVES := $(call run_data,candidate_lightcurves.csv)
FILEPATH_CANDIDATE_BINNED_LIGHTCURVES := $(call run_data,candidate_binned_lightcurves.csv)
FILEPATH_CHI_SQUARED := $(call run_data,chi_squared.csv)
FILEPATH_MINIMUM_CHI_SQUARED := $(call run_data,minimum_chi_squared.csv)

FILEPATH_MINIMUM_CHI2_LC_PDF := $(call run_fig,minimum_chi2_lightcurve.pdf)
FILEPATH_CHEVALIER_PDF := $(call run_fig,chevalier_diagram.pdf)
FILEPATH_CHEVALIER_COLORMAP_PDF := $(call run_fig,chevalier_colormap.pdf)

# 共通CLI引数（$<・$@はレシピ実行時に展開するため = を使う）
ARGS_INPUT = --input $<
ARGS_OUTPUT = --output $@
ARGS_OBSLC = --obslc $(FILEPATH_OBSLC)
ARGS_INTEGRAL_TABLE = --integral_table $(FILEPATH_SYNCHROTRON_FUNCTION_NUMERICAL_TABLE)
ARGS_MODEL_PARAMETER_TABLE = --model_parameter_table $(FILEPATH_MODEL_PARAMETER_TABLE)
ARGS_PEAK_GRID = --peak_grid $(FILEPATH_PEAK_GRID)
ARGS_CANDIDATE_LIGHTCURVES = --candidate_lightcurves $(FILEPATH_CANDIDATE_LIGHTCURVES)
ARGS_CANDIDATE_BINNED_LIGHTCURVES = --candidate_binned_lightcurves $(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES)

.PHONY: all view-best-lc view-chevalier view-chevalier_cmp \
    chevaliercm chevalier min_chi2 chi2 lc_bin lc candidates \
    peak_grid lc_all modeltable peak_table

all: view-best-lc view-chevalier

view-best-lc: $(FILEPATH_MINIMUM_CHI2_LC_PDF)
	$(PDF_VIEWER) "$<" >/dev/null 2>&1 &

$(FILEPATH_MINIMUM_CHI2_LC_PDF): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_MINIMUM_CHI_SQUARED) \
	$(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES) \
	$(FILEPATH_CANDIDATE_LIGHTCURVES) \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(FILEPATH_OBSLC) \
	$(PLOTCONFIGDIR)/estimated_lightcurve.yaml \
	scripts/plot_minimum_chi2_lightcurve.py
	$(PYTHON) -m scripts.plot_minimum_chi2_lightcurve \
		$(ARGS_INPUT) \
		--minimum_chi_squared $(FILEPATH_MINIMUM_CHI_SQUARED) \
		$(ARGS_CANDIDATE_BINNED_LIGHTCURVES) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		$(ARGS_CANDIDATE_LIGHTCURVES) \
		$(ARGS_OBSLC) \
		--plotconfig $(PLOTCONFIGDIR)/estimated_lightcurve.yaml \
		$(ARGS_OUTPUT)

# 既存のターゲット名を互換用に残す
view-chevalier_cmp: view-chevalier

chevaliercm: $(FILEPATH_CHEVALIER_COLORMAP_PDF)
$(FILEPATH_CHEVALIER_COLORMAP_PDF): \
	$(FILEPATH_MODEL_PARAMETER_TABLE) \
	$(FILEPATH_OBSLC) \
	$(PLOTCONFIGDIR)/chevalier_colormaps.yaml \
	module/chevalier.py \
	scripts/chevalier_colormaps.py
	$(PYTHON) -m scripts.chevalier_colormaps \
		--model_table $< \
		--plotconfig $(PLOTCONFIGDIR)/chevalier_colormaps.yaml \
		$(ARGS_OBSLC) \
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
	scripts/chevalier.py
	$(PYTHON) -m scripts.chevalier \
		$(ARGS_PEAK_GRID) \
		$(ARGS_MODEL_PARAMETER_TABLE) \
		--plotconfig $(PLOTCONFIGDIR)/chevalier_diagram.yaml \
		$(ARGS_OBSLC) \
		$(ARGS_OUTPUT)

#=== datafiles ===#
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
	$(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES) \
	$(FILEPATH_OBSLC) \
	scripts/calculate_chi_squared.py
	$(PYTHON) -m scripts.calculate_chi_squared \
		$(ARGS_INPUT) \
		$(ARGS_CANDIDATE_BINNED_LIGHTCURVES) \
		$(ARGS_OBSLC) \
		$(ARGS_OUTPUT)

lc_bin: $(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES)
$(FILEPATH_CANDIDATE_BINNED_LIGHTCURVES): \
	$(FILEPATH_INPUT) \
	$(FILEPATH_CANDIDATE_LIGHTCURVES) \
	$(FILEPATH_OBSLC) \
	scripts/compute_candidate_binned_lightcurves.py
	$(PYTHON) -m scripts.compute_candidate_binned_lightcurves \
		$(ARGS_INPUT) \
		$(ARGS_CANDIDATE_LIGHTCURVES) \
		$(ARGS_OBSLC) \
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

#=== config files ===#
$(PLOTCONFIGDIR)/%.yaml:
	@echo "エラー: プロットコンフィグがありません $@"
	@false
