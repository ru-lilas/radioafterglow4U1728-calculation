from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Hashable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from module import calculation_lightcurve as clc
from module import plot_utils
from module.models import ColumnNames, FluxName, MetadataNames, ModelParameters, SwitchKeys
from module.mydataclasses import LongformatDataFrame, QuantityArray1D, QuantityData
from module.observation import LongformatLightcurve, ObservedLightcurve
from module.utils import FileReader


@dataclass(frozen=True, slots=True)
class PlotMetadata:
    a_wind: QuantityData
    beta_sh: QuantityData
    eps_b: QuantityData
    eps_th: QuantityData
    mu_e_mu: QuantityData
    distance: QuantityData
    p_index: QuantityData
    f_e: QuantityData
    chi_squared: QuantityData

    @classmethod
    def from_model(
        cls,
        model: ModelParameters,
        chi_squared: float,
    ) -> PlotMetadata:
        return cls(
            a_wind=QuantityData.from_Quantity(model.a_wind.item()),
            beta_sh=QuantityData(model.beta_sh.item(), ""),
            eps_b=QuantityData(model.eps_b, ""),
            eps_th=QuantityData(model.eps_th, ""),
            mu_e_mu=QuantityData(model.molecular_weight.to_ratio(), ""),
            distance=model.distance,
            p_index=QuantityData(model.powerlaw.p_index, ""),
            f_e=QuantityData(model.powerlaw.f_e, ""),
            chi_squared=QuantityData(chi_squared, ""),
        )

    def to_dict(self) -> dict[str, QuantityData]:
        values: dict[str, QuantityData] = {}
        for field in fields(self):
            quantity = getattr(self, field.name)
            if quantity is not None:
                values[field.name] = quantity
        return values


@dataclass(frozen=True, slots=True)
class ColumnPlotData:
    observed: ObservedLightcurve
    binned: clc.BinnedLightcurveWithBackground
    continuous: clc.ContinuousLightcurve
    metadata: PlotMetadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("bestfit", "candidates"),
        default="bestfit",
        help="描画モード（既定値: bestfit）",
    )
    # 呼び出し側との互換性のため、既存の--inputも受け付ける。
    for name in (
        "input",
        "candidate_lightcurves",
        "model_parameter_table",
        "obslc",
        "plotconfig",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--binned_lightcurves", "--candidate_binned_lightcurves",
        dest="binned_lightcurves", type=Path, required=True,
    )
    parser.add_argument("--minimum_chi_squared", type=Path)
    args = parser.parse_args()
    if args.mode == "bestfit" and args.minimum_chi_squared is None:
        parser.error("--mode bestfitでは--minimum_chi_squaredが必要です")
    return args


def select_model_df(
    df: pd.DataFrame,
    model_index: int,
    nu_value: float,
) -> pd.DataFrame:
    selected = df.loc[df[ColumnNames.MODEL_INDEX] == model_index].copy()
    selected.attrs.update({
        ColumnNames.NU: nu_value,
        ColumnNames.MODEL_INDEX: model_index,
    })
    return selected


def build_continuous_lightcurve(
    df: pd.DataFrame,
    model_index: int,
    nu: QuantityData,
    flux_name: FluxName,
) -> clc.ContinuousLightcurve:
    return clc.ContinuousLightcurve(
        model_index=model_index,
        nu=nu,
        t=QuantityArray1D.from_df(df, ColumnNames.T, unit_key=MetadataNames.T_UNIT),
        fnu_net=QuantityArray1D.from_df(df, flux_name, unit_key=MetadataNames.FNU_UNIT),
    )


def prepare_column_data(
    result: clc.ChiSquaredResult,
    nu_unit: str,
    dfs_binned_nu: dict[Hashable, pd.DataFrame],
    dfs_continuous_nu: dict[Hashable, pd.DataFrame],
    lcs_obs: LongformatLightcurve,
    model_parameters: ModelParameters,
    *,
    flux_name: FluxName,
) -> ColumnPlotData:
    model_index = int(result.model_index)
    nu = QuantityData(value=result.nu, unit=nu_unit)
    df_binned = select_model_df(dfs_binned_nu[result.nu], model_index, result.nu)
    df_continuous = select_model_df(dfs_continuous_nu[result.nu], model_index, result.nu)
    selected_model = model_parameters.select_models([model_index])
    observed = lcs_obs.select_frequency(nu, unit=lcs_obs.metadata.nu_unit)
    return ColumnPlotData(
        observed=observed,
        binned=clc.BinnedLightcurveWithBackground.from_df(
            df_binned, observed.fnu_background.index(0),
        ),
        continuous=build_continuous_lightcurve(df_continuous, model_index, nu, flux_name),
        metadata=PlotMetadata.from_model(
            selected_model, result.chi_squared,
        ),
    )


def plot_lightcurves(
    plotter: plot_utils.Plotter,
    data: ColumnPlotData,
    plotconf: plot_utils.PlotConfig,
    col: int,
    *,
    x_unit: str,
    y_unit: str,
) -> None:
    components = plotconf.components
    observed = data.observed
    t_obs = observed.t.to_value_in(x_unit)
    background = observed.fnu_background.to_value_in(y_unit)
    plotter.line(
        x=t_obs,
        y=background,
        style=components["background"].resolve_line_style(col),
    )
    plotter.errorbar(
        x=t_obs,
        y=observed.fnu_total.to_value_in(y_unit),
        yerr=observed.fnu_total_err.to_value_in(y_unit),
        style=components["observed"].resolve_errorbar_style(col),
    )
    plotter.step(
        edges=data.binned.bin_edges.to_value_in(x_unit),
        values=data.binned.fnu_with_bg.to_value_in(y_unit),
        style=components["binned"].resolve_line_style(col),
        baseline=background[0],
    )
    plotter.line(
        x=data.continuous.t.to_value_in(x_unit),
        y=data.continuous.fnu_with_bg(observed.fnu_background.index(0)).to_value_in(y_unit),
        style=components["continuous"].resolve_line_style(col),
    )


def add_model_metadata(
    plotter: plot_utils.Plotter,
    metadata: PlotMetadata,
    plotconf: plot_utils.PlotConfig,
    synchrotron_label: str,
) -> None:
    plotter.text(
        x=0.98,
        y=0.98,
        label=synchrotron_label,
        style=plotconf.components["synchrotron_component"].text_style_config,
        coordinates="axes",
    )
    strings: list[str] = []
    for name, quantity in metadata.to_dict().items():
        plot_utils.MetadataComponent(
            quantity=quantity,
            config=plotconf.metadata.items[name],
        ).add_strings(strings)
    plotter.title(strings=strings, fontsize=plotconf.metadata.fontsize)


def main() -> None:
    args = parse_args()
    if args.mode == "candidates":
        raise NotImplementedError("candidatesモードの描画処理は未実装です")
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plotconf = plot_utils.PlotConfig.from_yaml(args.plotconfig)
    df_minchi2 = FileReader.table_from_csv(args.minimum_chi_squared)
    dfs_binned_nu = LongformatDataFrame.read_from(args.binned_lightcurves).parse(ColumnNames.NU)
    dfs_continuous_nu = LongformatDataFrame.read_from(args.candidate_lightcurves).parse(ColumnNames.NU)
    lcs_obs = LongformatLightcurve.from_csv(args.obslc)
    model_parameters = ModelParameters.from_df(FileReader.table_from_csv(args.model_parameter_table))
    nu_unit = df_minchi2.attrs[MetadataNames.NU_UNIT]
    thermal_only = model_parameters.switches[SwitchKeys.THERMAL_ONLY]
    flux_name = FluxName.FNU_THERMAL if thermal_only else FluxName.FNU_TOTAL
    synchrotron_label = "thermal" if thermal_only else "thermal+powerlaw"

    with PdfPages(outpath) as pdf:
        fig, axes = plotconf.layout.create()
        try:
            column_data: dict[int, ColumnPlotData] = {}
            for (row, col), ax in np.ndenumerate(axes):
                if col not in column_data:
                    column_data[col] = prepare_column_data(
                        clc.ChiSquaredResult.from_series(df_minchi2.iloc[col]),
                        nu_unit,
                        dfs_binned_nu,
                        dfs_continuous_nu,
                        lcs_obs,
                        model_parameters,
                        flux_name=flux_name,
                    )
                data = column_data[col]
                axis_config = plotconf.layout.axes.for_subplot(row, col)
                plotter = plot_utils.Plotter(ax)
                plot_lightcurves(
                    plotter, data, plotconf, col,
                    x_unit=axis_config.x_unit, y_unit=axis_config.y_unit,
                )
                add_model_metadata(plotter, data.metadata, plotconf, synchrotron_label)
            pdf.savefig(fig)
        finally:
            plt.close(fig)
    print(f"Saved lightcurves: {args.output}", flush=True)


if __name__ == "__main__":
    main()
