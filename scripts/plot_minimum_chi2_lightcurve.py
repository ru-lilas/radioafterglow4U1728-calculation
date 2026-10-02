import argparse
from pathlib import Path
from typing import Hashable, Literal, get_args
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import fields, dataclass
from matplotlib.backends.backend_pdf import PdfPages

from module import calculation_lightcurve as clc
from module import plot_utils
from module.models import ColumnNames, FluxName, MetadataNames, ModelParameters
from module.mydataclasses import LongformatDataFrame, QuantityArray1D, QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileReader

@dataclass(frozen=True,slots=True)
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

    def to_dict(self) -> dict[str, QuantityData]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "input",
        "minimum_chi_squared",
        "candidate_binned_lightcurves",
        "candidate_lightcurves",
        "model_parameter_table",
        "obslc",
        "plotconfig",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    outpath: Path = args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plotconf = plot_utils.PlotConfig.from_yaml(args.plotconfig)

    df_minchi2 = FileReader.table_from_csv(args.minimum_chi_squared)

    longdf_binned = LongformatDataFrame.read_from(args.candidate_binned_lightcurves)
    dfs_binned_nu = longdf_binned.parse(ColumnNames.NU)

    longdf_continuous = LongformatDataFrame.read_from(args.candidate_lightcurves)
    dfs_continuous_nu = longdf_continuous.parse(ColumnNames.NU)

    lcs_obs = LongformatLightcurve.from_csv(args.obslc)

    df_model_parameter = FileReader.table_from_csv(args.model_parameter_table)
    model_parameters = ModelParameters.from_df(df_model_parameter)

    layout = plotconf.layout
    components = plotconf.components
    metadata_config = plotconf.metadata
    nu_unit = df_minchi2.attrs[MetadataNames.NU_UNIT]

    with PdfPages(outpath) as pdf:
        fig, axes = layout.create()
        for (row,col), ax in np.ndenumerate(axes):
            row_minchi2 = clc.ChiSquaredResult.from_series(df_minchi2.iloc[col])
            axis_config = plotconf.layout.axes.for_subplot(row,col)
            plotter = plot_utils.Plotter(ax)
            x_unit:str = axis_config.x_unit
            y_unit:str = axis_config.y_unit
            model_index = int(row_minchi2.model_index)

            nu_value = row_minchi2.nu

            nu = QuantityData(
                value=nu_value,
                unit=nu_unit
            )
            lc_obs = lcs_obs.select_frequency(nu, unit=nu.unit)
            fnu_background = lc_obs.fnu_background.to_value_in(y_unit)

            # binned lightcurve
            df_binned_nu = dfs_binned_nu[nu_value]
            df_binned_plot: pd.DataFrame = df_binned_nu.loc[
                df_binned_nu[ColumnNames.MODEL_INDEX] == model_index
            ]
            df_binned_plot.attrs[ColumnNames.NU] = nu_value
            df_binned_plot.attrs[ColumnNames.MODEL_INDEX] = model_index
            binned_lc = clc.BinnedLightcurveWithBackground.from_df(df_binned_plot)


            # continuous
            df_continuous_nu = dfs_continuous_nu[nu_value]
            df_continuous_plot: pd.DataFrame = df_continuous_nu.loc[
                df_continuous_nu[ColumnNames.MODEL_INDEX] == model_index
            ]


            # observed lightcurve
            t_obs = lc_obs.t.to_value_in(x_unit)
            plotter.line(
                x=t_obs,
                y=fnu_background,
                style=components["background"].resolve_line_style(col)
            )
            plotter.errorbar(
                x=t_obs,
                y=lc_obs.fnu_total.to_value_in(y_unit),
                yerr=lc_obs.fnu_total_err.to_value_in(y_unit),
                style=components["observed"].resolve_errorbar_style(col)
            )

            # binned lightcurve
            plotter.step(
                edges=binned_lc.bin_edges.to_value_in(x_unit),
                values=binned_lc.fnu_with_bg.to_value_in(y_unit),
                style=components["binned"].resolve_line_style(col),
                baseline=fnu_background[0]
            )

            # continuous lightcurve
            continuous_lc = clc.ContinuousLightcurve(
                model_index=model_index,
                nu=nu,
                t=QuantityArray1D.from_df(
                    df_continuous_plot,
                    ColumnNames.T,
                    unit_key=MetadataNames.T_UNIT
                ),
                fnu_net=QuantityArray1D.from_df(
                    df_continuous_plot,
                    FluxName.FNU_TOTAL,
                    unit_key=MetadataNames.FNU_UNIT
                )
            )
            plotter.line(
                x=continuous_lc.t.to_value_in(x_unit),
                y=continuous_lc.fnu_with_bg(lc_obs.fnu_background.index(0)).to_value_in(y_unit),
                style=components["continuous"].resolve_line_style(col)
            )

            # model parameters
            minimum_chi2_model = model_parameters.select_models([model_index])
            metadata = PlotMetadata(
                a_wind=QuantityData.from_Quantity(minimum_chi2_model.a_wind.item()),
                beta_sh=QuantityData(minimum_chi2_model.beta_sh.item(),""),
                eps_b=QuantityData(minimum_chi2_model.eps_b,""),
                eps_th=QuantityData(minimum_chi2_model.eps_th,""),
                f_e=QuantityData(minimum_chi2_model.powerlaw.f_e,""),
                p_index=QuantityData(minimum_chi2_model.powerlaw.p_index,""),
                chi_squared=QuantityData(row_minchi2.chi_squared,""),
                mu_e_mu=QuantityData(minimum_chi2_model.molecular_weight.to_ratio(),""),
                distance=minimum_chi2_model.distance,
            )
            strings: list[str] = []

            # synchrotron components
            if model_parameters.synchrotron_model == "total":
                syn_model_str = "thermal+powerlaw"
            elif model_parameters.synchrotron_model == "thermal":
                syn_model_str = "thermal only"
            else:
                syn_model_str = model_parameters.synchrotron_model
            plotter.text(
                x=0.98,y=0.98,
                label=syn_model_str,
                style=components["synchrotron_component"].text_style_config,
                coordinates="axes"
            )

            for name, quantity in metadata.to_dict().items():
                plot_utils.MetadataComponent(
                    quantity=quantity,
                    config=metadata_config.items[name],
                ).add_strings(strings)

            plotter.title(
                strings=strings,
                fontsize=metadata_config.fontsize,
            )

        pdf.savefig(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
