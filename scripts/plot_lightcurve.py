from module import dataframe_processors, dataframe_utils
from module.inputs_as_dataclass import GeneralInputs
from dataclasses import dataclass
from module.mydataclasses import QuantityData, YAMLReadable
from module.utilities import filereaders as fr
from pathlib import Path
from module import lightcurve_plotter
import argparse
from module.plot import curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from module import plot_utils,observation
from module.observation import LongformatLightcurve, ObservedLightcurvePlotter
from module.calculation_lightcurve import CandidateModelFitResults, CandidateModelLightcurveContinuous, CandidateModelLightcurveTable, CandidateModelMetadata
from module.utils import FileReader

@dataclass(frozen=True,slots=True)
class LightcurveMetadataConfig:
    fontsize: float
    eps_b:  plot_utils.MetadataConfig
    eps_th: plot_utils.MetadataConfig
    mu_e_mu: plot_utils.MetadataConfig
    distance: plot_utils.MetadataConfig
    a_wind: plot_utils.MetadataConfig
    beta_sh: plot_utils.MetadataConfig
    nu: plot_utils.MetadataConfig

    def build_component(
        self,
        metadata_quantities: CandidateModelMetadata
    )->lightcurve_plotter.LightcurveMetadata:
        return lightcurve_plotter.LightcurveMetadata(
            fontsize=self.fontsize,
            eps_b=plot_utils.MetadataComponent(
                quantity=metadata_quantities.eps_b,
                config=self.eps_b
            ),
            eps_th=plot_utils.MetadataComponent(
                quantity=metadata_quantities.eps_th,
                config=self.eps_th
            ),
            mu_e_mu=plot_utils.MetadataComponent(
                quantity=metadata_quantities.mu_e_mu,
                config=self.mu_e_mu
            ),
            distance=plot_utils.MetadataComponent(
                quantity=metadata_quantities.distance,
                config=self.distance
            ),
            a_wind=plot_utils.MetadataComponent(
                quantity=metadata_quantities.a_wind,
                config=self.a_wind
            ),
            beta_sh=plot_utils.MetadataComponent(
                quantity=metadata_quantities.beta_sh,
                config=self.beta_sh
            ),
            nu=plot_utils.MetadataComponent(
                quantity=metadata_quantities.nu,
                config=self.nu
            ),
        )

@dataclass(frozen=True, slots=True)
class PlotConfig(YAMLReadable):
    layout: plot_utils.MultiPlotLayoutConfig
    styles: lightcurve_plotter.Styles
    metadata: LightcurveMetadataConfig

    @property
    def units(self)->tuple[str,str]:
        return (
            self.layout.axes.x.label.unit,
            self.layout.axes.y.label.unit
        )

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_binned_lightcurves",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_lightcurves",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--candidate_models",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--obslc",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--plotconfig",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    return parser.parse_args()

def main():
    args = parse_args()
    plot_all:bool = args.plot_all
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    conf = GeneralInputs.from_yaml(args.input)
    conf_plot = PlotConfig.from_yaml(args.plotconfig)
    t_unit,fnu_unit = conf_plot.units

    obslc_long = LongformatLightcurve.from_csv(args.obslc)

    candidate_lc_table = CandidateModelLightcurveTable.from_df_long(
        FileReader.table_from_csv(args.candidate_binned_lightcurves)
    )
    candidate_lcs_nu = candidate_lc_table.devide_into_frequency_groups()

    # dflong_cont_lcs = FileReader.table_from_csv(args.candidate_lightcurves)
    # dfs_cont_lcs = [
    #     df_cont_lcs
    #     for _,df_cont_lcs
    #     in dflong_cont_lcs.groupby("candidate_model_index")
    # ]
    # print(dfs_cont_lcs)
    # df_models = FileReader.table_from_csv(args.candidate_models)
    # n_models = len(cand_lcs.models)
    #
    # with PdfPages(outpath) as pdf:
    #     plot_layout = plot_utils.PlotLayout.create(
    #         layout_config=conf_plot.layout,
    #         n_plots=n_models
    #     )
    #     for i,plotter in enumerate(plot_layout.plotters):
    #
    #         cand_continuous_lc = CandidateModelLightcurve.from_df(
    #             candidate_model_index=i,
    #             df=dfs_cont_lcs[i]
    #         )
    #         lc = cand_lcs.models[i].lightcurve
    #         obslc = obslc_long.select_frequency(
    #             nu = lc.nu,
    #             unit = lc.nu.unit
    #         )
    #         model_metadata = CandidateModelMetadata.from_df(
    #             df_models=df_models,
    #             model_index = lc.model_index
    #         )
    #         metadata = conf_plot.metadata.build_component(
    #             metadata_quantities=model_metadata
    #         )
    #         lc_plotter = lightcurve_plotter.Lightcurve.from_data(
    #             obs_lc=obslc,
    #             binned_lc=lc,
    #             styles=conf_plot.styles,
    #             metadata=metadata,
    #             t_unit=t_unit,
    #             fnu_unit=fnu_unit
    #         )
    #         lc_plotter.plot(plotter)
    #         cand_continuous_lc.plot(
    #             plotter,
    #             style=conf_plot.styles.continuous,
    #             t_unit=t_unit,
    #             fnu_unit=fnu_unit,
    #             offset=obslc.fnu_background.to_value_in(fnu_unit)[0]
    #         )
    #     pdf.savefig(plot_layout.fig)
    #     plt.close(plot_layout.fig)
    return

if __name__ == "__main__":
    main()
