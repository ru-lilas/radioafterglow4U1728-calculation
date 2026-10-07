import argparse
from pathlib import Path
from typing import Any, Hashable, Literal, get_args
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import fields, dataclass
from matplotlib.backends.backend_pdf import PdfPages

from module import calculation_lightcurve as clc
from module import plot_utils
from module.dataframe_utils import groupby_to_dict, select_maximum_row
from module.models import ColumnNames, FluxName, MetadataNames, ModelParameters, SwitchKeys
from module.mydataclasses import LongformatDataFrame, QuantityArray1D, QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.chevalier import BestfitModelPoint

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
        "minimum_chi_squared",
        "binned_lightcurves",
        "model_parameter_table",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    df_minchi2 = FileReader.table_from_csv(args.minimum_chi_squared)
    df_model_parameter = FileReader.table_from_csv(args.model_parameter_table)
    longdf_binned = FileReader.table_from_csv(args.binned_lightcurves)
    dfgroup_binned_nu = groupby_to_dict(longdf_binned,ColumnNames.NU)

    model_parameters = ModelParameters.from_df(df_model_parameter)

    rows:list[dict[str,Any]] = df_minchi2.to_dict(orient="records")
    results:list[BestfitModelPoint] = []
    for row in rows:
        minchi2_index:int = row[ColumnNames.MODEL_INDEX]
        nu_value = float(row[ColumnNames.NU])
        longdf_binned_nu = dfgroup_binned_nu[nu_value]
        df_binned = longdf_binned[longdf_binned[ColumnNames.MODEL_INDEX]==minchi2_index]
        if not isinstance(df_binned,pd.DataFrame):
            raise TypeError(
                "df_binnedをDataFrameとして取得できませんでした"
            )
        results.append(BestfitModelPoint.from_df(df_binned))

    df = pd.concat([
        result.to_df() for result in results
    ],axis="index")

    FileWriter.df_to_csv(args.output,df)

    return

if __name__ == "__main__":
    main()
