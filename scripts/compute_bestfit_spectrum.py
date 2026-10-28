import argparse
from pathlib import Path
from typing import Any, Hashable, Literal, get_args
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import fields, dataclass
from matplotlib.backends.backend_pdf import PdfPages

from module.dataframe_utils import groupby_to_dict, select_maximum_row
from module.mydataclasses import FloatVectorSpace, LongformatDataFrame, QuantityArray1D, QuantityData
from module.observation import LongformatLightcurve
from module.utils import FileReader, FileWriter
from module.chevalier import BestfitModelPoint
from module.timing_utils import timed
from module.models import ModelParameters,ColumnNames,SynchrotronResults
from module.synchrotron_numerical_table import ThermalSynchrotronIntegrals

XM_SPACE = FloatVectorSpace(
    start=1.0e-02,
    stop =1.0e+04,
    num = 1024,
    scale="log"
)
XM = XM_SPACE.geomspace()
XM = XM[None,:]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "minimum_chi_squared",
        "model_parameter_table",
        "integral_table",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

times = [
    QuantityData(1.0,"min"),
    QuantityData(2.0,"min"),
    QuantityData(10.0,"min"),
]

@timed
def main() -> None:
    args = parse_args()
    df_bestfit = FileReader.table_from_csv(args.minimum_chi_squared)
    df_models = FileReader.table_from_csv(args.model_parameter_table)
    models = ModelParameters.from_df(df_models)
    integral = ThermalSynchrotronIntegrals.from_csv(args.integral_table)

    rows:list[dict[str,Any]] = df_bestfit.to_dict(orient="records")
    dfs:list[pd.DataFrame] = []
    for row in rows:
        for time in times:
            minchi2_index:int = int(row[ColumnNames.MODEL_INDEX])
            bestfit_model = models.select_models([minchi2_index])
            results = SynchrotronResults.compute_from(
                model_params=bestfit_model,
                xm=XM,
                table=integral
            )
            spectrum = results.to_spectrum(time)
            df = spectrum.to_df()
            dfs.append(df)
    df_out = pd.concat(dfs)
    FileWriter.df_to_csv(args.output,df_out)

if __name__ == "__main__":
    main()
