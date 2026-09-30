"""許容モデルの中で、周波数ごとにchi-squaredが最小のものを見つける"""

import argparse
from pathlib import Path

import pandas as pd

from module.dataframe_utils import extract_minimum_row
from module.models import ChiSquaredParameters, ColumnNames, MetadataNames
from module.mydataclasses import LongformatDataFrame
from module.utils import FileReader, FileWriter

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("input", "chi_squared", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    longdf_chi2 = FileReader.table_from_csv(args.chi_squared)
    chi2_result_nu = LongformatDataFrame(longdf_chi2).parse(ColumnNames.NU)

    results: list[pd.Series] = []
    for nu_value, df_nu in chi2_result_nu.items():
        if not isinstance(nu_value, float):
            raise TypeError(
                f"nu_valueのtypeが不正です:{type(nu_value)}"
            )
        row = extract_minimum_row(df_nu, ChiSquaredParameters.CHI_SQUARED)
        row[ColumnNames.NU] = nu_value
        results.append(row)
    df_results = pd.DataFrame(results)
    df_results.attrs[MetadataNames.NU_UNIT] = longdf_chi2.attrs[MetadataNames.NU_UNIT]

    FileWriter.df_to_csv(
        path=args.output,
        df=df_results
    )

if __name__ == "__main__":
    main()
