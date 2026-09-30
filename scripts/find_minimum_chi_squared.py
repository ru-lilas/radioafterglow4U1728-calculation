"""
    find_minimu_chi2_model.py
    許容モデルの中でchi-squaredが最小となるものを見つける
"""
import argparse
import pandas as pd
from pathlib import Path
from module import calculation_lightcurve as clc
from module.models import ChiSquaredParameters, ColumnNames, DefaultUnits, MetadataNames, ModelParameters, SynchrotronResults
from module.mydataclasses import LongformatDataFrame, QuantityData, WideFormatDataFrames
from module.utils import FileReader, FileWriter
from module.dataframe_utils import extract_minimum_row

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--chi_squared",
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

    longdf_chi2 = FileReader.table_from_csv(args.chi_squared)
    chi2_result = LongformatDataFrame(longdf_chi2)
    chi2_result_nu = chi2_result.parse(ColumnNames.NU)

    results = []
    for nu_value,df_nu in chi2_result_nu.items():
        nu_unit=df_nu.attrs[MetadataNames.NU_UNIT]
        if not isinstance(nu_value,float):
            raise TypeError(
                f"nu_valueのtypeが不正です:{type(nu_value)}"
            )
        nu = QuantityData(
            value=nu_value,
            unit=nu_unit
        )
        row = extract_minimum_row(df_nu,ChiSquaredParameters.CHI_SQUARED)
        row[ColumnNames.NU] = nu_value
        results.append(row)
    df_results = pd.DataFrame(results)
    df_results.attrs[MetadataNames.NU_UNIT] = longdf_chi2.attrs[MetadataNames.NU_UNIT]

    # df_long = FileReader.table_from_csv(args.candidate_binned_lightcurves)
    #
    # candidate_lc_table = clc.CandidateModelLightcurveTable.from_df_long(df_long)
    # candidate_lcs_nu = candidate_lc_table.devide_into_frequency_groups()
    # dfs = [
    #     lc_nu.to_df_minimum_chi2()
    #     for lc_nu in candidate_lcs_nu
    # ]
    # dfs_wide = WideFormatDataFrames(dfs)
    # df_long_minchi2 = dfs_wide.to_long_format()

    FileWriter.df_to_csv(
        path=args.output,
        df=df_results
    )

if __name__ == "__main__":
    main()
