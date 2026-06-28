import argparse
from pathlib import Path
from typing import cast
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module import arrange_observation_data
from module import dataframe_processors as processor
import pandas as pd
from dataclasses import dataclass, asdict

@dataclass
class UnitData:
    t_unit: str
    fnu_unit: str
    nu_unit: str
    phi_unit: str

parser = argparse.ArgumentParser()
parser.add_argument(
    "input",
    type=Path,
)
parser.add_argument(
    "--output",
    type=Path,
    required=True
)
args = parser.parse_args()

outpath:Path = args.output
outpath.parent.mkdir(parents=True,exist_ok=True)

inpath:Path = args.input
metadata_obs = fr.read_keyvalue(inpath)
units = UnitData(
    **metadata_obs,
    phi_unit = "GHz s"
)
df_lc = fr.read_csv(inpath)

output_data:list[dict] = []
for nu,df in df_lc.groupby("nu", sort=False):
    df = df.reset_index(drop=True)
    nu = cast(float,nu)
    print(df)
    t_peak, fnu_peak = processor.extract_peak_quadratic(
        df=df,
        column_x="t",
        column_y="fnu_net",
        column_yerr="fnu_err",
        n_sample=4,
        n_margin=1
    )
    phi_peak = arrange_observation_data.calculate_phi_peak(
        nu_value = nu,
        nu_unit = units.nu_unit,
        t_peak_value=t_peak,
        t_unit = units.t_unit,
        phi_unit = units.phi_unit
    )
    output_data.append({
        "nu": nu,
        "t_peak": t_peak,
        "fnu_net_peak": fnu_peak,
        "phi_peak": phi_peak
    })
df_output = pd.DataFrame(output_data)

print(df_output)
fw.write_csv_with_params(
    df_output,
    asdict(units),
    outpath
)
