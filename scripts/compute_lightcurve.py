import argparse
from pathlib import Path
from module import compute_lightcurve

parser = argparse.ArgumentParser()

parser.add_argument(
    "--parameter_table",
    type=Path,
)
parser.add_argument(
    "--lightcurve_config",
    type=Path,
)
parser.add_argument(
    "--table_integral",
    type=Path,
)

args = parser.parse_args()
path_conf_lc:Path = args.lightcurve_config
path_table:Path = args.table_integral

def main(
    path_lcconf:Path,
    path_table: Path,
    inputs: compute_lightcurve.Input
):
    table = compute_lightcurve.ThermalSynchrotronTable.from_csv(path_table)
    model = compute_lightcurve.ThermalSynchrotron(table)
    conf = compute_lightcurve.ModelConfigure.from_yaml(path_conf_lc)
