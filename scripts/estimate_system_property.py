"""
    estimate_system_property.py
    得られた物理パラメータから対象の物理量を推定する
"""

import argparse
from pathlib import Path

from module import compute_lightcurve as lc
from module import mystatistics
from module.burster_properties import BursterParameters, ObservedBursterProperty
from module.inputs_as_dataclass import GeneralInputs
from module.mystatistics import CalculateChiSquared
from module.observation import LongformatLightcurve
from module.utils import FileWriter
from module.models import ThermalSynchrotronTable

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--estimated",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    return parser.parse_args()

def main():
    args = parse_args()
    outpath:Path = args.output

    est_lc = lc.BinnedBestfitLightcurve.from_csv(args.estimated)
    nu = est_lc.nu
    general_inputs = GeneralInputs.from_yaml(args.input)
    obs_property = general_inputs.burster_property
    params = BursterParameters.from_chevalier(est_lc.params)

    m_acc = obs_property.m_acc.value_in("g")
    e_nuc = obs_property.e_nuc_max.value_in("erg")

    print(m_acc)
    print(e_nuc)

if __name__ == "__main__":
    main()
