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

    m_acc = obs_property.m_acc
    e_nuc = obs_property.e_nuc_max
    m_ej_min = params.m_swept(nu)
    e_kin_min = params.e_kin_min(nu)

    print(f"M_acc = {m_acc.quantity.to('g'):.2e}")
    print(f"M_ej_min = {m_ej_min.quantity.to('g'):.2e}")
    print(f"E_nuc_max = {e_nuc.quantity.to('erg'):.2e}")
    print(f"E_kin_min = {e_kin_min.quantity.to('erg'):.2e}")


if __name__ == "__main__":
    main()
