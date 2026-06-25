import argparse
from functools import cached_property
from typing import Any, cast
from module import fetch_numerical_table, quantity_converter
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from module.tabular import ThermalSynchrotronTable
from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from module.utilities import build_nparray
import pandas as pd
import astropy.units as u
from dataclasses import dataclass
from module.synchrotron_scaling_values import calculate_lambda_using_table

@dataclass
class InputArrays:
    t_value_arr: NDArray[np.float64]
    t_unit: u.Unit
    phi_theta_values: NDArray[np.float64]
    phi_unit: u.Unit
    nu_values: NDArray[np.float64]
    nu_unit: u.Unit
    l_theta_values: NDArray[np.float64]
    l_unit: u.Unit
    tau_theta: NDArray[np.float64]
    table:ThermalSynchrotronTable

    @cached_property
    def t_quantity_arr(self):
        return u.Quantity(self.t_value_arr,self.t_unit)

    @cached_property
    def phi_theta_quantites(self):
        return u.Quantity(self.phi_theta_values,self.phi_unit)

    @cached_property
    def nu_quantities(self):
        return u.Quantity(self.nu_values,self.nu_unit)

    @cached_property
    def l_quantities(self):
        return u.Quantity(self.l_theta_values,self.l_unit)

    @cached_property
    def phi_arr(self):
        phi = u.Quantity(self.t_quantity_arr[None,:]*self.nu_quantities[:,None])
        return u.Quantity(phi.to(self.phi_unit))

    @cached_property
    def xm_arr(self):
        return self.phi_arr/self.phi_theta_quantites[:,None]

    @cached_property
    def lambda_arr(self):
        return calculate_lambda_using_table(
            xm=self.xm_arr,
            tau_theta=self.tau_theta,
            table=self.table
        )

    @cached_property
    def lnu_arr(self):
        return u.Quantity(self.lambda_arr*self.l_quantities[:,None])

def main(args:argparse.Namespace):
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    table = fetch_numerical_table.fetch_numerical_table(args)

    confpath:Path = args.config
    calculation_input = fr.read_yaml(confpath)

    inpath:Path = args.input
    df_estimated = fr.read_csv(inpath)
    metadata_estimated = fr.read_keyvalue(inpath)

    inputarrs = InputArrays(
        t_value_arr= build_nparray.log(calculation_input["t_value_arr"]),
        t_unit = u.Unit(calculation_input["t_unit"]),
        phi_theta_values = np.asarray(df_estimated["phi_theta_value"],dtype=np.float64),
        phi_unit = u.Unit(metadata_estimated["phi_unit"]),
        nu_values = np.asarray(df_estimated["nu_value"],dtype=np.float64),
        nu_unit = u.Unit(metadata_estimated["nu_unit"]),
        l_theta_values= np.asarray(df_estimated["l_theta_value"],dtype=np.float64),
        l_unit = u.Unit(metadata_estimated["l_unit"]),
        tau_theta = np.asarray(df_estimated["tau_theta"],dtype=np.float64),
        table=table
    )
    d_value = float(metadata_estimated["d_value"])
    d_unit = u.Unit(metadata_estimated["d_unit"])
    d_quantity = u.Quantity(d_value,d_unit)
    fnu = quantity_converter.lnu_into_fnu(
        lnu=inputarrs.lnu_arr,
        distance=d_quantity
    )
    fnu_unit = u.Unit(calculation_input["fnu_unit"])

    metadata:dict[str,Any] = {
        "t_unit": inputarrs.t_unit.to_string(),
        "lnu_unit": inputarrs.l_unit.to_string(),
        "nu_unit": inputarrs.nu_unit.to_string(),
        "fnu_unit": fnu_unit.to_string(),
        "d_value": d_value,
        "d_unit": d_unit.to_string(),
        "eps_B": metadata_estimated["eps_B"],
        "eps_th": metadata_estimated["eps_th"],
        "mu": metadata_estimated["mu"],
        "mu_e": metadata_estimated["mu_e"],
        "a_wind_unit": metadata_estimated["a_wind_unit"]
    }
    dfs: list[pd.DataFrame] = []

    for i, nu in enumerate(inputarrs.nu_values):
        lnu_arr_nu = cast(u.Quantity,inputarrs.lnu_arr[i])
        fnu_arr_nu = cast(u.Quantity,fnu[i])
        df_lc = pd.DataFrame({
            "t_value": inputarrs.t_value_arr,
            "lnu_value": lnu_arr_nu.to_value(inputarrs.l_unit),
            "fnu_value": fnu_arr_nu.to_value(fnu_unit),
            "nu_value": nu,
        })

        dfs.append(df_lc)
    df = pd.concat(
        dfs,
        ignore_index=True,
    )
    fw.write_csv_with_params(df,metadata,outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--table",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(args)
