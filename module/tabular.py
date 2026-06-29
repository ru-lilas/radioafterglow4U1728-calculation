from functools import cached_property
from pathlib import Path

from numpy.typing import NDArray
from module.utilities import filereaders as fr
import numpy as np
import pandas as pd
from dataclasses import dataclass

def read_tabular(input:Path):
    tabular = fr.read_csv(input)
    return tabular

@dataclass
class ThermalSynchrotronTable:
    tabular: pd.DataFrame

    @cached_property
    def xi_tabular(self):
        return np.array(self.tabular["xi"], dtype=np.float64)

    @cached_property
    def log_ip_tabular(self):
        return np.array(self.tabular["ln_ip"], dtype=np.float64)

    @cached_property
    def xi_min(self):
        return self.xi_tabular[0]

    @cached_property
    def xi_max(self):
        return self.xi_tabular[-1]

    def calculate_log_ip(self,xi:NDArray[np.float64])->NDArray[np.float64]:
        return np.interp(xi, self.xi_tabular, self.log_ip_tabular)

@dataclass
class TauThetaTable:
    table: pd.DataFrame

    @cached_property
    def tabular_tau_theta(self):
        return np.array(self.table["tau_theta"], dtype=np.float64)

    @cached_property
    def tabular_lambda_peak(self):
        return np.array(self.table["lambda_peak"], dtype=np.float64)

    @cached_property
    def tabular_xm_peak(self):
        return np.array(self.table["xm_peak"], dtype=np.float64)

    @cached_property
    def tau_theta_min(self):
        return self.tabular_tau_theta[0]

    @cached_property
    def tau_theta_max(self):
        return self.tabular_tau_theta[-1]

    def can_interp(self,tau_theta_arr:NDArray[np.float64]):
        if np.any(tau_theta_arr < self.tau_theta_min) or np.any(tau_theta_arr > self.tau_theta_max):
            raise ValueError(
                f"x={tau_theta_arr} is outside interpolation range [{self.tau_theta_min}, {self.tau_theta_max}]"
            )
        else:
            return

    def calculate_xm_peak(
        self,
        tau_theta_arr:NDArray[np.float64]
    )->NDArray[np.float64]:
        self.can_interp(tau_theta_arr)
        return np.interp(
            tau_theta_arr,
            self.tabular_tau_theta,
            self.tabular_xm_peak
        )

    def calculate_lambda_peak(
        self,
        tau_theta_arr:NDArray[np.float64]
    )->NDArray[np.float64]:
        self.can_interp(tau_theta_arr)
        return np.interp(
            tau_theta_arr,
            self.tabular_tau_theta,
            self.tabular_lambda_peak
        )
