# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from module import synchrotron_scaling_values
from module import electron_temperature
from dataclasses import dataclass
from functools import cached_property
from module.tabular import ThermalSynchrotronTable
from module.utilities import bisection, unit_aliases as unit
from module import calculate_lnu_th

@dataclass
class InputParameters:
    eps_th: float
    eps_B: float
    mu: float
    mu_e: float
    beta_sh: float
    a_wind_value: float
    a_wind_unit: str

    @cached_property
    def a_wind_quantity(self)->u.Quantity:
        return u.Quantity(self.a_wind_value,u.Unit(self.a_wind_unit))

    @cached_property
    def theta(self)->float:
        return electron_temperature.calculate_theta_e(
            eps_th=self.eps_th,
            mu=self.mu,
            mu_e=self.mu_e,
            beta_sh=self.beta_sh
        )

    @cached_property
    def phi_theta(self):
        return synchrotron_scaling_values.calculate_phi_theta(
            theta=self.theta,
            eps_B=self.eps_B,
            a_wind=self.a_wind_quantity
        )
    
    @cached_property
    def tau_theta(self):
        return synchrotron_scaling_values.calculate_tau_theta(
            eps_B=self.eps_B,
            mu_e=self.mu_e,
            mu=self.mu,
            theta=self.theta,
            a_wind=self.a_wind_quantity,
            beta_sh=self.beta_sh
        )

    @cached_property
    def l_theta(self):
        return synchrotron_scaling_values.calculate_l_theta(
            beta_sh=self.beta_sh,
            eps_B=self.eps_B,
            theta=self.theta,
            a_wind=self.a_wind_quantity,
        )

@dataclass
class ThermalSynchrotronScalingValues:
    input:InputParameters
    nu_value: NDArray[np.float64]
    nu_unit: str
    table: ThermalSynchrotronTable

    @cached_property
    def nu(self)->u.Quantity:
        return u.Quantity(self.nu_value,u.Unit(self.nu_unit))

    @cached_property
    def t_theta(self):
        return synchrotron_scaling_values.calculate_t_theta(
            phi_theta=self.input.phi_theta,
            nu=self.nu
        )

    @cached_property
    def r_theta(self):
        return synchrotron_scaling_values.calculate_r_theta(
            beta_sh=self.input.beta_sh,
            t_theta=self.t_theta
        )

    @cached_property
    def n_e_theta(self):
        return synchrotron_scaling_values.calculate_n_ele_theta(
            beta_sh=self.input.beta_sh,
            mu=self.input.mu,
            mu_e=self.input.mu_e,
            a_wind = self.input.a_wind_quantity,
            t_theta=self.t_theta
        )

    @cached_property
    def b_mag_theta(self):
        return synchrotron_scaling_values.calculate_b_mag_theta(
            eps_B=self.input.eps_B,
            a_wind=self.input.a_wind_quantity,
            t_theta=self.t_theta
        )

    @cached_property
    def j_theta(self):
        return synchrotron_scaling_values.calculate_j_theta(self.input.theta,self.n_e_theta,self.b_mag_theta)

    @cached_property
    def alpha_theta(self):
        return synchrotron_scaling_values.calculate_alpha_theta(self.input.theta,self.n_e_theta,self.b_mag_theta)

    @cached_property
    def tau_theta(self):
        return self.input.tau_theta

    @cached_property
    def xi_est(self):
        tau_theta = self.input.tau_theta
        def f(xi:NDArray[np.float64]):
            return synchrotron_scaling_values.calculate_ln_tau(xi,tau_theta,self.table)
        xi_peak = bisection.bisection(f,1.0e-01,1.0e+04)
        return np.asarray(xi_peak,dtype=np.float64)

    @cached_property
    def lnu_est_dimless(self):
        lnu_peak_dimless = synchrotron_scaling_values.calculate_lambda_using_table(
            tau_theta=self.input.tau_theta,
            xm=self.xi_est,
            table=self.table
        )
        return lnu_peak_dimless

    @cached_property
    def phi_peak(self):
        phi_peak = u.Quantity(self.xi_est * self.input.phi_theta)
        return phi_peak

    @cached_property
    def t_peak(self)->u.Quantity:
        return u.Quantity(self.phi_peak/self.nu).to(self.t_theta.unit)

    @cached_property
    def lnu_peak(self):
        lnu_peak = u.Quantity(self.l_theta*self.lnu_est_dimless,self.l_theta.unit)
        return lnu_peak

@dataclass
class Frequency:
    value_array: np.ndarray
    unit: str

@dataclass
class SynchrotronSpectrum:
    inputs: InputParameters
    t_value: float
    t_unit: str
    nu_value: NDArray[np.float64]
    nu_unit: str
    tabular: ThermalSynchrotronTable

    @cached_property
    def scalings(self):
        return ThermalSynchrotronScalingValues(
            input=self.inputs,
            nu_value=self.nu_value,
            nu_unit=self.nu_unit,
            table=self.tabular
        )

    @property
    def t_quantity(self):
        return u.Quantity(self.t_value,u.Unit(self.t_unit))

    @property
    def phi(self)->u.Quantity:
        nu = self.scalings.nu
        return nu*self.t_quantity

    @property
    def xi(self):
        xi = np.asarray((self.phi/self.inputs.phi_theta).to_value(unit.dimensionless),dtype=np.float64)
        return xi

    @property
    def ln_tau(self):
        return synchrotron_scaling_values.calculate_ln_tau(
            xi=self.xi,
            tau_theta=self.scalings.tau_theta,
            table=self.tabular
        )

    @property
    def lnu_th_dimless(self):
        return calculate_lnu_th.dimless_tabular(
            x=self.xi,
            log_tau=self.ln_tau
        )

    @property
    def lnu_th(self):
        q = u.Quantity((self.inputs.l_theta * self.lnu_th_dimless).to(self.scalings.l_theta.unit))
        return q
