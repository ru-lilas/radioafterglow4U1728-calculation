# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import astropy.units as u
import numpy as np
from module import thermal
from module import synchrotron_scaling_values
from module import electron_temperature
from module.electron_temperature import calculate_theta_e
from module.electron_density import calculate_n_ele_for_strong_shock
from module.magnetic_field import magnetic_field
from module.n_upstream import calculate_n_wind
from module.radius import r_rad
from module.luminosity import calculate_lnu_th
from module.optical_depth import calculate_tau_nu,calculate_escape_fraction
from dataclasses import dataclass
from functools import cached_property
from module.utilities import bisection, unit_aliases as unit

@dataclass
class InputParameters:
    eps_th: float
    eps_B: float
    mu: float
    mu_e: float
    beta_sh: float
    a_wind_value: float
    a_wind_unit: str

@dataclass
class SynchrotronScalingValues:
    input:InputParameters
    nu_value: float
    nu_unit: str

    @cached_property
    def a_wind(self)->u.Quantity:
        return u.Quantity(self.input.a_wind_value,u.Unit(self.input.a_wind_unit))
    @cached_property
    def nu(self)->u.Quantity:
        return u.Quantity(self.nu_value,u.Unit(self.nu_unit))

    @cached_property
    def theta(self)->float:
        return electron_temperature.calculate_theta_e(
            eps_th=self.input.eps_th,
            mu=self.input.mu,
            mu_e=self.input.mu_e,
            beta_sh=self.input.beta_sh
        )

    @cached_property
    def phi_theta(self):
        return synchrotron_scaling_values.calculate_phi_theta(
            theta=self.theta,
            eps_B=self.input.eps_B,
            a_wind=self.a_wind
        )

    @cached_property
    def t_theta(self):
        return synchrotron_scaling_values.calculate_t_theta(
            phi_theta=self.phi_theta,
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
            a_wind = self.a_wind,
            t_theta=self.t_theta
        )

    @cached_property
    def b_mag_theta(self):
        return synchrotron_scaling_values.calculate_b_mag_theta(
            eps_B=self.input.eps_B,
            a_wind=self.a_wind,
            t_theta=self.t_theta
        )

    @cached_property
    def j_theta(self):
        return synchrotron_scaling_values.calculate_j_theta(self.theta,self.n_e_theta,self.b_mag_theta)

    @cached_property
    def alpha_theta(self):
        return synchrotron_scaling_values.calculate_alpha_theta(self.theta,self.n_e_theta,self.b_mag_theta)

    @cached_property
    def l_theta(self):
        return synchrotron_scaling_values.calculate_l_theta(
            r_theta=self.r_theta,
            j_theta=self.j_theta,
            alpha_theta=self.alpha_theta
        )

    @cached_property
    def tau_theta(self):
        return synchrotron_scaling_values.calculate_tau_theta(
            alpha_theta=self.alpha_theta,
            r_theta=self.r_theta
        )

    @cached_property
    def xi_peak(self):
        tau_theta = self.tau_theta[0]
        def f(xi:float):
            return synchrotron_scaling_values.func_ssa_peak(tau_theta,xi)
        xi_peak = bisection.bisection(f,1.0e-01,1.0e+04) 
        return np.asarray(xi_peak,dtype=np.float64)

    @cached_property
    def lnu_peak_dimless(self):
        lnu_peak_dimless = synchrotron_scaling_values.calculate_lnu_xi_dimless(
            tau_theta=self.tau_theta,
            xi=self.xi_peak
        )
        return lnu_peak_dimless

    @cached_property
    def phi_peak(self):
        phi_peak = u.Quantity(self.xi_peak * self.phi_theta)
        return phi_peak

    @cached_property
    def t_peak(self)->u.Quantity:
        return u.Quantity(self.phi_peak/self.nu).to(self.t_theta.unit)


@dataclass
class Frequency:
    value_array: np.ndarray
    unit: str

@dataclass
class SynchrotronSpectrum:
    inputparams: InputParameters
    t_value: float
    t_unit: str
    nu:Frequency

    @cached_property
    def a_wind_quantity(self):
        return u.Quantity(self.inputparams.a_wind_value,u.Unit(self.inputparams.a_wind_unit))

    @property
    def t(self):
        return u.Quantity(self.t_value,u.Unit(self.t_unit))

    @cached_property
    def nu_array_quantity(self):
        return u.Quantity(self.nu.value_array, u.Unit(self.nu.unit))

    @cached_property
    def theta_e(self)->float:
        return calculate_theta_e(
            eps_th=self.inputparams.eps_th,
            mu=self.inputparams.mu,
            mu_e=self.inputparams.mu_e,
            beta_sh=self.inputparams.beta_sh
        )

    @property
    def r(self)->u.Quantity:
        return r_rad(
            t=self.t,
            beta_sh=self.inputparams.beta_sh
        )

    @property
    def n_wind(self)->u.Quantity:
        return calculate_n_wind(
            a_wind=self.a_wind_quantity,
            r=self.r,
            mu=self.inputparams.mu
        )

    @property
    def n_ele(self)->u.Quantity:
        return calculate_n_ele_for_strong_shock(
            n_upstream=self.n_wind,
            mu_e=self.inputparams.mu_e
        )

    @property
    def b_mag(self):
        return magnetic_field(
            n_us=self.n_wind,
            beta_sh=self.inputparams.beta_sh,
            eps_B=self.inputparams.eps_B,
            mu=self.inputparams.mu
        )

    @cached_property
    def omega(self):
        return synchrotron_scaling_values.convert_nu_into_omega(self.nu_array_quantity)

    @property
    def omega_B(self):
        return synchrotron_scaling_values.omega_b(self.b_mag)

    @property
    def nu_B(self):
        return synchrotron_scaling_values.convert_omega_into_nu(omega=self.omega_B)

    @property
    def chi(self):
        return synchrotron_scaling_values.chi_gyro(omega=self.omega, omega_b=self.omega_B)

    @property
    def nu_crit(self):
        return synchrotron_scaling_values.calculate_nu_crit(nu_B=self.nu_B, theta=self.theta_e)

    @property
    def xm(self)->np.ndarray:
        return thermal.convert_xm(theta=self.theta_e, chi=self.chi)

    #---normalization quantities---
    @property
    def P0(self)->u.Quantity:
        return synchrotron_scaling_values.calculate_P0(self.b_mag)

    @property
    def j0(self)->u.Quantity:
        return synchrotron_scaling_values.calculate_j0(
            nu_B=self.nu_B,
            n_ele=self.n_ele
        )

    @property
    def a0(self)->u.Quantity:
        return synchrotron_scaling_values.calculate_alpha0(
            nu_B=self.nu_B,
            n_ele=self.n_ele,
            pnu0=self.P0
        )

    @property
    def S0(self)->u.Quantity:
        return self.j0/self.a0

    #---frequency dependence terms---
    @property
    def j_nu_th_dimless(self)->np.ndarray:
        return thermal.j_th(
            theta=self.theta_e,
            chi=self.chi
        )

    @property
    def a_nu_th_dimless(self)->np.ndarray:
        return thermal.anu_th_dimless(
            theta=self.theta_e,
            chi=self.chi,
        )

    @property
    def S_nu_th_dimless(self)->np.ndarray:
        value = self.j_nu_th_dimless/self.a_nu_th_dimless
        return value

    #---
    @property
    def j_nu_th(self)->u.Quantity:
        return self.j0*self.j_nu_th_dimless

    @property
    def a_nu_th(self)->u.Quantity:
        return self.a0*self.a_nu_th_dimless

    @property
    def S_nu_th(self)->u.Quantity:
        return self.S0*self.S_nu_th_dimless

    @property
    def tau_nu(self):
        return calculate_tau_nu(
            alpha_nu=self.a_nu_th,
            r=self.r
        )

    @property
    def f_esc(self)->np.ndarray:
        """ escape fraction
        """
        return calculate_escape_fraction(tau_nu=self.tau_nu)

    @property
    def lnu_th(self)->np.ndarray:
        lnu_th_quantity = calculate_lnu_th(snu_th=self.S_nu_th,f_esc=self.f_esc,r=self.r)
        return np.asarray(lnu_th_quantity.to_value(unit.specific_luminosity))
