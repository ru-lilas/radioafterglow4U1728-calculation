# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import astropy.units as u
import numpy as np
from module import thermal
from module import synchrotron_scaling_values
from module.electron_temperature import calculate_theta_e
from module.electron_density import calculate_n_ele_for_strong_shock
from module.magnetic_field import magnetic_field
from module.n_upstream import calculate_n_wind
from module.radius import r_rad
from module.luminosity import calculate_lnu_th
from module.optical_depth import calculate_tau_nu,calculate_escape_fraction
from dataclasses import dataclass

@dataclass
class SynchrotronSpectrum:
    eps_th: float
    eps_B: float
    mu: float
    mu_e: float
    beta_sh: float
    a_wind_value: float
    a_wind_unit: str
    t_value: float
    t_unit: str
    nu_array_value: float
    nu_array_unit: str

    @property
    def a_wind_quantity(self):
        return u.Quantity(self.a_wind_value,u.Unit(self.a_wind_unit))

    @property
    def t(self):
        return u.Quantity(self.t_value,u.Unit(self.t_unit))

    @property
    def nu_array_quantity(self):
        return u.Quantity(self.nu_array_value, u.Unit(self.nu_array_unit))

    @property
    def theta_e(self)->float:
        return calculate_theta_e(
            eps_th=self.eps_th,
            mu=self.mu,
            mu_e=self.mu_e,
            beta_sh=self.beta_sh
        )

    @property
    def r(self)->u.Quantity:
        return r_rad(
            t=self.t,
            beta_sh=self.beta_sh
        )

    @property
    def n_wind(self)->u.Quantity:
        return calculate_n_wind(
            a_wind=self.a_wind_quantity,
            r=self.r,
            mu=self.mu
        )

    @property
    def n_ele(self)->u.Quantity:
        return calculate_n_ele_for_strong_shock(
            n_upstream=self.n_wind,
            mu_e=self.mu_e
        )

    @property
    def b_mag(self):
        return magnetic_field(
            n_us=self.n_wind,
            beta_sh=self.beta_sh,
            eps_B=self.eps_B,
            mu=self.mu
        )

    @property
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
    def f_esc(self):
        """ escape fraction
        """
        return calculate_escape_fraction(tau_nu=self.tau_nu)

    @property
    def lnu_th(self):
        return calculate_lnu_th(snu_th=self.S_nu_th,f_esc=self.f_esc,r=self.r)
