import astropy.units as u
from module.electron_temperature import calculate_theta_e
from module import synchrotron_scaling_values
from module.magnetic_field import magnetic_field
from module import magnetic_field
from module.models import InputParameters
import numpy as np

def t_peak(inputparams:InputParameters,nu:u.Quantity):
    theta = calculate_theta_e(
        eps_th=inputparams.eps_th,
        mu=inputparams.mu,
        mu_e=inputparams.mu_e,
        beta_sh=inputparams.beta_sh
    )
    a_wind = u.Quantity(inputparams.a_wind_value,u.Unit(inputparams.a_wind_unit))
    return synchrotron_scaling_values.calculate_t_peak_nu(
        eps_B=inputparams.eps_B,
        theta=theta,
        a_wind=a_wind,
        nu = nu
    )

def nu_crit(inputparams:InputParameters,t:u.Quantity):
    theta = calculate_theta_e(
        eps_th=inputparams.eps_th,
        mu=inputparams.mu,
        mu_e=inputparams.mu_e,
        beta_sh=inputparams.beta_sh
    )
    a_wind = u.Quantity(inputparams.a_wind_value,u.Unit(inputparams.a_wind_unit))
    b_mag = magnetic_field.wind_profile(
        eps_B=inputparams.eps_B,
        a_wind=a_wind,
        t=t
    )
    nu_b = synchrotron_scaling_values.calculate_nu_b(b_mag)
    return synchrotron_scaling_values.calculate_nu_crit(nu_b,theta)

def calculate_phi_theta(inputparams:InputParameters):
    theta = calculate_theta_e(
        eps_th=inputparams.eps_th,
        mu=inputparams.mu,
        mu_e=inputparams.mu_e,
        beta_sh=inputparams.beta_sh
    )
    a_wind = u.Quantity(inputparams.a_wind_value,u.Unit(inputparams.a_wind_unit))
    return synchrotron_scaling_values.calculate_phi_theta(
        theta=theta,
        eps_B=inputparams.eps_B,
        a_wind=a_wind
    )

def calculate_tau_theta(inputparams:InputParameters):
    theta = calculate_theta_e(
        eps_th=inputparams.eps_th,
        mu=inputparams.mu,
        mu_e=inputparams.mu_e,
        beta_sh=inputparams.beta_sh
    )
    a_wind = u.Quantity(inputparams.a_wind_value,u.Unit(inputparams.a_wind_unit))
    return synchrotron_scaling_values.calculate_tau_theta(
        eps_B=inputparams.eps_B,
        theta = theta,
        mu=inputparams.mu,
        mu_e=inputparams.mu_e,
        beta_sh=inputparams.beta_sh,
        a_wind=a_wind
    )

params = InputParameters(
    eps_th=1.0,
    eps_B=0.1,
    mu=0.62,
    mu_e = 1.18,
    beta_sh= 0.1,
    a_wind_value=1.0e+07,
    a_wind_unit="g/cm"
)

nu_array = u.Quantity(np.logspace(6,12,128),u.Unit("Hz"))
phi_theta = calculate_phi_theta(params)
tau_theta = calculate_tau_theta(params)
print(params)
print(f"phi = {phi_theta:.2e}")
print(f"tau = {tau_theta:.2e}")
