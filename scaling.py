from numpy import asarray
import numpy
from module.mydataclasses import QuantityArray1D, QuantityData
from module.parameter_table import ModelParameterTable
from module import models

eps_b = 1.0e-02
eps_th = 0.40
a_wind= QuantityArray1D(
    values=asarray([1.0e+07]),
    unit="g cm-1"
)
a_unit = "g cm-1"
phi_unit = "GHz min"
a_syn = models.WindParameterNormalization.synchrotron(
    theta=asarray([3.0])
)
theta_sample=models.ElectronTemperature(asarray([3.0]))
print(f"A_syn = {a_syn.to_value_in(a_unit).item():.2e} {a_unit}")
phi_a = models.NormalizedWindParameter.from_a_wind(
    a_wind = a_wind,
    a_norm = a_syn
)
phi_a_sample=models.NormalizedWindParameter(
    values=asarray([2.0]),
    unit="GHz min"
)
print(f"phi_A = {phi_a.to_value_in(phi_unit).item():.2e} {phi_unit}")
phi_th = models.ThermalSynchrotronPhase.from_phiA_syn(
    eps_b=eps_b,
    phi_a=phi_a_sample
)
print(f"phi_th = {phi_th.to_value_in(phi_unit).item():.2e} {phi_unit}")
mep = models.calculate_effective_mass_ratio(mu=0.5,mu_e=1.0)
tau_theta = models.OpticalDepthParameter.from_params_exact(
    mep=mep,
    eps_b=eps_b,
    theta=theta_sample,
    phi_a=phi_a_sample
)
print(f"tau_theta = {tau_theta.values.item():.2e}")
lnu_unit = "erg s-1 Hz-1"
lnu_th = models.ThermalSynchrotronLuminosity.from_params(
    mep=mep,
    eps_th=eps_th,
    eps_b=eps_b,
    theta=theta_sample,
    phi_a=phi_a_sample
)
print(f"lnu_theta = {lnu_th.to_value_in(lnu_unit).item():.2e} {lnu_unit}")
theta0 = models.ElectronTemperatureScaling(values=asarray(9.0))
beta_sh = theta0.to_beta(eps_th=eps_th,mep=mep)
print(theta0.values)
print(beta_sh)

def calculate_thermally_thin_theta(
    mep:float,
    phi_theta:models.ThermalSynchrotronPhase,
    epsb:float
):
    phi_theta_val = phi_theta.decompose()
    log_phi = numpy.log10(phi_theta_val)
    log_mep = numpy.log10(mep)
    log_epsb = numpy.log10(epsb)
    log2 = numpy.log10(2.0)
    log3 = numpy.log10(3.0)
    print(f"log10(epsB)={log_epsb}")
    print(f"log10(mep)={log_mep}")
    print(f"log10(phi_theta)={log_phi}")
    print(
        -8.0*log3 \
        +3.0*log2 \
        -2.0*log_epsb \
        +log_mep
    )
    return asarray((
        2.0*log_phi \
        -8.0*log3 \
        +3.0*log2 \
        -2.0*log_epsb \
        +log_mep
    )/15.0)

def transrelativisticlimit(epsb:float,epsth:float,mep:float):
    phi_a_sample=models.NormalizedWindParameter(
        values=asarray([2.0]),
        unit="GHz min"
    )
    phi_th = models.ThermalSynchrotronPhase(
        values = asarray([1.0]),
        unit=phi_unit

    )
    theta0 = models.ElectronTemperatureScaling(asarray(3.0))
    theta = theta0.to_theta()
    beta_sh = theta0.to_beta(eps_th=epsth,mep=mep).item()
    print(f"beta_sh_tr = {beta_sh:.4f}")
    print(f"phi_th = {phi_th.to_value_in(phi_unit).item():.2e} {phi_unit}")
    print(f"phi_th = {phi_th.decompose().item():.2e}")
    tau_tr_exact = models.OpticalDepthParameter.from_params_exact(
        mep=mep,
        eps_b=epsb,
        theta=theta,
        phi_a=phi_a_sample
    ).values.item()
    tau_tr_hot = models.OpticalDepthParameter.from_params_exact(
        mep=mep,
        eps_b=epsb,
        theta=theta,
        phi_a=phi_a_sample,
        hot=True
    ).values.item()
    print(f"tau_theta_tr = {tau_tr_exact:.2e}")
    print(f"tau_theta_hot_tr = {tau_tr_hot:.2e}")
    log_tau = numpy.log(tau_tr_exact)
    log_tau_hot = numpy.log(tau_tr_hot)
    err = abs((log_tau_hot - log_tau) / log_tau)
    print(f"err = {err:.2e}")

    lnu_th = models.ThermalSynchrotronLuminosity.from_params(
        mep=mep,
        eps_b=epsb,
        eps_th=epsth,
        theta=theta,
        phi_a=phi_a_sample
    ).to_value_in(lnu_unit).item()
    print(f"lnu_theta = {lnu_th:.2e} {lnu_unit}")

    log_theta_thin = calculate_thermally_thin_theta(
        mep=mep,
        epsb=epsb,
        phi_theta=phi_th
    ).item()
    theta_thin = (10**log_theta_thin)
    print(f"log10(Theta_thin) = {log_theta_thin:.3f}")
    print(f"Theta_thin = {theta_thin:.3f}")


transrelativisticlimit(
    epsth=eps_th,
    epsb=eps_b,
    mep=mep
)
