from module.models import SynchrotronSpectrum,InputParameters
from dataclasses import asdict
import pandas as pd

def metadata(inputparams:InputParameters,ss:SynchrotronSpectrum):
    metadata = {
        **asdict(inputparams),
        "t_value" : ss.t_value,
        "t_unit" : ss.t_unit,
        "r_value" : ss.r.value,
        "r_unit" : ss.r.unit,
        "n_wind_value" : ss.n_wind.value,
        "n_wind_unit" : ss.n_wind.unit,
        "b_mag_value" : ss.b_mag.value,
        "b_mag_unit" : ss.b_mag.unit,
        "theta_e" : ss.theta_e,
        "nu_B_value" : ss.nu_B.value,
        "nu_crit_value" : ss.nu_crit.value,
        "j0_value" : ss.j0.value,
        "j0_unit" : ss.j0.unit,
        "a0_value" : ss.a0.value,
        "a0_unit" : ss.a0.unit,
    }

    return metadata

def tabledata(ss:SynchrotronSpectrum)->pd.DataFrame:
    df = pd.DataFrame({
        "nu":ss.nu.value_array,
        "chi":ss.chi,
        "x_m":ss.xm,
        "jnu_dimless":ss.j_nu_th_dimless,
        "anu_dimless":ss.a_nu_th_dimless,
        "snu_dimless":ss.S_nu_th_dimless,
        "lnu":ss.lnu_th,
        "tau_nu":ss.tau_nu
    })
    return df
