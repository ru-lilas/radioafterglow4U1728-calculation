from numpy import asarray, float64
from module.models import TransRelativisticLimit
from module.mydataclasses import QuantityArray1D


tr_lim = TransRelativisticLimit(
    eps_b = 1.0e-02,
    eps_th = 0.40,
    mu = 0.5,
    mu_e = 1.0,
)
a_wind = 1.0e+06

theta_trlim = tr_lim.theta
tau_theta_trlim = tr_lim.tau_theta(
    a_wind=QuantityArray1D(
        values = asarray([a_wind],dtype=float64),
        unit = "g cm-1"
    )
)
print(f"a_wind={a_wind:.1e} g cm-1")
print(f"tau_theta = {tau_theta_trlim[0]:.2e}")
