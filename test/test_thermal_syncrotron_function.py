from module import synchrotron_function,integrator
from numpy import logspace
import pandas as pd

x = logspace(-4.0,3.0,64)

gauss_laguerre_integrator = integrator.GaussLaguerreIntegrator(64)
gauss_legendre_integrator = integrator.GaussLegendreIntegrator(64)
I = synchrotron_function.thermal_I(x,gauss_laguerre_integrator)
def I_func(x):
    return synchrotron_function.thermal_I(x,gauss_laguerre_integrator)
Ip = synchrotron_function.thermal_Ip_unwrapped(x,gauss_legendre_integrator,I_func)
Ip_asym = synchrotron_function.thermal_Ip_asym(x)

df = pd.DataFrame({
    "x":x,
    "I":I,
    "Ip":Ip,
    "Ip_asym": Ip_asym
})
df.to_csv("data/test/thermal_syncrotron.csv",index=False)
