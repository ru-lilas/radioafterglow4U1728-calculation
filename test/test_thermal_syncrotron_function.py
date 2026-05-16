from module import synchrotron_function,integrator
from numpy import logspace
import pandas as pd

x = logspace(-4.0,3.0,64)

glint = integrator.GaussLaguerreIntegrator(64)
I = synchrotron_function.thermal_I(x,glint)
Ip_asym = synchrotron_function.thermal_Ip_asym(x)

df = pd.DataFrame({
    "x":x,
    "I":I,
    "Ip_asym": Ip_asym
})
df.to_csv("data/test/thermal_syncrotron.csv",index=False)
