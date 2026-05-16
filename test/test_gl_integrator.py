from module import integrator
import numpy as np

glint = integrator.GaussLaguerreIntegrator(64)

print("f(x) = x**3")
f = lambda x: x**3
integral = glint.integrate(f)
print(integral)

print("f(x) = cos(x)")
f = lambda x: np.cos(x)
integral = glint.integrate(f)
print(integral)

print("f(x) = x**2/(1+x**2)")
f = lambda x: x**2 / (1+x**2)
integral = glint.integrate(f)
print(integral)
