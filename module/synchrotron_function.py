
import numpy as np
from scipy import special
from module.integrator import GaussLaguerreIntegrator

def F(x:np.ndarray)->np.ndarray:
    """
    analytical fitting of synchrotron function F(x)
    see http://arxiv.org/pdf/1301.6908.pdf
    """
    x = np.asarray(x)
    x = np.clip(x,1.0e-300,None)
    x13 = x**(1.0/3.0)
    x12 = np.sqrt(x)
    GAMMA13 = special.gamma(1/3)
    
    F1 = (np.pi*2.0**(5.0/3.0) / np.sqrt(3.0) / GAMMA13) * x13;
    F2 = np.sqrt(0.5*np.pi)*np.exp(-x)* x12
    
    H1 = (-0.97947838884478688 * x
          -0.83333239129525072 * x12
          +0.1554179602681624 * x13
          )
    delta1 = np.exp(H1);
    
    H2 = (-0.0469247165562628882 * x
          -0.70055018056462881 * x12
          +0.0103876297841949544 * x13)
    delta2 = - np.expm1(H2)
    
    return F1*delta1+F2*delta2;

def thermal_I(
    x:np.ndarray,
    integrator:GaussLaguerreIntegrator
)->np.ndarray:

    def integrand(z):
        return z**2 * F(x/z**2)

    return (1.0/x)* integrator.integrate(integrand)
