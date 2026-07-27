# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false

import numpy as np
from scipy import special
from module.integrator import GaussLaguerreIntegrator,GaussLegendreIntegrator
from typing import Callable

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

    result = np.zeros_like(x)

    for i, xi in enumerate(x):

        def integrand(z):
            return z**2 * F(xi / z**2)

        result[i] = (1.0 / xi) * integrator.integrate(integrand)

    return result

def thermal_Ip_unwrapped(
    x: np.ndarray,
    integrator: GaussLegendreIntegrator,
    I: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:

    mu = integrator.x
    w = integrator.w

    if mu is None or w is None:
        raise RuntimeError("Integrator is not initialized properly.")

    sin_theta = np.sqrt(np.clip(1.0 - mu**2, 0.0, None))

    result = np.zeros_like(x, dtype=np.float64)

    for i, xi in enumerate(x):
        result[i] = np.sum(w * I(xi / sin_theta))

    return result

def thermal_Ip(
    x: np.ndarray
):
    gla_int = GaussLaguerreIntegrator(64)
    gle_int = GaussLegendreIntegrator(64)

    def thI(x:np.ndarray):
        return thermal_I(x,gla_int)

    return thermal_Ip_unwrapped(x,gle_int,thI)

def thermal_Ip_asym(x:np.ndarray) -> np.ndarray:
    """
    Asymptotic form of I'(x) from Mahadevan et al. (1996), Eq.(32).

    Parameters
    ----------
    x : ArrayLike
        Dimensionless frequency ratio (ν / ν_th). Must be positive.

    Returns
    -------
    np.ndarray
        Asymptotic approximation of I'(x).
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-300, None)

    a1 = 4.0505
    b1 = 0.40
    b2 = 0.5316
    c0 = 1.8899

    return (
        a1
        * (1.0 + b1 * x**(-0.25) + b2 * x**(-0.5))
        * np.exp(-c0 * x**(1.0 / 3.0))
        / x**(1.0 / 6.0)
    )

def angle_averaged_dimensionless_synchrotron_power(
    chi: np.ndarray,
    beta_gamma: np.ndarray,
    N:int = 50,
)->np.ndarray:
    """
    無次元化されたsynchrotron specific power <Pi_nu> の積分計算を行う
    積分計算はGauss-Jacobiの方法で実行する

    Parameters
    ----------
    chi: np.ndarray
        nu/nu_b, where nu_b is a gyro frequency
    gb: np.ndarray 
        4-vector of electrons
    N : int, optional
        Gauss–Jacobi 積分点数（default: 50）

    Returns
    -------
    float or ndarray
        ピッチ角平均された無次元 specific power <Pi_nu>
    """
    # Gauss–Jacobi nodes and weights (alpha = beta = 1/2)
    mu, w = special.roots_jacobi(N, 0.5, 0.5)

    # sinθ = sqrt(1 - μ^2)
    sin_theta = np.sqrt(1.0 - mu**2)
    sin_theta = np.clip(sin_theta, 1e-300, None)

    beta_gamma = np.asarray(beta_gamma)
    gamma2 = (1.0 + beta_gamma**2)

    chi = np.asarray(chi)
    x = chi[...,None,None]/(1.5*gamma2[None,...,None]*sin_theta)

    # F(x)
    Fval = F(x)

    # 積分（重みはすでに含まれている）
    integral = np.sum(w * Fval, axis=-1)

    # prefactor 1/2
    return 0.5 * integral

