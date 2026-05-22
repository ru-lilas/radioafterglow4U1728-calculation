import numpy as np

def convert_beta_into_u(beta:float):
    gamma_factor = 1.0/(np.sqrt(1.0-beta**2))
    return beta*gamma_factor
    
