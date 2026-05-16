import numpy as np
from numpy.typing import NDArray
from modules.utilities import fetch_file_list,filereaders,refer_dict

def log10(parameter_config):
    parameter_array:NDArray = np.logspace(
        refer_dict.require_key(parameter_config,"log10_min"),
        refer_dict.require_key(parameter_config,"log10_max"),
        refer_dict.require_key(parameter_config,"num"),
    )
    return parameter_array

