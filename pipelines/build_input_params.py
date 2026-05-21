from omegaconf import DictConfig
import numpy as np

def build_frequency_array(frequency_config:DictConfig)->dict:
    return np.logspace(
        frequency_config.log10_min,
        frequency_config.log10_max,
        frequency_config.num
    )

def build_frequency_config(frequency_config:DictConfig)->dict:
    nu_array = build_frequency_array(frequency_config)
    return {
        "nu_array_value":nu_array,
        "nu_array_unit": frequency_config.unit
    }

def build_time_config(time_config:DictConfig)->dict:
    return {
        "t_value": time_config.value,
        "t_unit": time_config.unit,
    }

def build_wind_config(wind_config:DictConfig)->dict:
    return {
        "a_wind_value": wind_config.value,
        "a_wind_unit": wind_config.unit,
    }

def config_builder(config:DictConfig)->dict:
    return {

        **config.plasma,
        **build_wind_config(config.wind),
        **build_time_config(config.time),
        **build_frequency_config(config.frequency),
    }

