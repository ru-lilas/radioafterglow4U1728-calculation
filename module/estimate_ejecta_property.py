"""
    Estimate swept mass and kinetic energy of ejecta
"""
from functools import cached_property
from typing import Any, Self
import astropy.units as u
from dataclasses import dataclass

import pandas as pd
from module import dataframe_processors, quantity_converter
from numpy.typing import NDArray
import numpy as np
from module.mydataclasses import QuantityData


def require_scalar_float(value: object) -> float:
    """Validate a real scalar or a single-element array and return a float"""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("単一の数値が必要です")
        return require_scalar_float(value.item())
    if isinstance(value, np.generic):
        return require_scalar_float(value.item())
    if isinstance(value, float):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"実数のスカラーが必要です: {type(value).__name__}")

def calculate_swept_mass(
    a_wind: u.Quantity,
    v_ej: u.Quantity,
    t: u.Quantity
):
    return u.Quantity(a_wind*v_ej*t)

def calculate_kinetic_energy(
    m_ej: u.Quantity,
    v_ej: u.Quantity
):
    return u.Quantity(0.5*m_ej*v_ej*v_ej)

def calculate_accumulated_mass(
    mdot_acc: u.Quantity,
    t_acc: u.Quantity
):
    return u.Quantity(mdot_acc*t_acc)

def calculate_released_energy_nuc(
    m_burnt: u.Quantity,
    eps_nuc: u.Quantity
):
    return u.Quantity(m_burnt*eps_nuc)

def build_quantity_data(
    metadata:dict[str,Any],
    quantity_name: str
):
    value = np.asarray(metadata[f"{quantity_name}_value"],dtype=np.float64)
    unit = metadata[f"{quantity_name}_unit"]
    return u.Quantity(
        value,
        unit
    )

@dataclass(frozen=True, slots=True)
class BursterProperty:
    burster_name: str
    mdot_acc: u.Quantity
    t_acc: u.Quantity
    eps_nuc: u.Quantity
    
    @property
    def accumulated_mass(self):
        return calculate_accumulated_mass(
            mdot_acc = self.mdot_acc,
            t_acc = self.t_acc
        )

    def nuclear_energy(self,fraction:float=1.0):
        return calculate_released_energy_nuc(
            m_burnt=fraction*self.accumulated_mass,
            eps_nuc=self.eps_nuc
        )

@dataclass(frozen=True, slots=True)
class EjectaProperties:
    a_wind: u.Quantity
    beta_sh: float | NDArray[np.float64]
    t_peak: u.Quantity

    @property
    def v_ej(self):
        return quantity_converter.beta_into_velocity(self.beta_sh)

    @property
    def ejected_mass(self):
        m_ej = calculate_swept_mass(
            a_wind = self.a_wind,
            v_ej = self.v_ej,
            t = self.t_peak
        )
        return m_ej

    @property
    def kinetic_energy(self):
        e_kin = calculate_kinetic_energy(
            m_ej = self.ejected_mass,
            v_ej = self.v_ej
        )
        return e_kin

@dataclass(frozen=True, slots=True)
class Energetics:
    mass_fraction: float
    energy_fraction: float

    @classmethod
    def build_from(
        cls,
        burster: BursterProperty,
        ejecta: EjectaProperties,
        *,
        burnt_mass_fraction: float
    )->Self:
        return cls(
            mass_fraction = require_scalar_float(
                (ejecta.ejected_mass / burster.accumulated_mass).to_value(u.dimensionless_unscaled)
            ),
            energy_fraction = require_scalar_float(
                (ejecta.kinetic_energy / burster.nuclear_energy(burnt_mass_fraction)).to_value(u.dimensionless_unscaled)
            ),
        )
