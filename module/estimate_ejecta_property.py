"""
    Estimate swept mass and kinetic energy of ejecta
"""
from functools import cached_property
from typing import Any
import astropy.units as u
from dataclasses import dataclass

from pandas import DataFrame
from module import dataframe_processors, quantity_converter
from numpy.typing import NDArray
from module.utilities import quantity_data
import numpy as np

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
    return quantity_data.QuantityData(
        value=value,
        unit=unit
    ).quantity

@dataclass
class BursterProperty:
    mdot_acc_value: float
    mdot_acc_unit: str
    t_acc_value: float
    t_acc_unit: str
    eps_nuc_value: float
    eps_nuc_unit: str
    
    @cached_property
    def mdot_acc(self):
        return quantity_data.QuantityData(
            value = np.asarray(self.mdot_acc_value,dtype=np.float64),
            unit = self.mdot_acc_unit
        )

    @cached_property
    def t_acc(self):
        return quantity_data.QuantityData(
            value = np.asarray(self.t_acc_value,dtype=np.float64),
            unit = self.t_acc_unit
        )

    @cached_property
    def eps_nuc(self):
        return quantity_data.QuantityData(
            value = np.asarray(self.eps_nuc_value,dtype=np.float64),
            unit = self.eps_nuc_unit
        )

    @cached_property
    def accumulated_mass(self):
        return calculate_accumulated_mass(
            mdot_acc = self.mdot_acc.quantity,
            t_acc = self.t_acc.quantity
        )

@dataclass
class EstimatedParameters:
    a_wind_value: NDArray[np.float64]
    a_wind_unit: str
    beta_sh: NDArray[np.float64]
    t_peak_value: NDArray[np.float64]
    t_peak_unit: str

    @cached_property
    def a_wind(self):
        return quantity_data.QuantityData(
            value = self.a_wind_value,
            unit = self.a_wind_unit
        )

    @cached_property
    def t_peak(self):
        return quantity_data.QuantityData(
            value = self.t_peak_value,
            unit = self.t_peak_unit
        )

    @cached_property
    def v_ej(self):
        return quantity_converter.beta_into_velocity(self.beta_sh)

@dataclass
class EstimatedEjectaProperty:
    burster_property: BursterProperty
    estimated_params: EstimatedParameters
    energy_unit: str
    mass_unit: str

    @cached_property
    def mdot_acc(self):
        return self.burster_property.mdot_acc.quantity

    @cached_property
    def t_acc(self):
        return self.burster_property.t_acc.quantity

    @cached_property
    def eps_nuc(self):
        return self.burster_property.eps_nuc.quantity

    @cached_property
    def a_wind(self):
        return self.estimated_params.a_wind.quantity

    @cached_property
    def t_peak(self):
        return self.estimated_params.t_peak.quantity

    @cached_property
    def v_ej(self):
        return self.estimated_params.v_ej 

    @cached_property
    def ejected_mass(self):
        m_ej = calculate_swept_mass(
            a_wind = self.a_wind,
            v_ej = self.v_ej,
            t = self.t_peak
        )
        return m_ej.to(u.Unit(self.mass_unit))

    @cached_property
    def kinetic_energy(self):
        e_kin = calculate_kinetic_energy(
            m_ej = self.ejected_mass,
            v_ej = self.v_ej
        )
        return e_kin.to(u.Unit(self.energy_unit))

    @cached_property
    def accumulated_mass(self):
        return self.burster_property.accumulated_mass
    
    @cached_property
    def nuclear_energy(self):
        e_nuc = calculate_released_energy_nuc(
            m_burnt = self.accumulated_mass,
            eps_nuc = self.eps_nuc
        )
        return e_nuc.to(u.Unit(self.energy_unit))

    @cached_property
    def eta(self):
        """
            ratio of ejected mass to acculated mass
        """
        eta = (self.ejected_mass / self.accumulated_mass).to_value(u.dimensionless_unscaled)
        return eta

    @cached_property
    def xi(self):
        """
            ratio of ejeca kinetic energy to nuclear energy
        """
        xi = (self.kinetic_energy / self.nuclear_energy).to_value(u.dimensionless_unscaled)
        return xi

def build_dataframe(ejecta_property:EstimatedEjectaProperty):
    return DataFrame({
        "m_ej_value": ejecta_property.ejected_mass,
        "e_kin_value": ejecta_property.kinetic_energy,
        "eta": ejecta_property.eta,
        "xi": ejecta_property.xi,
        "a_wind_value": ejecta_property.estimated_params.a_wind_value,
        "beta_sh": ejecta_property.estimated_params.beta_sh,
    })
    
def estimate_ejecta_property(
    burster_property: dict[str,Any],
    df_params: DataFrame,
    metadata_params: dict[str,Any]
):
    bp = BursterProperty(**burster_property)
    ep = EstimatedParameters(
        a_wind_value=dataframe_processors.convert_ndarray(df_params,"a_wind_est_value"),
        a_wind_unit=metadata_params["a_wind_unit"],
        beta_sh=dataframe_processors.convert_ndarray(df_params,"beta_sh_est"),
        t_peak_value=dataframe_processors.convert_ndarray(df_params,"t_peak_value"),
        t_peak_unit=metadata_params["t_unit"]
    )

    ejecta_property = EstimatedEjectaProperty(
        burster_property=bp,
        estimated_params=ep,
        energy_unit= "erg",
        mass_unit = "g"
    )

    metadata_output = {
        **{k: metadata_params[k] for k in (
            "eps_th",
            "eps_B",
            "a_wind_unit",
            "mu",
            "mu_e"
        ) if k in metadata_params},
        "energy_unit": "erg",
        "mass_unit": "g"
    }
    
    return metadata_output,build_dataframe(ejecta_property)
