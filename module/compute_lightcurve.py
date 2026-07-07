from typing import cast
from dataclasses import dataclass
from module import quantity_converter
from module.utilities.quantity_data import QuantityData
from module.tabular import ThermalSynchrotronTable
import numpy as np
from numpy.typing import NDArray
from functools import cached_property
import astropy.units as u
from module.synchrotron_scaling_values import calculate_lambda_using_table

@dataclass
class Lightcurve:
    t: QuantityData
    nu: QuantityData
    table_integral:ThermalSynchrotronTable

    @cached_property
    def t_quantity(self):
        return self.t.quantity

    @cached_property
    def nu_quantity(self):
        return self.nu.quantity

    def phi_arr(self,phi_theta:QuantityData):
        phi = u.Quantity(self.t_quantity*self.nu_quantity)
        return u.Quantity(phi.to(phi_theta.unit))

    def xm_arr(
        self,
        phi_theta:QuantityData,
    ):
        xm = (self.phi_arr(phi_theta)/phi_theta.quantity).to_value(u.dimensionless_unscaled)
        return np.asarray(xm,dtype=np.float64)

    def lambda_arr(
        self,
        phi_theta:QuantityData,
        tau_theta: NDArray[np.float64]
    ):
        return calculate_lambda_using_table(
            xm=self.xm_arr(phi_theta),
            tau_theta=tau_theta,
            table=self.table_integral
        )

    def lnu_arr(
        self,
        phi_theta:QuantityData,
        lnu_theta: QuantityData,
        tau_theta: NDArray[np.float64]
    ):
        lnu =  u.Quantity(
            self.lambda_arr(phi_theta,tau_theta)
                *lnu_theta.quantity
        )
        lnu_value = np.asarray(
            lnu.to_value(lnu_theta.unit),dtype=np.float64
        )
        return QuantityData(lnu_value,lnu_theta.unit)

    def fnu(
        self,
        phi_theta:QuantityData,
        lnu_theta: QuantityData,
        tau_theta: NDArray[np.float64],
        d_src: QuantityData
    ):
        lnu = self.lnu_arr(phi_theta,lnu_theta,tau_theta)
        fnu = quantity_converter.lnu_into_fnu(
            lnu=lnu.quantity,
            distance=d_src.quantity
        )
        fnu_value = np.asarray(
            fnu.value,dtype=np.float64
        )
        fnu_unit = cast(u.Unit,fnu.unit).to_string()
        return QuantityData(fnu_value,fnu_unit)
