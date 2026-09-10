from module.compute_lightcurve import ChevalierParameters
from module.mydataclasses import QuantityData, YAMLReadable
from dataclasses import dataclass,fields
from typing import Self
import astropy.units as u

from module.quantity_converter import VelocityDevidedByLightspeed


@dataclass(frozen=True,slots=True)
class ObservedBursterProperty(YAMLReadable):
    burster_name: str
    mdot_acc: QuantityData
    t_acc: QuantityData
    eps_nuc: QuantityData

    @property
    def m_acc(self)->QuantityData:
        mdot_acc = self.mdot_acc.quantity
        t_acc = self.t_acc.quantity
        m_acc = mdot_acc*t_acc
        return QuantityData.from_Quantity(m_acc)

    @property
    def e_nuc_max(self)->QuantityData:
        m_acc = self.m_acc.quantity
        eps_nuc = self.eps_nuc.quantity
        e_nuc_max = m_acc*eps_nuc
        return QuantityData.from_Quantity(e_nuc_max)

@dataclass(frozen=True,slots=True)
class BursterParameters(ChevalierParameters):
    pass

    @classmethod
    def from_chevalier(
        cls,
        params: ChevalierParameters,
    ) -> Self:
        return cls(
            **{
                field.name: getattr(params, field.name)
                for field in fields(ChevalierParameters)
            }
        )

    @property
    def v_sh(self):
        beta_sh = VelocityDevidedByLightspeed(
            value=self.beta_sh
        )
        return beta_sh.to_velocity()

    def t_peak(
        self,
        nu:QuantityData
    )->QuantityData:
        phi_peak_q = self.phi_peak.quantity
        nu_q = nu.quantity
        t_peak_q = phi_peak_q/nu_q
        if not isinstance(t_peak_q,u.Quantity):
            raise ValueError("t_peakの値が不正です.")
        else:
            return QuantityData.from_Quantity(t_peak_q)

    def m_swept(
        self,
        nu:QuantityData
    )->QuantityData:
        t_peak = self.t_peak(nu)
        m_swept = self.a_wind.quantity \
            * self.v_sh.quantity \
            * t_peak.quantity
        return QuantityData.from_Quantity(m_swept)

    def e_kin_min(
        self,
        nu:QuantityData
    )->QuantityData:
        m_swept = self.m_swept(nu).quantity
        v_ej = self.v_sh.quantity
        e_kin_min = 0.5*(m_swept*v_ej**2)
        if not isinstance(e_kin_min,u.Quantity):
            raise ValueError("e_kin_minの値が不正です.")
        else:
            return QuantityData.from_Quantity(e_kin_min)
