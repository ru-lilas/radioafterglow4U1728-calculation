# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownMemberType=false
from dataclasses import dataclass
from typing import Self
import astropy.units as u
import numpy as np
import pandas as pd

from module.mydataclasses import (
    FloatVector, FloatVectorSpace,
    QuantityArrayBase, QuantityData,
)
from module.models import (
    SynchrotronResults,
    DefaultUnits,
    ColumnNames,
    MetadataNames,
)

@dataclass(frozen=True, slots=True)
class SynchrotronSpectrum(SynchrotronResults):
    t: QuantityData

    @property
    def nu(self) -> u.Quantity:
        return self.phi / self.t.quantity

    @classmethod
    def from_results(
        cls,
        results:SynchrotronResults,
        t:QuantityData
    )->Self:
        return cls(
            model_params=results.model_params,
            xm=results.xm,
            j_nu=results.j_nu,
            alpha_nu=results.alpha_nu,
            snu=results.snu,
            tau=results.tau,
            lnu=results.lnu,
            fnu_obsframe=results.fnu_obsframe,
            t=t,
        )

    def to_df(
        self,
        *,
        fnu_unit: str = DefaultUnits.fnu,
        phi_unit: str = DefaultUnits.phi,
        nu_unit: str = DefaultUnits.nu,
        jnu_unit: str = DefaultUnits.jnu,
        alpha_nu_unit: str = DefaultUnits.anu,
        t_unit: str = DefaultUnits.t,
        lnu_unit: str = DefaultUnits.lnu,
        index_column_name: str = ColumnNames.MODEL_INDEX,
    ) -> pd.DataFrame:
        df = SynchrotronResults.to_df(
            self,
            fnu_unit=fnu_unit,
            phi_unit=phi_unit,
            index_column_name=index_column_name
        )
        jnu = self.j_nu
        alpha = self.alpha_nu
        t = self.t.quantity
        df["jnu_thermal"] = jnu.thermal(t).to_value(jnu_unit).ravel()
        df["jnu_powerlaw"] = jnu.powerlaw(t).to_value(jnu_unit).ravel()
        df["jnu_total"] = jnu.total(t).to_value(jnu_unit).ravel()
        df["alpha_nu_thermal"] = alpha.thermal(t).to_value(alpha_nu_unit).ravel()
        df["alpha_nu_powerlaw"] = alpha.powerlaw(t).to_value(alpha_nu_unit).ravel()
        df["alpha_nu_total"] = alpha.total(t).to_value(alpha_nu_unit).ravel()
        df[ColumnNames.NU] = self.nu.to_value(nu_unit).ravel()
        df[ColumnNames.T] = self.t.value_in(t_unit)
        df.attrs[MetadataNames.NU_UNIT] = str(nu_unit)
        df.attrs[MetadataNames.T_UNIT] = str(t_unit)
        df.attrs["jnu_unit"]=str(jnu_unit)
        df.attrs["alpha_nu_unit"]=str(alpha_nu_unit)
        return df

