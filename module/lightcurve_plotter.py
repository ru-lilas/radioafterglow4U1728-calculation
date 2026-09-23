from typing import Self
from dataclasses import dataclass
from matplotlib.patches import StepPatch
from numpy.typing import NDArray
import pandas as pd
import numpy as np

from module import models, observation, plot_utils
from module.compute_lightcurve import CandidateModelTotalLightcurve
from module.mydataclasses import QuantityData, YAMLReadable

@dataclass(frozen=True,slots=True)
class LightcurveMetadataValues(models.ScenarioParameter):
    a_wind_value: float
    beta_sh: float
    nu_value: float

    @classmethod
    def from_df(
        cls,
        model_index: int,
        df:pd.DataFrame,
    ):
        row:pd.Series = df.loc[df[models.ColumnNames.MODEL_INDEX] == model_index].iloc[0]
        a_wind_value = row[models.ColumnNames.A_WIND]

@dataclass(frozen=True, slots=True)
class BinnedLightcurvePlotter:
    fnu_tot: NDArray[np.float64]
    t_bin_edges: NDArray[np.float64]
    style: plot_utils.LineStyle
    baseline: float

    @classmethod
    def from_lightcurve(
        cls,
        lightcurve: CandidateModelTotalLightcurve,
        style: plot_utils.LineStyle,
        *,
        t_unit: str,
        fnu_unit: str
    )->Self:
        return cls(
            t_bin_edges = lightcurve.bin_edges.to_value_in(t_unit),
            fnu_tot = lightcurve.fnu_total.to_value_in(fnu_unit),
            style=style,
            baseline=lightcurve.fnu_bg.to_value_in(fnu_unit)[0]
        )

    def plot(
        self,
        plotter: plot_utils.Plotter,
    )->StepPatch:
        return plotter.step(
            values=self.fnu_tot,
            edges=self.t_bin_edges,
            style=self.style,
            baseline=self.baseline
        )

@dataclass(frozen=True, slots=True)
class BackgroundEmission:
    t:NDArray[np.float64]
    fnu_bg: NDArray[np.float64]
    style: plot_utils.LineStyle

    @classmethod
    def from_observed_lightcurve(
        cls,
        lightcurve: observation.ObservedLightcurve,
        style: plot_utils.LineStyle,
        *,
        t_unit: str,
        fnu_unit: str
    )->Self:
        return cls(
            t=lightcurve.t.to_value_in(t_unit),
            fnu_bg=lightcurve.fnu_background.to_value_in(fnu_unit),
            style=style
        )

    def plot(
        self,
        plotter: plot_utils.Plotter
    ):
        return plotter.line(
            self.t,self.fnu_bg,
            style=self.style
        )

@dataclass(frozen=True, slots=True)
class Styles:
    observed: plot_utils.ScatterStyle
    binned: plot_utils.LineStyle
    continuous: plot_utils.LineStyle
    background: plot_utils.LineStyle

@dataclass(frozen=True,slots=True)
class LightcurveMetadata:
    fontsize: float
    eps_b:  plot_utils.MetadataComponent
    eps_th: plot_utils.MetadataComponent
    mu_e_mu: plot_utils.MetadataComponent
    distance: plot_utils.MetadataComponent
    a_wind: plot_utils.MetadataComponent
    beta_sh: plot_utils.MetadataComponent

    def plot(
        self,
        plotter:plot_utils.Plotter,
    ):
        strings:list[str] = []
        self.eps_b.add_strings(strings)
        self.eps_th.add_strings(strings)
        self.mu_e_mu.add_strings(strings)
        self.distance.add_strings(strings)
        self.a_wind.add_strings(strings)
        self.beta_sh.add_strings(strings)
        plotter.title(strings,self.fontsize)


@dataclass(frozen=True, slots=True)
class Lightcurve:
    observed: observation.ObservedLightcurvePlotter
    binned: BinnedLightcurvePlotter
    background: BackgroundEmission
    metadata: LightcurveMetadata

    @classmethod
    def from_data(
        cls,
        obs_lc: observation.ObservedLightcurve,
        binned_lc: CandidateModelTotalLightcurve,
        metadata: LightcurveMetadata,
        styles: Styles,
        *,
        t_unit:str,
        fnu_unit:str
    )->Self:
        return cls(
            observed=observation.ObservedLightcurvePlotter.from_observed_lightcurve(
                obs_lc,
                style=styles.observed,
                t_unit=t_unit,
                fnu_unit=fnu_unit
            ),
            binned=BinnedLightcurvePlotter.from_lightcurve(
                binned_lc,
                style=styles.binned,
                t_unit=t_unit,
                fnu_unit=fnu_unit
            ),
            background=BackgroundEmission.from_observed_lightcurve(
                obs_lc,
                style=styles.background,
                t_unit=t_unit,
                fnu_unit=fnu_unit
            ),
            metadata=metadata,
        )

    def plot(
        self,
        plotter:plot_utils.Plotter
    ):
        self.observed.plot(plotter)
        self.binned.plot(plotter)
        self.background.plot(plotter)
        self.metadata.plot(plotter)
        return
