"""Calculate ejecta properties and energetics for bestfit models"""

import argparse
from pathlib import Path

import numpy as np

import astropy.units as u
import pandas as pd

from module.estimate_ejecta_property import BursterProperty, EjectaProperties, Energetics, require_scalar_float
from module.inputs_as_dataclass import GeneralInputs
from module.models import ColumnNames, DefaultUnits, MetadataNames, ModelParameters
from module.utils import FileReader, FileWriter

MASS_UNIT = "g"
A_WIND_UNIT = "g/cm"
ENERGY_UNIT = "erg"
BURNT_MASS_FRACTION = 1.0
# MetadataNamesに追加されていない版でも同じキーで保存できる
BURNT_MASS_FRACTION_KEY = "burnt_mass_fraction"


def require_model_index(value: object) -> int:
    """Validate an integer model identifier without truncating fractional values"""
    if isinstance(value, np.generic):
        return require_model_index(value.item())
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError("model_indexには整数が必要です")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "chevalier_bestfit_points", "model_parameter_table", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def build_burster_property(inputs: GeneralInputs) -> BursterProperty:
    """Build accretion and nuclear-burning properties from the input configuration"""
    config = inputs.burster_property
    return BursterProperty(
        burster_name=config.burster_name,
        mdot_acc=config.mdot_acc.quantity,
        t_acc=config.t_acc.quantity,
        eps_nuc=config.eps_nuc.quantity,
    )


def build_ejecta_properties(
    model: ModelParameters,
    t_peak: float,
    t_unit: str,
) -> EjectaProperties:
    """Build scalar ejecta properties from one model and its peak time"""
    a_wind = model.a_wind.item()
    if not isinstance(a_wind, u.Quantity):
        raise TypeError("a_windにはQuantityが必要です")
    return EjectaProperties(
        a_wind=a_wind,
        beta_sh=require_scalar_float(model.beta_sh.item()),
        t_peak=u.Quantity(t_peak, t_unit),
    )


def calculate_energetics_table(
    df_bests: pd.DataFrame,
    models: ModelParameters,
    burster: BursterProperty,
    *,
    burnt_mass_fraction: float,
) -> pd.DataFrame:
    """Calculate an energetics table while preserving bestfit row order"""
    if df_bests.empty:
        raise ValueError("計算対象のbestfit行がありません")

    t_unit = df_bests.attrs[MetadataNames.T_UNIT]
    records: list[dict[str, int | float]] = []
    for row in df_bests.to_dict(orient="records"):
        model_index = require_model_index(row[ColumnNames.MODEL_INDEX])
        selected_model = models.select_models([model_index])
        ejecta = build_ejecta_properties(
            selected_model,
            t_peak=require_scalar_float(row[ColumnNames.T_CENTER]),
            t_unit=t_unit,
        )
        energetics = Energetics.build_from(
            burster, ejecta, burnt_mass_fraction=burnt_mass_fraction,
        )
        records.append({
            "model_index": model_index,
            "mass_fraction": energetics.mass_fraction,
            "energy_fraction": energetics.energy_fraction,
            "nu": require_scalar_float(row[ColumnNames.NU]),
            "kinetic_energy": require_scalar_float(ejecta.kinetic_energy.to_value(ENERGY_UNIT)),
            "ejected_mass": require_scalar_float(ejecta.ejected_mass.to_value(MASS_UNIT)),
            "a_wind": require_scalar_float(ejecta.a_wind.to_value(A_WIND_UNIT)),
            "beta_sh": require_scalar_float(ejecta.beta_sh),
            "t_peak": require_scalar_float(ejecta.t_peak.to_value(DefaultUnits.t)),
            MetadataNames.EPS_B: selected_model.eps_b,
            MetadataNames.EPS_TH: selected_model.eps_th,
            MetadataNames.MU_E_MU: selected_model.molecular_weight.to_ratio(),
        })

    df = pd.DataFrame.from_records(records)
    df.attrs = {
        MetadataNames.DISTANCE_VALUE: models.distance.value,
        MetadataNames.DISTANCE_UNIT: models.distance.unit,
        "energy_unit": ENERGY_UNIT,
        "mass_unit": MASS_UNIT,
        MetadataNames.T_UNIT: DefaultUnits.t,
        "a_wind_unit": A_WIND_UNIT,
        BURNT_MASS_FRACTION_KEY: burnt_mass_fraction,
    }
    if MetadataNames.NU_UNIT in df_bests.attrs:
        df.attrs[MetadataNames.NU_UNIT] = df_bests.attrs[MetadataNames.NU_UNIT]
    return df


def main() -> None:
    args = parse_args()
    inputs = GeneralInputs.from_yaml(args.input)
    models = ModelParameters.from_df(FileReader.table_from_csv(args.model_parameter_table))
    df_bests = FileReader.table_from_csv(args.chevalier_bestfit_points)
    df = calculate_energetics_table(
        df_bests,
        models,
        build_burster_property(inputs),
        burnt_mass_fraction=BURNT_MASS_FRACTION,
    )
    print(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    FileWriter.df_to_csv(path=args.output, df=df)


if __name__ == "__main__":
    main()
