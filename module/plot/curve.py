from matplotlib.axes import Axes
from module.plot.plot_utils import CurveConfigure
import pandas as pd
from module.utilities import quantity_data
from module import dataframe_processors as dfp

def curve(
    ax:Axes,
    x,y,curveconf:CurveConfigure
):
    ax.plot(
        x,y,
        ls=curveconf.linestyle,
        lw=curveconf.linewidth,
        color=curveconf.color,
        label=curveconf.label
    )

def build_axisarray(
    df:pd.DataFrame,
    column_name: str,
    old_unit: str,
    new_unit: str
):
    quantity = quantity_data.QuantityData(
        value = dfp.convert_ndarray(df,column_name),
        unit = old_unit
    )
    return quantity.to_ndarray(new_unit)

