import pandas as pd
from module.mydataclasses import QuantityArray1D


@pd.api.extensions.register_dataframe_accessor("q")
class QuantityAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._df = pandas_obj

    def array1d(
        self,
        column: str,
        unit_key: str,
    ) -> QuantityArray1D:
        return QuantityArray1D.from_df(
            self._df,
            column,
            unit_key=unit_key,
        )
