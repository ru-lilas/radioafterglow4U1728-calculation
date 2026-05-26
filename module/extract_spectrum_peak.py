import numpy as np
import pandas as pd


def extract_peak_row(df:pd.DataFrame) -> dict[str, float]:

    nu = df["nu"].to_numpy()
    lnu = df["lnu"].to_numpy()

    imax = int(np.argmax(lnu))

    nu_peak = float(nu[imax])
    lnu_peak = float(lnu[imax])

    return {
        "nu_peak": nu_peak,
        "lnu_peak": lnu_peak,
    }

