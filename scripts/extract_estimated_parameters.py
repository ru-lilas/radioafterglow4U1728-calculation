from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
from pathlib import Path
import argparse
import numpy as np
from module import nearest_neighbor_search

def main(args:argparse.Namespace):
    inpath:Path = args.input
    scatpath:Path = args.scatters
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    df_contour = fr.read_csv(inpath)
    metadata_contour = fr.read_keyvalue(inpath)
    df_scatters = fr.read_csv(scatpath)
    metadata_scatters = fr.read_keyvalue(scatpath)

    df_output = df_scatters.copy()

    phi_peak_bg = np.asarray(df_contour["phi_peak"],dtype=np.float64)
    l_peak_bg = np.asarray(df_contour["l_peak"],dtype=np.float64)
    phi_peak_pt = np.asarray(df_scatters["phi_peak"],dtype=np.float64)
    l_peak_pt = np.asarray(df_scatters["l_peak"],dtype=np.float64)

    log10_phi_peak_bg = np.log10(phi_peak_bg)
    log10_l_peak_bg = np.log10(l_peak_bg)
    log10_phi_peak_pt = np.log10(phi_peak_pt)
    log10_l_peak_pt = np.log10(l_peak_pt)

    idx = nearest_neighbor_search.linear(
        log10x_bg=log10_phi_peak_bg,
        log10y_bg=log10_l_peak_bg,
        log10x_pt=log10_phi_peak_pt,
        log10y_pt=log10_l_peak_pt
    )
    a_wind_arr = np.asarray(df_contour["a_wind_value"],dtype=np.float64)
    beta_sh_arr = np.asarray(df_contour["beta_sh"],dtype=np.float64)

    df_output["phi_peak_est"] = phi_peak_bg[idx]
    df_output["l_peak_est"] = l_peak_bg[idx]
    df_output["a_wind_est"] = a_wind_arr[idx]
    df_output["beta_sh_est"] = beta_sh_arr[idx]

    metadata_output = {
        **metadata_scatters,
        **metadata_contour
    }

    fw.write_csv_with_params(df_output,metadata_output,outpath)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "input",
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--scatters",
        type=Path,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True
    )
    
    args = parser.parse_args()
    
    main(args)


