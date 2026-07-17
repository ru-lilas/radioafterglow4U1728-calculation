from dataclasses import asdict
from pathlib import Path
import argparse
from module.utilities import filewriters as fw
from module import input_reader
from run import run_simulation,build_tag_microphys,build_tag_reso,build_tag_sampling
from subprocess import run

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=Path,
    default=Path("input/physical_parameters.yaml")
)
parser.add_argument(
    "--sampling",
    type=Path,
    default=Path("input/sampling.yaml")
)
parser.add_argument(
    "--data_dir",
    type=Path,
    default=Path("data")
)
args = parser.parse_args()

inpath: Path = args.input
sampling_path:Path = args.sampling
data_dir: Path = args.data_dir

phys_params = input_reader.read_physical_parameters(inpath)
sampling_data = input_reader.read_sampling(sampling_path)

t_max_list = [10.0,12.0,14.0,16.0]
eps_th_list = [1.0,0.6,0.4]
eps_b_list = [1.0e-03]

for eps_b in eps_b_list:
    for eps_th in eps_th_list:
        for t_max in t_max_list:
            phys_params.microphysics.eps_b = eps_b
            phys_params.microphysics.eps_th = eps_th
            sampling_data.max = t_max

            tag_reso = build_tag_reso(phys_params)
            tag_microphys = build_tag_microphys(phys_params)
            tag_sampling = build_tag_sampling(sampling_data)
            figdir = Path('fig',tag_reso,tag_microphys,tag_sampling)
            figdir.mkdir(parents=True,exist_ok=True)
            run(
                [
                    "make",
                    f"TAG_RESO={tag_reso}",
                    f"TAG_MICROPHYS={tag_microphys}",
                    f"TAG_SAMPLING={tag_sampling}",
                    f"{data_dir/tag_reso/tag_microphys/tag_sampling/'estimated_lightcurve.csv'}",
                    f"{data_dir/tag_reso/tag_microphys/tag_sampling/'chi2_estimated_parameters.csv'}"
                    # f"{figdir/'chi2_colormap.pdf'}"
                ],
                check=True
            )
