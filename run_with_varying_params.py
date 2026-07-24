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

t_max_list = [6.0,8.0,10.0,12.0,14.0,16.0]
eps_th_list = [0.4]
eps_b_list = [1.0e-02]

for eps_b in eps_b_list:
    phys_params.microphysics.eps_b = eps_b
    for eps_th in eps_th_list:
        phys_params.microphysics.eps_th = eps_th
        tag_reso = build_tag_reso(phys_params)
        tag_microphys = build_tag_microphys(phys_params)
        pathlist_chi2est:list[Path] = []
        tags_sampling:list[str] = []
        for t_max in t_max_list:
            sampling_data.max = t_max

            tag_sampling = build_tag_sampling(sampling_data)
            tags_sampling.append(tag_sampling)
            figdir = Path('fig',tag_reso,tag_microphys,tag_sampling)
            figdir.mkdir(parents=True,exist_ok=True)
            path_chi2est = data_dir/tag_reso/tag_microphys/tag_sampling/'chi2_estimated_parameters.csv'
            pathlist_chi2est.append(path_chi2est)
            run(
                [
                    "make",
                    f"TAG_RESO={tag_reso}",
                    f"TAG_MICROPHYS={tag_microphys}",
                    f"TAG_SAMPLING={tag_sampling}",
                    f"{path_chi2est}",
                    f"all"
                ],
                check=True
            )
        run(
            [
                "make",
                f"TAG_RESO={tag_reso}",
                f"TAG_MICROPHYS={tag_microphys}",
                f"CHI2_ESTIMATED_LIST={' '.join(map(str, pathlist_chi2est))}",
                f"{data_dir/tag_reso/tag_microphys/"chi2_estimated_summary.csv"}",
                f"{Path("fig",tag_reso,tag_microphys,"chi2test_timewindow.pdf")}"
            ],
            check=True
        )
