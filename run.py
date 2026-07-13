from pathlib import Path
import argparse
from module.input_reader import read_sampling
from module.utilities import filereaders as fr
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
conf = fr.read_yaml(inpath)
sampling_conf = read_sampling(sampling_path)

# parameter space resolution
reso_beta:int = conf["beta_sh_arr"]["num"]
reso_awind:int = conf["beta_sh_arr"]["num"]

tag_reso = f"a{reso_awind:04d}b{reso_beta:04d}"

# microphysics parameters
conf_microphys_params = conf["microphysics"]
eps_th:float = conf_microphys_params["eps_th"]
eps_b:float = conf_microphys_params["eps_b"]
fmt_eps_th = int(eps_th*100)
fmt_eps_b = int(eps_b*100)
tag_microphys = f"epsth{fmt_eps_th:03d}epsb{fmt_eps_b:03d}"

scenario_dir = data_dir/tag_reso/tag_microphys

scenario_dir.mkdir(parents=True,exist_ok=True)

t_min = int(sampling_conf.min)
t_max = int(sampling_conf.max)
tag_sampling = f"min{t_min:02d}max{t_max:02d}"
sampling_dir = scenario_dir/tag_sampling
sampling_dir.mkdir(parents=True,exist_ok=True)


run(
    [
        "make",
        f"TAG_RESO={tag_reso}",
        f"TAG_MICROPHYS={tag_microphys}",
        f"TAG_SAMPLING={tag_sampling}",
        f"all"
    ]
)
