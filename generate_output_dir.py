from pathlib import Path
import argparse
import shutil
from module.utilities import filereaders as fr

parser = argparse.ArgumentParser()

parser.add_argument(
    "input",
    type=Path,
)
parser.add_argument(
    "--data_dir",
    type=Path,
    default=Path("data")
)
args = parser.parse_args()

confpath: Path = args.input
conf = fr.read_yaml(confpath)

# parameter space resolution
conf_physical_params = conf["physical_parameters"]
reso_beta:int = conf_physical_params["beta_arr"]["num"]
reso_awind:int = conf_physical_params["beta_arr"]["num"]

tag_reso = Path(f"a{reso_awind:04d}b{reso_beta:04d}")

# microphysics parameters
conf_microphys_params = conf_physical_params["microphysics"]
eps_th:float = conf_microphys_params["eps_th"]
eps_b:float = conf_microphys_params["eps_b"]

fmt_eps_th = int(eps_th*100)
fmt_eps_b = int(eps_b*100)
tag_microphys = Path(f"epsth{fmt_eps_th:03d}epsb{fmt_eps_b:04d}")

# sampling
t_min = int(conf["sampling"]["min"])
t_max = int(conf["sampling"]["max"])

tag_sampling = Path(f"min{t_min:02d}_max{t_max:02d}")

output_dir = args.data_dir/tag_reso/tag_microphys/tag_sampling
output_dir.mkdir(parents=True,exist_ok=True)
shutil.copy2(confpath, output_dir/confpath.name)
