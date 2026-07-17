from pathlib import Path
import argparse
from module import input_reader
from module.input_dataclasses import SAMPLING, PhysicalParameters
from module.input_reader import read_sampling
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

def build_tag_reso(data_phys_param:PhysicalParameters):
    reso_awind = data_phys_param.a_wind_arr.num
    reso_beta = data_phys_param.beta_sh_arr.num

    return f"a{reso_awind:04d}b{reso_beta:04d}"

def build_tag_microphys(data_phys_param:PhysicalParameters):
    data_microphys_param = data_phys_param.microphysics
    eps_th = data_microphys_param.eps_th
    eps_b = data_microphys_param.eps_b
    fmt_eps_th = int(eps_th*100)
    fmt_eps_b = int(eps_b*100)
    return f"epsth{fmt_eps_th:03d}epsb{fmt_eps_b:03d}"

def build_tag_sampling(conf_sampling:SAMPLING):
    t_min = int(conf_sampling.min)
    t_max = int(conf_sampling.max)
    return f"min{t_min:02d}max{t_max:02d}"

def run_simulation(
        inpath:Path,
        sampling_path:Path,
        data_dir:Path
):
    data_phys_param = input_reader.read_physical_parameters(inpath)
    sampling_conf = read_sampling(sampling_path)

    # parameter space resolution
    tag_reso = build_tag_reso(data_phys_param)

    # microphysics parameters
    tag_microphys = build_tag_microphys(data_phys_param)

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
        ],
        check=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ...
    args = parser.parse_args()

    run_simulation(
        args.input,
        args.sampling,
        args.data_dir,
    )
