from dataclasses import asdict
from module.utilities import filewriters as fw
from pathlib import Path
import argparse
from module import input_reader
from run import build_tag_microphys,build_tag_reso,build_tag_sampling

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

t_max_list = [5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]
eps_th_list = [1.0,0.8,0.6,0.4]
eps_b_list = [1.0e-1,1.0e-2,1.0e-3]

for eps_b in eps_b_list:
    print(eps_b)
    for eps_th in eps_th_list:
        print(eps_th)
        for t_max in t_max_list:

            phys_params.microphysics.eps_b = eps_b
            phys_params.microphysics.eps_th = eps_th
            sampling_data.max = t_max

            tag_reso = build_tag_reso(phys_params)
            tag_microphys = build_tag_microphys(phys_params)
            tag_sampling = build_tag_sampling(sampling_data)

            scenario_dir = data_dir/tag_reso/tag_microphys
            scenario_dir.mkdir(parents=True,exist_ok=True)

            sampling_dir = scenario_dir/tag_sampling
            sampling_dir.mkdir(parents=True,exist_ok=True)

            path_output_phys_params = scenario_dir/"physical_parameters.yaml"
            path_output_sampling = sampling_dir/"sampling.yaml"

            fw.write_dict_as_yaml(asdict(phys_params),path_output_phys_params) 
            fw.write_dict_as_yaml(asdict(sampling_data),path_output_sampling) 
            print(f"output {path_output_phys_params}")
            print(f"output {path_output_sampling}")
