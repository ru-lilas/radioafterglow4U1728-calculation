from pathlib import Path
import argparse
from module.utilities import filereaders as fr

parser = argparse.ArgumentParser()

parser.add_argument(
    "config",
    type=Path,
)
args = parser.parse_args()

confpath: Path = args.config
conf = fr.read_yaml(confpath)

t_min = int(conf["min"])
t_max = int(conf["max"])

conf_tag = f"min{t_min:02d}_max{t_max:02d}"
print(conf_tag)

