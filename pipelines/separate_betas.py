from pathlib import Path
from module.utilities.filereaders import read_yaml
from module.utilities.filewriters import write_dict_as_yaml
import argparse

def main(args:argparse.Namespace):
    outdir = Path("input/thermal_only/beta")

    inpath = args.infile
    input = read_yaml(inpath)
    
    for beta in input["beta_array"]:
        beta_text = f"{beta*100:03.0f}"
        dump_content = {
            "beta_sh":beta
        }
        outpath = outdir/f"beta_{beta_text}.yaml"
        print(f"output {outpath}")
        write_dict_as_yaml(dump_content,outpath)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "infile",
        type=Path
    )
    
    args = parser.parse_args()
    
    print(f"input {args.infile}")
    main(args)

