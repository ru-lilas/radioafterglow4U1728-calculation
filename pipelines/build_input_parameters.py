from pathlib import Path
from omegaconf import OmegaConf
import argparse

def main(args:argparse.Namespace):
    input_freq = OmegaConf.load(args.frequency)
    input_time = OmegaConf.load(args.time)
    input_fixed = OmegaConf.load(args.fixed)
    input_varying = OmegaConf.load(args.varying)

    output_dir:Path = args.outdir
    output_dir.mkdir(parents=True,exist_ok=True)

    input_precombined = OmegaConf.merge(input_freq,input_time,input_fixed)

    for beta in input_varying.beta_array:
        input_combined = OmegaConf.merge(input_precombined,{"beta_sh":beta})
        outpath = output_dir/f"{beta*100:03.0f}.yaml"
        OmegaConf.save(input_combined,outpath)

        print(f"output {outpath}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--frequency",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--time",
        type=Path,
        required=True
    )
    # 変化パラメータの配列を記載したファイルのパス
    parser.add_argument(
        "--varying",
        type=Path,
        required=True
    )
    # 固定パラメータの配列を記載したファイルのパス
    parser.add_argument(
        "--fixed",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True
    )
    
    args = parser.parse_args()
    
    main(args)
