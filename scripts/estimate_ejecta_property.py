import argparse
from pathlib import Path

from module.utilities import filereaders as fr
from module.utilities import filewriters as fw
import pandas as pd
from module import estimate_ejecta_property

def main(args:argparse.Namespace):
    burster_property_path:Path = args.burster_property
    outpath:Path = args.output
    outpath.parent.mkdir(parents=True,exist_ok=True)

    # inpath:Path = args.input
    # df_input = fr.read_csv(inpath)
    # metadata_input = fr.read_keyvalue(inpath)
    burster_data_dict = fr.read_yaml_pyyaml(burster_property_path)

    burster_data = estimate_ejecta_property.BursterProperty(
        **burster_data_dict
    )

    m_acc = burster_data.accumulated_mass
    e_nuc = burster_data.nuclear_energy()

    m_acc = m_acc.to_value("g")
    e_nuc = e_nuc.to_value("erg")

    metadata = {
        "m_unit": "g",
        "e_unit": "erg"
    }

    df = pd.DataFrame({
        "m_acc": [m_acc],
        "e_nuc": [e_nuc]
    })

    fw.write_csv_with_params(
        df,metadata,outpath
    )

    # metadata_output,df = estimate_ejecta_property.estimate_ejecta_property(
    #     burster_property=burster_data,
    #     df_params=df_input,
    #     metadata_params=metadata_input
    # )
    # print(metadata_output)
    #
    # print(df)
    #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # parser.add_argument(
    #     "input",
    #     type=Path,
    # )
    parser.add_argument(
        "--burster_property",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    args = parser.parse_args()
    main(args)

