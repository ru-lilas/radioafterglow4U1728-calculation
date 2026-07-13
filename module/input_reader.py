from module.utilities import filereaders as fr
from dacite import from_dict
from pathlib import Path
from module import input_dataclasses

def read_physical_parameters(path:Path):
    dict_data:dict = fr.read_yaml(path)
    return from_dict(
        data_class=input_dataclasses.PhysicalParameters,
        data=dict_data
    )

def read_sampling(path:Path):
    dict_data:dict = fr.read_yaml(path)
    return from_dict(
        data_class=input_dataclasses.SAMPLING,
        data=dict_data
    )
