from argparse import Namespace
from module import tabular
from pathlib import Path

def fetch_numerical_table(args:Namespace):
    tabular_path:Path = args.table
    df_table = tabular.read_tabular(tabular_path)
    return tabular.ThermalSynchrotronTable(df_table)

def fetch_tau_table(args:Namespace):
    tabular_path:Path = args.table
    df_table = tabular.read_tabular(tabular_path)
    return tabular.TauThetaTable(df_table)

