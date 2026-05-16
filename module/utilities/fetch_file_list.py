""" utilities/fetch_file_list.py
    あるディレクトリに含まれるファイルをリストで取得する
"""
from pathlib import Path

def csv(directory:Path,recursive:bool=False)->list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"エラー:ディレクトリ'{directory}'は存在しません")

    if not directory.is_dir():
        raise NotADirectoryError(f"エラー:'{directory}'はディレクトリではありません")

    if recursive:
        paths = sorted(directory.rglob("*.csv"))
    else:
        paths = sorted(directory.glob("*.csv"))

    if not paths:
        raise FileNotFoundError(f"{directory}にcsvファイルが存在しません")

    return paths

def yaml(directory:Path):
    yaml_files = list(directory.glob("*.yaml"))
    return yaml_files
