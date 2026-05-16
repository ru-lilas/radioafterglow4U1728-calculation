
def require_key(d: dict, key: str):
    if key not in d:
        raise KeyError(
            f"キー'{key}'が見つかりません. 存在するキー:{list(d.keys())}"
        )
    return d[key]

def require_keys(d: dict, keys: list[str]):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing keys: {missing}")
