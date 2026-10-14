from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    script_name = Path(func.__code__.co_filename).name

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()
        print(f"[{script_name}] started", flush=True)
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            print(f"[{script_name}] elapsed: {elapsed:.2f} s", flush=True)

    return wrapper
