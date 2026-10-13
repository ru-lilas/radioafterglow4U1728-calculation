"""Estimate array memory and check a configurable preflight budget."""

import math
from collections.abc import Sequence

GIB = 1024**3


def estimate_array_memory_bytes(
    shape: Sequence[int],
    array_count: int,
    itemsize: int = 8,
) -> int:
    """Estimate memory for array_count dense arrays of the given shape."""
    if any(size < 0 for size in shape) or array_count <= 0 or itemsize <= 0:
        raise ValueError("配列形状は非負、配列数と要素サイズは正の値が必要です")
    return math.prod(shape) * array_count * itemsize


def check_memory_budget(estimated_bytes: int, max_memory_gib: float) -> None:
    """Reject estimates above the budget; this is not a hard memory limit."""
    if estimated_bytes < 0:
        raise ValueError("推定メモリは非負の値が必要です")
    if not math.isfinite(max_memory_gib) or max_memory_gib <= 0:
        raise ValueError("max_memory_gibには正の有限値を指定してください")
    if estimated_bytes > max_memory_gib * GIB:
        raise MemoryError(
            f"計算配列の推定メモリ {estimated_bytes / GIB:.2f} GiB が"
            f"設定上限 {max_memory_gib:.2f} GiB を超えています。"
            "計算を分割するか、格子点数を減らしてください。"
        )
