import numpy as np
import pytest
from numpy.testing import assert_allclose
from module.compute_lightcurve import Lightcurve

@pytest.mark.parametrize(
    ("t_obs", "width", "drop_incomplete_bin", "expected"),
    [
        # 末端がbin境界と一致する
        (
            np.array([0.0, 60.0, 120.0, 180.0]),
            60.0,
            True,
            np.array([0.0, 60.0, 120.0, 180.0]),
        ),
        (
            np.array([0.0, 60.0, 120.0, 180.0]),
            60.0,
            False,
            np.array([0.0, 60.0, 120.0, 180.0]),
        ),

        # 不完全な末端binを捨てる
        (
            np.array([0.0, 60.0, 120.0, 200.0]),
            60.0,
            True,
            np.array([0.0, 60.0, 120.0, 180.0]),
        ),

        # 不完全な末端binを残す
        (
            np.array([0.0, 60.0, 120.0, 200.0]),
            60.0,
            False,
            np.array([0.0, 60.0, 120.0, 180.0, 200.0]),
        ),

        # 開始時刻が0ではない
        (
            np.array([30.0, 80.0, 150.0, 230.0]),
            60.0,
            True,
            np.array([30.0, 90.0, 150.0, 210.0]),
        ),
        (
            np.array([30.0, 80.0, 150.0, 230.0]),
            60.0,
            False,
            np.array([30.0, 90.0, 150.0, 210.0, 230.0]),
        ),

        # 全時間範囲がbin_widthより短い
        (
            np.array([0.0, 30.0]),
            60.0,
            True,
            np.array([0.0]),
        ),
        (
            np.array([0.0, 30.0]),
            60.0,
            False,
            np.array([0.0, 30.0]),
        ),
    ],
)
def test_make_bin_edges(
    t_obs: np.ndarray,
    width: float,
    drop_incomplete_bin: bool,
    expected: np.ndarray,
) -> None:
    actual = Lightcurve._make_bin_edges(
        t_obs,
        width,
        drop_incomplete_bin,
    )

    assert actual.dtype == np.float64
    assert_allclose(actual, expected)
