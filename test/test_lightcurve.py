import numpy as np
import pytest
from numpy.testing import assert_allclose
from module.utils import Integrator
from module.compute_lightcurve import CalculationLightcurve
from numpy.testing import assert_allclose
import astropy.units as u
from module.types import FloatArray

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
    actual = CalculationLightcurve._make_bin_edges(
        t_obs,
        width,
        drop_incomplete_bin,
    )
    print(f"\nt_edge = {actual}")

    assert actual.dtype == np.float64
    assert_allclose(actual, expected)

@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        # 定数関数 y = 2：積分値 = 2 × (3 - 0) = 6
        (
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64),
            6.0,
        ),

        # 一次関数 y = 2x + 1
        # 0から3までの積分値 = [x² + x]₀³ = 12
        (
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64),
            12.0,
        ),

        # 不等間隔の一次関数 y = 2x + 1
        (
            np.array([0.0, 0.5, 1.7, 3.0], dtype=np.float64),
            np.array([1.0, 2.0, 4.4, 7.0], dtype=np.float64),
            12.0,
        ),
    ],
)
def test_trapezoid(
    x: np.ndarray,
    y: np.ndarray,
    expected: float,
) -> None:
    actual = Integrator.trapezoid(x, y)

    assert actual == pytest.approx(expected)
