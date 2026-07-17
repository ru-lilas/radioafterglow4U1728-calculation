from pathlib import Path
from numpy.typing import NDArray
import numpy as np
from scipy import stats
from dataclasses import dataclass
from module.input_reader import InputReader

type NDArray64 = NDArray[np.float64]

def calculate_chi2(
    y_model:NDArray64,
    y_obs:NDArray64,
    y_err:NDArray64,
):
    chi_arr = (y_model - y_obs)/y_err
    return np.sum(chi_arr**2).item()

def calculate_reduced_chi2(
    chi2: float,
    ndof: int
):
    return chi2/float(ndof)

def calculate_pvalue(
    chi2_value: float,
    ndof: int
):
    return float(stats.chi2.sf(chi2_value,df=ndof))

def sigma_to_alpha(sigma:float, two_sided:bool = True):
    if two_sided:
        return 2.0 * float(stats.norm.sf(sigma))
    else:
        return float(stats.norm.sf(sigma))

def convert_pvalue_to_sigma(pvalue:float, two_sided:bool = True):
    if two_sided:
        return float(stats.norm.isf(pvalue / 2.0))
    else:
        return float(stats.norm.isf(pvalue))

@dataclass(frozen=True)
class Chi2TestResult:
    chi2: float
    ndof: int
    p_value: float
    alpha: float
    two_sided: bool

    @property
    def reject(self) -> bool:
        return self.p_value < self.alpha

    @property
    def significance(self):
        return convert_pvalue_to_sigma(
            self.p_value,
            self.two_sided,
        )

@dataclass(frozen=True)
class Chi2Test:
    sigma_significance: float
    two_sided: bool

    @property
    def alpha(self):
        return sigma_to_alpha(
            self.sigma_significance,self.two_sided
        )

    def test(
        self,
        chi2: float,
        ndof: int,
    ) -> Chi2TestResult:
        return Chi2TestResult(
            chi2=chi2,
            ndof=ndof,
            p_value=calculate_pvalue(chi2, ndof),
            alpha=self.alpha,
            two_sided=self.two_sided
        )

def load_chi2_test_conf(path:Path = Path("input/chi2_test.yaml")):
    return InputReader.read(path,Chi2Test)
