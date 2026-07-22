from pathlib import Path
from numpy.typing import NDArray
import numpy as np
from scipy import stats
from dataclasses import dataclass,asdict
from module.input_reader import InputReader

type NDArray64 = NDArray[np.float64]

def calculate_chi2(
    x_model:NDArray64|float,
    x_data:NDArray64,
    sigma:NDArray64,
):
    chi_arr = (x_model - x_data)/sigma
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

def calculate_muhat(
    x:NDArray64,
    sigma:NDArray64
):
    """
        誤差e_iがN(0,sigma_i)に従う場合の最尤推定母平均と
        その不確かさ(真の値からのずれ)を計算する
        x: データ
        sigma: データの誤差

        return:
            muhat: 推定母平均
            sigma_mu: 推定母平均の不確かさ
    """
    w = 1.0 / sigma**2
    muhat = np.average(x,weights=w)
    sigma_mu = np.float64(np.sqrt(1.0/np.sum(w)))
    return muhat,sigma_mu

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

@dataclass
class EstimatedBackground:
    mean: float
    mean_err: float
    chi2: float
    reduced_chi2: float
    ndof: int

def estimate_background(
    fnu:NDArray64,
    fnu_err:NDArray64
):
    muhat,muhat_err = calculate_muhat(fnu,fnu_err)
    chi2 = calculate_chi2(muhat,fnu,fnu_err)
    ndof = len(fnu) - 1
    reduced_chi2 = calculate_reduced_chi2(chi2,ndof)
    return EstimatedBackground(
        mean = muhat,
        mean_err = muhat_err,
        chi2 = chi2,
        reduced_chi2 = reduced_chi2,
        ndof = ndof
    )
