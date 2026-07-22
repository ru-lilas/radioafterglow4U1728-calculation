from enum import StrEnum, auto

class KeyNames(StrEnum):
    T = auto()
    T_ERR = auto()
    T_UNIT = auto()
    FNU = auto()
    FNU_NET = auto()
    FNU_ERR = auto()
    FNU_UNIT = auto()
    FNU_WITH_BG = auto()
    NU = auto()
    NU_UNIT = auto()
    D_VALUE = auto()
    D_UNIT = auto()
    MICROSCOPIC_PARAMETERS = auto()
    PHI_THETA = auto()
    PHI_UNIT = auto()
    DISTANCE = auto()
    LNU_UNIT = auto()
    BETA_SH = auto()
    A_WIND = auto()
    LNU_THETA = auto()
    TAU_THETA = auto()
    CHI2 = auto()
    REDUCED_CHI2 = auto()
    EPS_B = auto()
    EPS_TH = auto()
    PHI = auto()
    DOPPLER_DELTA = auto()

class PlotConfigNames(StrEnum):
    FIGSIZE = auto()

class SamplingConfigNames(StrEnum):
    T_MIN = auto()
    T_MAX = auto()
    T_UNIT = auto()

class EstimationConfigNames(StrEnum):
    OBS_T_WINDOW = auto()
    MIN = auto()
    MAX = auto()
    UNIT = auto()

class ChevalierContourNames(StrEnum):
    PHI_PEAK = auto()
    FNU_NET_PEAK = auto()

class Chi2Columns(StrEnum):
    IDX = auto()
    CHI2 = auto()
    DELTA_CHI2 = auto()
    NPARAM = auto()
    NSAMPLE = auto()
    NDOF = auto()
    P_VALUE = auto()
    REJECT = auto()
    SIGMA = auto()

class LightcurveColumns(StrEnum):
    NU = auto()
    T = auto()
    FNU = auto()
    FNU_ERR = auto()
    BG = auto()
    BG_ERR = auto()
    FNU_NET = auto()
    FNU_NET_ERR = auto()

class FileNames(StrEnum):
    SAMPLING = auto()
    CHI2_ESTIMATED_PARAMETERS = auto()
    CHI2TEST_SUMMARY = auto()

class FileExtension(StrEnum):
    YAML = auto()
    CSV = auto()
    PDF = auto()
