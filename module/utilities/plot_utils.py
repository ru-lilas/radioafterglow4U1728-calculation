"""modules/plot/utilities.py

"""

from matplotlib.axes import Axes
from contextlib import contextmanager
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import TypeAlias, Literal

FloatPair:TypeAlias = tuple[float,float]
PlotScale = Literal["linear","log"]

@dataclass
class TicksConfigure:
    xlim:FloatPair
    ylim:FloatPair
    xscale:PlotScale
    yscale:PlotScale
    fontsize:int

@dataclass
class CurveConfigure:
    linestyle: str
    linewidth: int
    color: str
    label: str = ""
    zorder: int = 1

@dataclass
class ContourConfigure:
    linestyles: str
    linewidths: int
    colors: str
    levels: list[float]
    label: str = ""

@dataclass
class LabelConfigure:
    xlabel:str
    ylabel:str
    fontsize:int

@dataclass
class LegendConfigure:
    use: bool
    key: str
    label: str
    fontsize:int

@dataclass
class AnnotationConfigure:
    use: bool
    fontsize: int
    text: str
    pos:FloatPair=(1.00,1.01)
    ha:str="right"
    va:str="bottom"

def configure_tick(ax:Axes,config_tick:TicksConfigure):
    """軸の目盛およびスケール設定を行う。

    Parameters
    ----------
    ax : Axes
        設定対象のmatplotlib Axesオブジェクト。
    config_tick : models.TicksConfigure
        x/y軸の範囲、スケール、フォントサイズなどを保持する設定オブジェクト。

    Notes
    -----
    - x軸・y軸の範囲とスケールを設定する。
    - 主目盛・副目盛のスタイル（長さ・太さ・向き）を統一的に指定する。
    - x軸主目盛のラベルパディングを調整する。
    """
    ax.set_xlim(config_tick.xlim)
    ax.set_ylim(config_tick.ylim)
    ax.set_xscale(config_tick.xscale)
    ax.set_yscale(config_tick.yscale)
    ax.tick_params(axis="both",labelsize=config_tick.fontsize)
    ax.tick_params(axis="both",which="major", length=10, width=1, direction="in")
    ax.tick_params(axis="both",which="minor", length=8, width=1, direction="in")
    ax.tick_params(axis="x",which="major", pad=20)
    return

def configure_label(ax:Axes,config_label:LabelConfigure):
    """軸ラベルの設定を行う。

    Parameters
    ----------
    ax : Axes
        設定対象のmatplotlib Axesオブジェクト。
    config_label : models.LabelConfigure
        x/yラベルおよびフォントサイズを保持する設定オブジェクト。

    Notes
    -----
    LaTeX形式の文字列（raw文字列）としてラベルを設定する。
    """
    ax.set_xlabel(rf"{config_label.xlabel}",fontsize=config_label.fontsize)
    ax.set_ylabel(rf"{config_label.ylabel}",fontsize=config_label.fontsize)
    return

def annotation(
    ax:Axes,
    config:AnnotationConfigure,
):
    """Axes上にアノテーション文字列を描画する。

    Parameters
    ----------
    ax : Axes
        描画対象のAxes。
    annotconf : dict
        アノテーション設定辞書（少なくとも"fontsize"を含む）。
    text : str
        描画する文字列（通常はLaTeX形式）。
    pos : tuple of float, optional
        Axes座標系での描画位置（デフォルトは右上外側）。
    ha : str, optional
        水平方向の配置（"left", "center", "right"）。
    va : str, optional
        垂直方向の配置（"top", "center", "bottom"）。

    Notes
    -----
    - 座標はAxesの正規化座標系（0〜1）で指定される。
    - 主に固定パラメータの表示に使用される。
    """
    ax.text(
        *config.pos, config.text,
        fontsize=config.fontsize,
        transform=ax.transAxes,ha=config.ha,va=config.va
    )
