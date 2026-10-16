"""Render one RUN through the shared Axes-based contour implementation"""
from module import plot_utils
from module.chevalier_plotting import (
    ChevalierInputPaths, ChevalierPlotData, ContourComponent,
    draw_contour_figure, parse_args, write_single_pdf,
)


def build_contour_components() -> tuple[ContourComponent, ...]:
    """Edit single-RUN contour appearance and level policy here, independently of YAML."""
    return (
        ContourComponent("a_wind", plot_utils.ContourStyleConfig(
            unit="g cm-1",
            lines=plot_utils.ContourLineStyle(colors="#84919B", linestyles="-", linewidths=1.0),
            levels=plot_utils.ContourLevelConfig(
                scale="log", values=[1e4, 1e5, 1e6, 1e7, 1e8, 1e9]),
            label=plot_utils.ContourLabelStyle(inline=True, inline_spacing=5, fontsize=16),
            label_format=plot_utils.QuantityLabel(r"A_{\mathrm{w}}", ".0e", r"\mathrm{g\,cm^{-1}}"),
        )),
        ContourComponent("beta_sh", plot_utils.ContourStyleConfig(
            unit="",
            lines=plot_utils.ContourLineStyle(colors="#84919B", linestyles="--", linewidths=1.0),
            levels=plot_utils.ContourLevelConfig(
                scale="linear", values=[0.20, 0.30, 0.40, 0.50, 0.60]),
            label=plot_utils.ContourLabelStyle(inline=True, inline_spacing=5, fontsize=16),
            label_format=plot_utils.QuantityLabel(r"\beta_{\mathrm{sh}}", ".2f", ""),
        )),
    )


def main() -> None:
    args = parse_args()
    paths = ChevalierInputPaths.from_args(args)
    config = plot_utils.PlotConfig.from_yaml(args.plotconfig)
    data = ChevalierPlotData.from_paths(paths, config)
    components = build_contour_components()
    write_single_pdf(args.output, lambda: draw_contour_figure(data, components=components),
                     sources=(*paths.sources(), args.plotconfig))


if __name__ == "__main__":
    main()
