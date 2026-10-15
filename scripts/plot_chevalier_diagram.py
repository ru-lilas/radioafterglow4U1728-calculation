"""Render one RUN through the shared Axes-based contour implementation"""
from module import plot_utils
from module.chevalier_plotting import ChevalierInputPaths, ChevalierPlotData, draw_contour_figure, parse_args, write_single_pdf


def main() -> None:
    args = parse_args()
    paths = ChevalierInputPaths.from_args(args)
    config = plot_utils.PlotConfig.from_yaml(args.plotconfig)
    data = ChevalierPlotData.from_paths(paths, config)
    write_single_pdf(args.output, lambda: draw_contour_figure(data), sources=(*paths.sources(), args.plotconfig))


if __name__ == "__main__":
    main()
