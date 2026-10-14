"""Chevalier contours with shared preparation and overlays."""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module import plot_utils
from module.chevalier import ChevalierContours
from module.chevalier_plotting import ChevalierPlotData, parse_args


def main():
    args = parse_args()
    data = ChevalierPlotData.from_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        fig, axes = data.config.layout.create()
        try:
            plotter = plot_utils.Plotter(axes.flat[0])
            for name, grid in (
                ("a_wind", data.grids.a_wind_grid(data.model_parameters)),
                ("beta_sh", data.grids.beta_sh_grid(data.model_parameters)),
            ):
                ChevalierContours.build(
                    data.plotregion, grid,
                    style=data.config.components[name].contour_style_config,
                    mask=data.invisible,
                ).plot(plotter)
            data.draw_boundaries(plotter)
            data.draw_observations(plotter)
            plotter.title(strings=data.title_strings(), fontsize=data.config.metadata.fontsize)
            pdf.savefig(fig)
        finally:
            plt.close(fig)
    print(f"Saved Chevalier contours: {args.output}", flush=True)

if __name__ == "__main__":
    main()

