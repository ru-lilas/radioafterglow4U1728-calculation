"""One row, two columns: beta_sh and A_wind in the Chevalier plane."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from module import plot_utils
from module.chevalier_plotting import ChevalierPlotData, parse_args


def draw_parameter_map(fig, ax, data, name, grid):
    style = data.config.components[name].colormap_style_config
    z = grid.to_FloatGrid(style.colorbar.label.unit).values.copy()
    z[data.invisible.values] = np.nan
    plotter = plot_utils.Plotter(ax)
    # Keep the original model-grid topology and finite, unmasked coordinates.
    mesh = plotter.colormap(
        x=data.plotregion.x.values, y=data.plotregion.y.values,
        z=z, style=style,
    )
    plot_utils.ColorbarComponent(mesh, style.colorbar).draw(fig=fig, axes=[ax])
    data.draw_boundaries(plotter)
    data.draw_observations(plotter)
    return mesh


def main():
    args = parse_args()
    data = ChevalierPlotData.from_args(args)
    subplots = data.config.layout.subplots
    if (subplots.nrows, subplots.ncols) != (1, 2):
        raise ValueError("Chevalier colormap requires nrows: 1, ncols: 2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        fig, axes = data.config.layout.create()
        try:
            for ax, name, grid in (
                (axes[0, 0], "beta_sh", data.grids.beta_sh_grid(data.model_parameters)),
                (axes[0, 1], "a_wind", data.grids.a_wind_grid(data.model_parameters)),
            ):
                draw_parameter_map(fig, ax, data, name, grid)
            fig.suptitle(" | ".join(data.title_strings()), fontsize=data.config.metadata.fontsize)
            pdf.savefig(fig)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
