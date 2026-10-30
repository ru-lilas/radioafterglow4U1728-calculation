"""Draw f_e and p_index sweeps in one Figure using the panel API"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from scripts import plot_lightcurve as lc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f-e-sweep-index", type=Path, required=True)
    parser.add_argument("--p-index-sweep-index", type=Path, required=True)
    parser.add_argument("--minimum-chi-squared-path", type=Path, required=True)
    parser.add_argument("--model-parameter-table-path", type=Path, required=True)
    parser.add_argument("--candidate-lightcurves-path", type=Path, required=True)
    parser.add_argument("--binned-lightcurves-path", type=Path)
    parser.add_argument("--frequencies", type=float, nargs="+", default=[5.5, 9.0])
    parser.add_argument("--obslc", type=Path, required=True)
    parser.add_argument("--plotconfig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    f_e_source = lc.SweepSource(
        parameter="f_e",
        sweep_index=args.f_e_sweep_index,
        minimum_chi_squared_path=args.minimum_chi_squared_path,
        model_parameter_table_path=args.model_parameter_table_path,
        candidate_lightcurves_path=args.candidate_lightcurves_path,
        binned_lightcurves_path=args.binned_lightcurves_path,
    )
    p_index_source = replace(
        f_e_source, parameter="p_index", sweep_index=args.p_index_sweep_index,
    )
    # Change either source's result paths here if their RUN layouts differ
    settings = replace(
        lc.build_sweep_settings(), show_binned=args.binned_lightcurves_path is not None,
    )
    plotconf = lc.pu.PlotConfig.from_yaml(args.plotconfig)
    plotconf = replace(
        plotconf,
        layout=replace(
            plotconf.layout,
            figure=replace(plotconf.layout.figure, figsize=(24.0, 8.0)),
        ),
    )
    # 既存YAMLの2列目が9 GHz用の軸設定である場合
    axes_9ghz = plotconf.layout.axes.for_subplot(0, 1)

    panels = (
        lc.SweepPanelSpec(
            row=0,
            col=0,
            source=f_e_source,
            frequency=lc.QuantityData(9.0, "GHz"),
            axes=axes_9ghz,
            settings=settings,
        ),
        lc.SweepPanelSpec(
            row=0,
            col=1,
            source=p_index_source,
            frequency=lc.QuantityData(9.0, "GHz"),
            axes=axes_9ghz,
            settings=settings,
        ),
    )
    # To edit one panel, replace that instance before passing the tuple below
    lc.draw_sweep(
        panels,
        obslc=args.obslc,
        plotconf=plotconf,
        output=args.output,
        plotconfig_path=args.plotconfig,
    )
    print(f"Saved lightcurves: {args.output}", flush=True)


if __name__ == "__main__":
    main()
