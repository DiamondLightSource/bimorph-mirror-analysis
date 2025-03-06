"""Interface for ``python -m bimorph_mirror_analysis``."""

import datetime
import os

import numpy as np
import typer

from bimorph_mirror_analysis.maths import (
    check_voltages_fit_constraints,
    find_voltage_corrections,
    find_voltage_corrections_with_restraints,
)
from bimorph_mirror_analysis.plots import (
    InfluenceFunctionPlot,
    MirrorSurfacePlot,
    PencilBeamScanPlot,
)
from bimorph_mirror_analysis.read_file import read_bluesky_plan_output

from . import __version__

__all__ = ["main"]

app = typer.Typer()


@app.command(name=None, context_settings={"ignore_unknown_options": True})
def calculate_voltages(
    file_path: str = typer.Argument(help="The path to the csv file to be read."),
    voltage_range: tuple[int, int] = typer.Argument(
        help="The minimum and maximum values a voltage can take. expects two integers\
 separated by a space"
    ),
    max_consecutive_voltage_difference: int = typer.Argument(
        help="The maximum voltage difference allowed between two consecutive actuators\
on the bimorph mirror."
    ),
    baseline_voltage_scan: int = typer.Option(
        help="The index of the pencil beam scan which had no increment applied.",
        default=0,
    ),
    output_path: str | None = typer.Option(
        None,
        help="The path to save the output optimal voltages to, optional.",
    ),
    human_readable: str | None = typer.Option(
        None,
        help="The path to save the human readable pencil beam scan table. \
If the --human-readable flag is not supplied, the table is not saved.",
    ),
    slit_range: tuple[float, float] | None = typer.Option(
        None,
        help="The minimum and maximum\
 values for slit positions that should be considered when performing the analysis",
    ),
):
    file_type = file_path.split(".")[-1]
    if human_readable is not None:
        pivoted, *_ = read_bluesky_plan_output(file_path)
        pivoted.to_csv(human_readable)
        print(f"The human-readable file has been written to {human_readable}")

    optimal_voltages = calculate_optimal_voltages(
        file_path,
        voltage_range=voltage_range,
        max_consecutive_voltage_difference=max_consecutive_voltage_difference,
        baseline_voltage_scan=baseline_voltage_scan,
        slit_range=slit_range,
    )
    optimal_voltages = np.round(optimal_voltages, 2)
    date = datetime.datetime.now().date()

    if output_path is None:
        output_path = f"{file_path.replace(f'.{file_type}', '')}\
_optimal_voltages_{date}.csv"

    np.savetxt(
        output_path,
        optimal_voltages,
        fmt="%.2f",
    )
    print(f"The optimal voltages have been saved to {output_path}")
    print(
        f"The optimal voltages are: [{', '.join([str(i) for i in optimal_voltages])}]"
    )


def calculate_optimal_voltages(
    file_path: str,
    voltage_range: tuple[int, int],
    max_consecutive_voltage_difference: int,
    baseline_voltage_scan: int = 0,
    slit_range: tuple[float, float] | None = None,
) -> np.typing.NDArray[np.float64]:
    """Calculate the optimal voltages for the bimorph mirror actuators.

    Args:
        file_path: The path to the csv file to be read
        voltage_range: The minimum and maximum values a voltage can take
        max_consecutive_voltage_difference: The maximum voltage difference allowed\
 between two consecutive actuators
        baseline_voltage_scan: The index of the pencil beam scan which had no increment\
 applied

    Returns:
        The optimal voltages for the bimorph mirror actuators.
    """
    pivoted, initial_voltages, increment = read_bluesky_plan_output(file_path)

    if slit_range is not None:
        pivoted_in_range = pivoted[  # type: ignore
            (pivoted["slit_position_x"] >= slit_range[0])
            & (pivoted["slit_position_x"] <= slit_range[1])  # type: ignore
        ]
        data = pivoted_in_range[pivoted_in_range.columns[1:]].to_numpy()  # type: ignore

    else:
        data = pivoted[pivoted.columns[1:]].to_numpy()  # type: ignore

    voltage_adjustments = find_voltage_corrections(
        data,  # type: ignore
        increment,
        baseline_voltage_scan=baseline_voltage_scan,
    )
    optimal_voltages = initial_voltages + voltage_adjustments
    if check_voltages_fit_constraints(
        optimal_voltages, voltage_range, max_consecutive_voltage_difference
    ):
        return optimal_voltages

    else:
        print("The optimal voltages calculated with multiple linear regression are:")
        print(optimal_voltages)
        print(
            "However, these do not fit the constraints provided. Using an iterative\
 approach to find the optimal voltages which fit the constraints"
        )
        voltage_adjustments = find_voltage_corrections_with_restraints(
            data,  # type: ignore
            increment,
            initial_voltages,
            voltage_range,
            max_consecutive_voltage_difference,
            baseline_voltage_scan=baseline_voltage_scan,
        )
        optimal_voltages = initial_voltages + voltage_adjustments

        return optimal_voltages


def version_callback(value: bool):
    if value:
        typer.echo(f"Version: {__version__}")
        raise typer.Exit()


@app.command(name=None, context_settings={"ignore_unknown_options": True})
def generate_plots(
    file_path: str = typer.Argument(help="The path to the csv file to be read."),
    output_dir: str = typer.Argument(
        help="The directory to save the output plots to.",
    ),
    voltage_range: tuple[int, int] = typer.Argument(
        help="The minimum and maximum values a voltage can take."
    ),
    max_consecutive_voltage_difference: int = typer.Argument(
        help="The maximum voltage difference allowed between two consecutive actuators\
on the bimorph mirror."
    ),
    baseline_voltage_scan: int = typer.Option(
        help="The index of the pencil beam scan which had no increment applied.",
        default=0,
    ),
):
    # add trailing slash to output_dir if not present
    if output_dir[-1] != "/":
        output_dir += "/"

    # if dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pivoted, initial_voltages, increment = read_bluesky_plan_output(file_path)
    pencil_beam_scan_cols = [
        col for col in pivoted.columns if "pencil_beam_scan" in col
    ]
    slit_positions: np.typing.NDArray[np.float64] = pivoted[
        "slit_position_x"
    ].to_numpy()  # type: ignore

    for col in pencil_beam_scan_cols:
        i = int(col.split("_")[-1])
        plot = PencilBeamScanPlot(pivoted, i)
        plot.save_plot(output_dir + "pencil_beam_scan_" + str(i) + ".png")
    print(f"Pencil Beam Scan plots have been saved to {output_dir}")

    data: np.typing.NDArray[np.float64] = pivoted[pivoted.columns[1:]].to_numpy()  # type: ignore
    responses = np.diff(
        data, axis=1
    )  # calculate the response of each actuator by subtracting previous pencil beam
    interation_matrix = responses / increment  # response per unit charge

    for actuator_num in range(responses.shape[1]):
        centroids = interation_matrix[:, actuator_num]
        plot = InfluenceFunctionPlot(slit_positions, centroids, actuator_num)
        plot.save_plot(output_dir + f"actuator_{actuator_num}_influence_function.png")
    print(f"influence function plots have been saved to {output_dir}")

    # Add in predicted centroid positions using restrained and unrestrained voltage
    # corrections once the other prs are merged
    unrestrained_voltage_corrections = find_voltage_corrections(
        data, increment, baseline_voltage_scan=baseline_voltage_scan
    )
    restrained_voltage_corrections = find_voltage_corrections_with_restraints(
        data,
        increment,
        initial_voltages,
        voltage_range,
        max_consecutive_voltage_difference,
        baseline_voltage_scan=baseline_voltage_scan,
    )
    unrestrained_centroid_corrections = np.matmul(
        interation_matrix, unrestrained_voltage_corrections
    )
    restrained_centroid_corrections = np.matmul(
        interation_matrix, restrained_voltage_corrections
    )

    baseline_centroids = pivoted[f"pencil_beam_scan_{baseline_voltage_scan}"].to_numpy()  # type: ignore

    plot = MirrorSurfacePlot(
        slit_positions,
        baseline_centroids,  # type: ignore
        baseline_centroids + unrestrained_centroid_corrections,
        baseline_centroids + restrained_centroid_corrections,
    )
    plot.save_plot(output_dir + "mirror_surface_plot.png")
    print(f"The mirror surface plot has been saved to {output_dir}")


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the application's version and exit",
    ),
):
    pass


if __name__ == "__main__":
    app()
