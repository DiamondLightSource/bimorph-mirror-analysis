"""Interface for ``python -m bimorph_mirror_analysis``."""

import datetime

import numpy as np
import typer

from bimorph_mirror_analysis.app import run_server
from bimorph_mirror_analysis.maths import find_voltage_corrections
from bimorph_mirror_analysis.read_file import read_bluesky_plan_output

from . import __version__

__all__ = ["main"]

app = typer.Typer()


@app.command(name=None)
def calculate_voltages(
    file_path: str = typer.Argument(help="The path to the csv file to be read."),
    output_path: str | None = typer.Option(
        None,
        help="The path to save the output\
optimal voltages to, optional.",
    ),
    human_readable: str | None = typer.Option(
        None,
        help="The path to save the human readable pencil beam scan table. \
            If the --human-readable flag is not supplied, the table is not saved.",
    ),
):
    file_type = file_path.split(".")[-1]
    if human_readable is not None:
        pivoted, *_ = read_bluesky_plan_output(file_path)
        pivoted.to_csv(human_readable)
        print(f"The human-readable file has been written to {human_readable}")

    optimal_voltages = calculate_optimal_voltages(file_path)
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


def version_callback(value: bool):
    if value:
        typer.echo(f"Version: {__version__}")
        raise typer.Exit()


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


def calculate_optimal_voltages(file_path: str) -> np.typing.NDArray[np.float64]:
    pivoted, initial_voltages, increment = read_bluesky_plan_output(file_path)
    # numpy array of pencil beam scans
    data = pivoted[pivoted.columns[1:]].to_numpy()  # type: ignore

    voltage_adjustments = find_voltage_corrections(data, increment)  # type: ignore
    optimal_voltages = initial_voltages + voltage_adjustments
    return optimal_voltages  # type: ignore


app.command()


def server():
    run_server()


if __name__ == "__main__":
    app()
