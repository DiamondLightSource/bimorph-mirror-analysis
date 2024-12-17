"""Interface for ``python -m bimorph_mirror_analysis``."""

import datetime
from argparse import ArgumentParser
from collections.abc import Sequence

import numpy as np

from bimorph_mirror_analysis.maths import find_voltages
from bimorph_mirror_analysis.read_file import read_bluesky_plan_output

from . import __version__

__all__ = ["main"]


def main(args: Sequence[str] | None = None) -> None:
    """Argument parser for the CLI."""
    parser = ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the file containing the output of the Bluesky plan.",
    )
    a = parser.parse_args(args)
    file_path = a.file_path
    file_type = a.file_path.split(".")[-1]
    optimal_voltages = calculate_optimal_voltages(file_path)
    optimal_voltages = np.round(optimal_voltages, 2)
    date = datetime.datetime.now().date()
    np.savetxt(
        f"{a.file_path.replace(f'.{file_type}', '')}_optimal_voltages_{date}.csv",
        optimal_voltages,
        fmt="%.2f",
    )
    print(
        f"The optimal voltages are: [{', '.join([str(i) for i in optimal_voltages])}]"
    )


# implement this into main
def calculate_optimal_voltages(file_path: str) -> np.typing.NDArray[np.float64]:
    pivoted, initial_voltages, increment = read_bluesky_plan_output(file_path)
    # numpy array of pencil beam scans
    data = pivoted[pivoted.columns[1:]].to_numpy()  # type: ignore

    voltage_adjustments = find_voltages(data, increment)  # type: ignore
    optimal_voltages = initial_voltages + voltage_adjustments
    return optimal_voltages  # type: ignore


if __name__ == "__main__":
    main()
