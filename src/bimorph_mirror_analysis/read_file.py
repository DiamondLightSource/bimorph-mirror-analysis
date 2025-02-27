from typing import TypedDict

import numpy as np
import pandas as pd


class Metadata(TypedDict):
    voltage_increment: float
    dimension: str
    num_slit_positions: int
    channels: int


def read_metadata(filepath: str) -> Metadata:
    """Read the metadata from the csv file

    Args:
        filepath: The path to the csv file to be read.

    Returns:
        The metadata from the csv file.
    """
    metadata: Metadata = {}  # type: ignore
    with open(filepath) as file:
        lines = file.readlines()
        for line in lines:
            if line[0] != "#":
                continue
            key = line.split(" ")[0][1:]  # explude the first character which is a #
            match key:
                case "voltage_increment":
                    metadata[key] = float(line.split(" ")[1])
                case "dimension":
                    metadata[key] = line.split(" ")[1].lower().strip()
                case "slit_positions":
                    metadata["num_slit_positions"] = int(line.split(" ")[1])
                case "channels":
                    metadata[key] = int(line.split(" ")[1])
                case _:
                    print("an error has occured when reading the csv metadata")

    return metadata


def read_bluesky_plan_output(
    filepath: str,
    baseline_voltage_scan_index: int = 0,
    detector_dimension: str | None = None,
) -> tuple[pd.DataFrame, np.typing.NDArray[np.float64], float, str, str]:
    """Read the csv file putput by the bluesky plan

    Reads the file and returns the dataframe with individual pecil beam scans as
    columns, the initial voltages and the voltage increment.

    Args:
        filepath: The path to the csv file to be read.
        baseline_voltage_scan_index: The scan number of the baseline voltage.

    Returns:
        A tuple containing the DataFrame, the initial voltages array and the voltage
        incrememnt.
    """
    metadata = read_metadata(filepath)

    data = pd.read_csv(filepath, comment="#")  # type: ignore
    data = data.apply(pd.to_numeric, errors="coerce")  # type: ignore

    voltage_cols = [col for col in data.columns if "voltage" in col]
    if baseline_voltage_scan_index >= 0:
        baseline_idx = baseline_voltage_scan_index
    else:
        baseline_idx = len(data) + baseline_voltage_scan_index - 1
    initial_voltages = data.loc[baseline_idx, voltage_cols].to_numpy()  # type: ignore
    # voltages from any other scan will have a change in the voltages
    voltage_increment = metadata["voltage_increment"]
    varied_slit_dimension = metadata["dimension"]
    slit_position_column = data.columns[
        [f"{varied_slit_dimension}_centre" in col for col in data.columns]
    ].to_list()[0]

    if detector_dimension is not None:
        assert detector_dimension in ["x", "y", "X", "Y"], "Invalid detector dimension"
        detector_column_name = f"Centroid{detector_dimension.upper()}"
    else:
        detector_column_name = f"Centroid{metadata['dimension'].upper()}"

    pivoted = pd.pivot_table(  # type: ignore
        data,
        values=detector_column_name,
        index=[slit_position_column],
        columns=["scan_index"],
    )
    pivoted.columns = ["pencil_beam_scan_" + str(col) for col in pivoted.columns]
    pivoted.reset_index(inplace=True)
    return (
        pivoted,
        initial_voltages,
        voltage_increment,
        slit_position_column,
        detector_column_name,
    )  # type: ignore
