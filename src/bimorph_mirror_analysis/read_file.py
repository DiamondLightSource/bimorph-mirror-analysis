import numpy as np
import pandas as pd


def read_bluesky_plan_output(
    filepath: str,
    baseline_voltage_scan_index: int = 0,
) -> tuple[pd.DataFrame, np.typing.NDArray[np.float64], float]:
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
    data = pd.read_csv(filepath)  # type: ignore
    data = data.apply(pd.to_numeric, errors="coerce")  # type: ignore

    voltage_cols = [col for col in data.columns if "voltage" in col]
    if baseline_voltage_scan_index >= 0:
        baseline_idx = baseline_voltage_scan_index
    else:
        baseline_idx = len(data) + baseline_voltage_scan_index - 1
    initial_voltages = data.loc[baseline_idx, voltage_cols].to_numpy()  # type: ignore
    # voltages from any other scan will have a change in the voltages
    num_slit_positions = len(data) // len(voltage_cols)
    if baseline_voltage_scan_index == 0:
        other_idx = num_slit_positions + 1
    else:
        other_idx = 0
    other_voltages = data.loc[other_idx, voltage_cols].to_numpy()  # type: ignore

    diff_in_voltages: np.typing.NDArray[np.float64] = other_voltages - initial_voltages  # type: ignore
    max_diff, min_diff = np.max(diff_in_voltages), np.min(diff_in_voltages)  # type: ignore
    if abs(max_diff) > abs(min_diff):
        voltage_increment = max_diff
    else:
        voltage_increment = min_diff

    pivoted = pd.pivot_table(  # type: ignore
        data,
        values="centroid_position_x",
        index=["slit_position_x"],
        columns=["pencil_beam_scan_number"],
    )
    pivoted.columns = ["pencil_beam_scan_" + str(col) for col in pivoted.columns]
    pivoted.reset_index(inplace=True)
    return pivoted, initial_voltages, voltage_increment  # type: ignore
