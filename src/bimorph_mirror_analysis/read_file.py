import numpy as np
import pandas as pd


def read_bluesky_plan_output(
    filepath: str,
) -> tuple[pd.DataFrame, np.typing.NDArray[np.float64], float]:
    """Read the csv file putput by the bluesky plan

    Reads the file and returns the dataframe with individual pecil beam scans as
    columns, the initial voltages and the voltage increment.

    Args:
        filepath (str): The path to the csv file to be read.

    Returns:
        A tuple containing the DataFrame, the initial voltages array and the voltage
        incrememnt.
    """
    data = pd.read_csv(filepath)  # type: ignore
    data = data.apply(pd.to_numeric, errors="coerce")  # type: ignore

    voltage_cols = [col for col in data.columns if "voltage" in col]
    initial_voltages = data.loc[0, voltage_cols].to_numpy()  # type: ignore
    final_voltages = data.loc[len(data) - 1, voltage_cols].to_numpy()  # type: ignore

    voltage_increment = final_voltages[0] - initial_voltages[0]  # type: ignore

    pivoted = pd.pivot_table(  # type: ignore
        data,
        values="centroid_position_x",
        index=["slit_position_x"],
        columns=["pencil_beam_scan_number"],
    )
    pivoted.columns = ["pencil_beam_scan_" + str(col) for col in pivoted.columns]
    pivoted.reset_index(inplace=True)
    return pivoted, initial_voltages, voltage_increment  # type: ignore
