import numpy as np


def find_voltages(
    data: np.typing.NDArray[np.float64],
    v: float,
    baseline_voltage_scan: int = 0,
) -> np.typing.NDArray[np.float64]:
    """Calculate voltage corrections to apply to bimorph.

    Given a matrix of beamline centroid data, with columns of beamline scans at
    different actuator voltages and rows of slit positions, calculate the necessary
    voltages corrections to achive the target centroid position.

    Args:
        data: A matrix of beamline centroid data, with rows of different slit positions
            and columns of pencil beam scans at different actuator voltages
        v: The voltage increment applied to the actuators between pencil beam scans
        baseline_voltage_scan: The pencil beam scan to use as the baseline for the
            centroid calculation. 0 is the first scan, 1 is the second scan, etc.
            -1 can be used for the last scan and -2 for the second to last scan etc.

    Returns:
        An array of voltage corrections required to move the centroid of each pencil
        beam scan to the target position.
    """

    if baseline_voltage_scan < -data.shape[1] or baseline_voltage_scan >= data.shape[1]:
        raise IndexError(
            f"baseline_voltage_scan is out of range, it must be between\
                  {-1*data.shape[1]} and {data.shape[1]-1}"
        )

    responses = np.diff(
        data, axis=1
    )  # calculate the response of each actuator by subtracting previous pencil beam

    interation_matrix = responses / v  # response per unit charge
    # add columns of 1's to the left of H
    interation_matrix = np.hstack(
        (np.ones((interation_matrix.shape[0], 1)), interation_matrix)
    )
    # calculate the Moore-Penrose pseudo inverse of H
    interation_matrix_inv = np.linalg.pinv(interation_matrix)

    baseline_voltage_beamline_positions = data[:, baseline_voltage_scan]

    target = np.mean(baseline_voltage_beamline_positions)
    Y = target - baseline_voltage_beamline_positions

    voltage_corrections = np.matmul(
        interation_matrix_inv, Y
    )  # calculate the voltage required to move the centroid to the target position

    return voltage_corrections[1:]  # return the voltages
