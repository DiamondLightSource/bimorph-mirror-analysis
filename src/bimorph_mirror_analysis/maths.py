from collections.abc import Callable
from typing import TypedDict

import numpy as np
from scipy.optimize import minimize


def find_voltage_corrections(
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
                  {-1 * data.shape[1]} and {data.shape[1] - 1}"
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

    voltage_corrections: np.typing.NDArray[np.float64] = np.matmul(
        interation_matrix_inv, Y
    )  # calculate the voltage required to move the centroid to the target position

    return np.round(voltage_corrections[1:], decimals=2)  # return the voltages


def objective_function(
    voltages: np.typing.NDArray[np.float64],
    coefficients: np.typing.NDArray[np.float64],
    targets: np.typing.NDArray[np.float64],
) -> float:
    """Least-Squares based objective function

    Given a set of values for the voltages and their influence function coefficients
    for each slit position, along with the set of target centroid adjustments, calulate
    the sum of squared errors across all slit positions.

    Args:
        voltages: A list of values for the voltages
        coefficients: A 2D array of coefficients for each voltage, where rows are the
            slit positions and columns are the different actuators
        targets: A list of target values for each equation (row in the coefficients
            matrix)

    Returns:
        The sum of squared errors for the set of values of the voltages.

    ---

    The code below outlines what the function does in a readable (but slower) way:
    sum = 0
    for row_num in range(len(coefficients)):
        f = np.sum([coefficients[row_num][i]*voltages[i] for i in range(len(voltages))])
        f -= targets[row_num]
        sum+=f**2

    return sum
    """
    # assert the inputs are the correct shape
    assert len(voltages) == coefficients.shape[1], (
        f"Number of voltages: {len(voltages)}, Number of coefficients: \
{coefficients.shape[1]}\nThe number of voltages provided must match the number of\
columns in the coefficients matrix"
    )
    assert coefficients.shape[0] == len(targets), (
        f"Number of coefficients: {coefficients.shape[0]},\
    Number of target centroid positions: {len(targets)}\
        \nThe number of rows in the coefficients matrix must match the number of target\
centroid positions"
    )

    return np.sum((np.matmul(coefficients, voltages) - targets) ** 2)


def find_voltage_corrections_with_restraints(
    data: np.typing.NDArray[np.float64],
    v: float,
    voltage_range: tuple[int, int],
    max_consecutive_voltage_difference: int,
    baseline_voltage_scan: int = 0,
) -> np.typing.NDArray[np.float64]:
    """Calculate voltage corrections to apply to bimorph.

    Given a matrix of beamline centroid data, with columns of beamline scans at
    different actuator voltages and rows of slit positions, calculate the necessary
    voltages corrections to achive the target centroid position. Uses the SLSQP
    algorithm to optimise the voltages.

    Args:
        data: A matrix of beamline centroid data, with rows of different slit positions
            and columns of pencil beam scans at different actuator voltages
        v: The voltage increment applied to the actuators between pencil beam scans
        voltage_range: The minimum and maximum values a voltage can take.
        max_consecutive_voltage_difference: The maximum voltage difference between two
            consecutive actuators on the bimorph mirror.
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
                  {-1 * data.shape[1]} and {data.shape[1] - 1}"
        )

    responses = np.diff(
        data, axis=1
    )  # calculate the response of each actuator by subtracting previous pencil beam

    interation_matrix = responses / v  # response per unit charge
    # add columns of 1's to the left of H
    interation_matrix = np.hstack(
        (np.ones((interation_matrix.shape[0], 1)), interation_matrix)
    )
    baseline_voltage_beamline_positions = data[:, baseline_voltage_scan]

    target = np.mean(baseline_voltage_beamline_positions)
    Y = target - baseline_voltage_beamline_positions

    # set initial guess voltages to all 1s
    initial_guess = np.ones(interation_matrix.shape[1])

    bounds = [voltage_range for _ in range(interation_matrix.shape[1])]

    class Constraint(TypedDict):
        type: str
        fun: Callable[[np.typing.NDArray[np.float64]], float]

    # build list of contraints objects
    constraints: list[Constraint] = []
    for i in range(interation_matrix.shape[1] - 1):

        def func(
            voltages: np.typing.NDArray[np.float64],
            max_diff: int = max_consecutive_voltage_difference,
            idx: int = i,
        ) -> float:
            return max_diff - abs(voltages[idx] - voltages[idx + 1])

        constraints.append({"type": "ineq", "fun": func})
    # minimise the objective function
    result = minimize(
        objective_function,
        initial_guess,
        args=(interation_matrix, Y),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,  # type: ignore
        options={"maxiter": 3 * 10**5},
    )

    return np.round(result.x[1:], decimals=2)  # first item is not a voltage
