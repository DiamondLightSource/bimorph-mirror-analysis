import numpy as np
import pytest

from bimorph_mirror_analysis.maths import (
    check_voltages_fit_constraints,
    find_voltage_corrections,
    find_voltage_corrections_with_restraints,
)


@pytest.mark.parametrize(
    "actuator_data",
    [
        [
            "tests/data/8_actuator_data.txt",
            "tests/data/8_actuator_output.txt",
            "tests/data/8_actuator_initial_voltages.txt",
        ],
        [
            "tests/data/16_actuator_data.txt",
            "tests/data/16_actuator_output.txt",
            "tests/data/16_actuator_initial_voltages.txt",
        ],
    ],
    indirect=True,
)
def test_find_voltages_correct_output(
    actuator_data: tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ],
):
    data, expected_corrections, _ = actuator_data
    v = -100
    np.testing.assert_almost_equal(
        find_voltage_corrections(data, v, baseline_voltage_scan=-1),
        expected_corrections,
        decimal=2,
    )


@pytest.mark.parametrize(
    "actuator_data",
    [
        [
            "tests/data/8_actuator_data.txt",
            "tests/data/8_actuator_output.txt",
            "tests/data/8_actuator_initial_voltages.txt",
        ],
    ],
    indirect=True,
)
def test_find_voltage_corrections_index_error_throw(
    actuator_data: tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ],
):
    data, *_ = actuator_data
    with pytest.raises(IndexError):
        find_voltage_corrections(data, -100, baseline_voltage_scan=12)


@pytest.mark.parametrize(
    "actuator_data",
    [
        [
            "tests/data/8_actuator_data.txt",
            "tests/data/8_actuator_output.txt",
            "tests/data/8_actuator_initial_voltages.txt",
        ],
        [
            "tests/data/16_actuator_data.txt",
            "tests/data/16_actuator_output.txt",
            "tests/data/16_actuator_initial_voltages.txt",
        ],
    ],
    indirect=True,
)
def test_find_voltage_corrections_with_restraints_correct_output(
    actuator_data: tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ],
):
    data, expected_corrections, _ = actuator_data
    v = -100
    np.testing.assert_almost_equal(
        find_voltage_corrections_with_restraints(
            data, v, (-1000, 1000), 500, baseline_voltage_scan=-1
        ),
        expected_corrections,
        decimal=1,
    )


@pytest.mark.parametrize(
    "actuator_data",
    [
        [
            "tests/data/8_actuator_data.txt",
            "tests/data/8_actuator_output.txt",
            "tests/data/8_actuator_initial_voltages.txt",
        ],
    ],
    indirect=True,
)
def test_find_voltage_corrections_with_restraints_index_error_throw(
    actuator_data: tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ],
):
    data, *_ = actuator_data
    with pytest.raises(IndexError):
        find_voltage_corrections_with_restraints(
            data, -100, (-1000, 1000), 500, baseline_voltage_scan=12
        )


@pytest.mark.parametrize(
    "voltages, voltage_range, max_diff, expected",
    [
        (np.array([0, 0, 0]), (-1000, 1000), 500, True),  # np arrays should work
        ([-1000, -500, 0, 500, 1000], (-1000, 1000), 500, True),
        ([0, 0, 501], (-1000, 1000), 500, False),  # diff too big
        ([1000, 1000, 1001], (-1000, 1000), 500, False),  # value out of range
    ],
)
def test_check_voltages_fit_constraints(
    voltages: np.typing.NDArray[np.float64],
    voltage_range: tuple[int, int],
    max_diff: int,
    expected: bool,
):
    assert check_voltages_fit_constraints(voltages, voltage_range, max_diff) == expected
