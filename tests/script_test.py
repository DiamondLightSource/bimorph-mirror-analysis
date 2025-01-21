import numpy as np
import pytest

from bimorph_mirror_analysis.maths import (
    find_voltage_corrections,
    find_voltage_corrections_with_restraints,
)


@pytest.mark.parametrize(
    "actuator_data",
    [
        ["tests/data/8_actuator_data.txt", "tests/data/8_actuator_output.txt"],
        ["tests/data/16_actuator_data.txt", "tests/data/16_actuator_output.txt"],
    ],
    indirect=True,
)
def test_find_voltages_correct_output(
    actuator_data: tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
):
    data, expected_output = actuator_data
    v = -100
    np.testing.assert_almost_equal(
        find_voltage_corrections(data, v, baseline_voltage_scan=-1),
        expected_output,
        decimal=2,
    )


@pytest.mark.parametrize(
    "actuator_data",
    [
        ["tests/data/8_actuator_data.txt", "tests/data/8_actuator_output.txt"],
    ],
    indirect=True,
)
def test_find_voltage_corrections_index_error_throw(
    actuator_data: tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
):
    data, _ = actuator_data
    with pytest.raises(IndexError):
        find_voltage_corrections(data, -100, baseline_voltage_scan=12)


@pytest.mark.parametrize(
    "actuator_data",
    [
        ["tests/data/8_actuator_data.txt", "tests/data/8_actuator_output.txt"],
        ["tests/data/16_actuator_data.txt", "tests/data/16_actuator_output.txt"],
    ],
    indirect=True,
)
def test_find_voltage_corrections_with_restraints_correct_output(
    actuator_data: tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
):
    data, expected_output = actuator_data
    v = -100
    np.testing.assert_almost_equal(
        find_voltage_corrections_with_restraints(
            data, v, (-1000, 1000), 500, baseline_voltage_scan=-1
        ),
        expected_output,
        decimal=1,
    )
