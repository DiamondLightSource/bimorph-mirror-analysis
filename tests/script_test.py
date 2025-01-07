import numpy as np
import pytest

from bimorph_mirror_analysis.maths import find_voltages


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
        find_voltages(data, v, baseline_voltage_scan=-1), expected_output
    )


def test_find_voltages_index_error_throw():
    data = np.loadtxt("tests/data/8_actuator_data.txt", delimiter=",")
    with pytest.raises(IndexError):
        find_voltages(data, -100, baseline_voltage_scan=12)
