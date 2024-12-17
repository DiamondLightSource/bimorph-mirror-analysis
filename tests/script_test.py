import numpy as np
import pytest

from bimorph_mirror_analysis.maths import find_voltages


@pytest.mark.parametrize(
    ["input_path", "output_path"],
    [
        ["tests/data/8_actuator_data.txt", "tests/data/8_actuator_output.txt"],
        ["tests/data/16_actuator_data.txt", "tests/data/16_actuator_output.txt"],
    ],
)
def test_find_voltages_correct_output(input_path: str, output_path: str):
    data = np.loadtxt(input_path, delimiter=",")
    v = -100
    expected_output = np.loadtxt(output_path, delimiter=",")
    np.testing.assert_almost_equal(
        find_voltages(data, v, baseline_voltage_scan=-1), expected_output
    )


def test_find_voltages_index_error_throw():
    data = np.loadtxt("tests/data/8_actuator_data.txt", delimiter=",")
    with pytest.raises(IndexError):
        find_voltages(data, -100, baseline_voltage_scan=12)
