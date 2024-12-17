import numpy as np
import pytest

from bimorph_mirror_analysis.__main__ import calculate_optimal_voltages


@pytest.mark.parametrize(
    ["input_file", "expected_voltages"],
    [["tests/data/raw_data.csv", np.array([72.14, 50.98, 18.59])]],
)
def test_calculate_optimal_voltages(
    input_file: str, expected_voltages: np.typing.NDArray[np.float64]
):
    voltages = calculate_optimal_voltages(input_file)
    voltages = np.round(voltages, 2)
    np.testing.assert_almost_equal(voltages, expected_voltages)
