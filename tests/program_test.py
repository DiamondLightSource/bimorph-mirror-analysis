from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bimorph_mirror_analysis.__main__ import calculate_optimal_voltages
from bimorph_mirror_analysis.maths import find_voltage_corrections


def test_calculate_optimal_voltages_mocked(raw_data_pivoted: pd.DataFrame):
    with (
        patch(
            "bimorph_mirror_analysis.__main__.read_bluesky_plan_output"
        ) as mock_read_bluesky_plan_output,
        patch(
            "bimorph_mirror_analysis.__main__.find_voltage_corrections"
        ) as mock_find_voltage_corrections,
    ):
        # set the mock return values
        mock_read_bluesky_plan_output.return_value = (
            raw_data_pivoted,
            np.array([0.0, 0.0, 0.0]),
            100,
        )
        mock_find_voltage_corrections.side_effect = find_voltage_corrections
        voltages = calculate_optimal_voltages("input_file", (-1000, 1000), 500)
        voltages = np.round(voltages, 2)
        # assert correct voltages calculated
        np.testing.assert_almost_equal(voltages, np.array([72.14, 50.98, 18.59]))

        # assert mock was called
        mock_read_bluesky_plan_output.assert_called()
        mock_read_bluesky_plan_output.assert_called_with("input_file")
        mock_find_voltage_corrections.assert_called()
        expected_data: np.typing.NDArray[np.float64] = raw_data_pivoted[
            raw_data_pivoted.columns[1:]
        ].to_numpy()  # type: ignore
        np.testing.assert_array_equal(
            mock_find_voltage_corrections.call_args[0][0], expected_data
        )  # type: ignore
        np.testing.assert_almost_equal(
            mock_find_voltage_corrections.call_args[0][1], 100
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
def test_calculate_optimal_voltages(
    actuator_data: tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ],
):
    with patch(
        "bimorph_mirror_analysis.__main__.read_bluesky_plan_output"
    ) as mock_read_bluesky_plan_output:
        data, expected_corrections, initial_voltages = actuator_data
        mock_read_bluesky_plan_output.return_value = (
            pd.DataFrame(
                np.hstack(
                    [np.ones((data.shape[0], 1)), data]
                )  # add blank column in place of slit position
            ),
            initial_voltages,
            -100,
        )
        voltages = calculate_optimal_voltages(
            "data", (-1000, 1000), 500, baseline_voltage_scan=-1
        )

        # assert correct voltages calculated
        voltages = np.round(voltages, 2)
        np.testing.assert_almost_equal(
            voltages, initial_voltages + expected_corrections, decimal=2
        )
