from unittest.mock import patch

import numpy as np
import pandas as pd

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
        voltages = calculate_optimal_voltages("input_file")
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
