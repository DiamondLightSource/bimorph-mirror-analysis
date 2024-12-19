from unittest.mock import patch

import numpy as np
import pandas as pd

from bimorph_mirror_analysis.__main__ import calculate_optimal_voltages


def test_calculate_optimal_voltages(raw_data: pd.DataFrame):
    with patch("bimorph_mirror_analysis.read_file.pd.read_csv") as mock_read_csv:
        mock_read_csv.return_value = raw_data
        voltages = calculate_optimal_voltages("input_file")
        voltages = np.round(voltages, 2)
        np.testing.assert_almost_equal(voltages, np.array([72.14, 50.98, 18.59]))


def test_calculate_optimal_voltages_mocked(raw_data_pivoted: pd.DataFrame):
    with (
        patch(
            "bimorph_mirror_analysis.__main__.read_bluesky_plan_output"
        ) as mock_read_bluesky_plan_output,
        patch(
            "bimorph_mirror_analysis.__main__.find_voltages"
        ) as mock_calculate_voltages,
    ):
        # set the mock return values
        mock_read_bluesky_plan_output.return_value = (
            raw_data_pivoted,
            np.array([0.0, 0.0, 0.0]),
            100,
        )
        mock_calculate_voltages.return_value = np.array([72.14, 50.98, 18.59])
        calculate_optimal_voltages("input_file")
        # assert mock was called
        mock_read_bluesky_plan_output.assert_called()
        mock_calculate_voltages.assert_called()
