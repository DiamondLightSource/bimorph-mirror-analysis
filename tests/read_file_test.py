from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bimorph_mirror_analysis.read_file import read_bluesky_plan_output


def test_read_raw_data(raw_data: pd.DataFrame, raw_data_pivoted: pd.DataFrame):
    with patch("bimorph_mirror_analysis.read_file.pd.read_csv") as mock_read_csv:
        mock_read_csv.return_value = raw_data
        pivoted, initial_voltages, increment = read_bluesky_plan_output("input_path")
        pd.testing.assert_frame_equal(pivoted, raw_data_pivoted)
        np.testing.assert_array_equal(initial_voltages, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_equal(increment, np.float64(100.0))
        mock_read_csv.assert_called()


@pytest.mark.xfail(
    reason="This test is expected to fail, the incrememnt should be 100, not 101"
)
def test_read_raw_data_fail(raw_data_pivoted: pd.DataFrame):
    with patch(
        "bimorph_mirror_analysis.read_file.read_bluesky_plan_output"
    ) as mock_read_bluesky_plan_output:
        mock_read_bluesky_plan_output.return_value = (
            raw_data_pivoted,
            np.array([0.0, 0.0, 0.0]),
            np.float64(101.0),
        )
        pivoted, initial_voltages, increment = mock_read_bluesky_plan_output()
        expected_output = pd.read_csv("tests/data/raw_data_pivoted.csv")  # type: ignore
        pd.testing.assert_frame_equal(pivoted, expected_output)
        np.testing.assert_array_equal(initial_voltages, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_equal(increment, np.float64(100.0))
