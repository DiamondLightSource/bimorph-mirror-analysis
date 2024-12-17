import numpy as np
import pandas as pd
import pytest

from bimorph_mirror_analysis.read_file import read_bluesky_plan_output


@pytest.mark.parametrize(
    ["input_path", "output_path"],
    [
        ["tests/data/raw_data.csv", "tests/data/raw_data_pivoted.csv"],
    ],
)
def test_read_raw_data(input_path: str, output_path: str):
    pivoted, initial_voltages, increment = read_bluesky_plan_output(input_path)
    expected_output = pd.read_csv(output_path)  # type: ignore
    pd.testing.assert_frame_equal(pivoted, expected_output)
    np.testing.assert_array_equal(initial_voltages, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_equal(increment, np.float64(100.0))


@pytest.mark.xfail(
    reason="This test is expected to fail, the incrememnt should be 100, not 101"
)
def test_read_raw_data_fail():
    pivoted, initial_voltages, increment = read_bluesky_plan_output(
        "tests/data/raw_data.csv"
    )
    expected_output = pd.read_csv("tests/data/raw_data_pivoted.csv")  # type: ignore
    pd.testing.assert_frame_equal(pivoted, expected_output)
    np.testing.assert_array_equal(initial_voltages, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_equal(increment, np.float64(101.0))
