from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest

from bimorph_mirror_analysis.read_file import (
    DetectorDimension,
    read_bluesky_plan_output,
    read_metadata,
)


@pytest.mark.parametrize(
    "detector_dimension",
    [
        "X",
        "Y",
    ],
)
def test_detector_dimension_option(
    detector_dimension: DetectorDimension, raw_data: pd.DataFrame
):
    with (
        patch("bimorph_mirror_analysis.read_file.pd.read_csv") as mock_read_csv,
        patch("bimorph_mirror_analysis.read_file.read_metadata") as mock_read_metadata,
    ):
        mock_read_csv.return_value = raw_data
        mock_read_metadata.return_value = {
            "voltage_increment": 100.0,
            "dimension": "x",
            "num_slit_positions": 21,
            "channels": 3,
        }
        (
            _,
            _,
            _,
            _,
            detector_column_name,
        ) = read_bluesky_plan_output(
            "input_path", detector_dimension=detector_dimension
        )
        assert detector_column_name == f"Centroid{detector_dimension}"


def test_read_raw_data(raw_data: pd.DataFrame, raw_data_pivoted: pd.DataFrame):
    with (
        patch("bimorph_mirror_analysis.read_file.pd.read_csv") as mock_read_csv,
        patch("bimorph_mirror_analysis.read_file.read_metadata") as mock_read_metadata,
    ):
        mock_read_csv.return_value = raw_data
        mock_read_metadata.return_value = {
            "voltage_increment": 100,
            "dimension": "x",
            "num_slit_positions": 1,
            "channels": 3,
        }
        pivoted, initial_voltages, increment, slit_position_column, detector_columns = (
            read_bluesky_plan_output("input_path")
        )
        pd.testing.assert_frame_equal(pivoted, raw_data_pivoted)
        np.testing.assert_array_equal(initial_voltages, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_equal(increment, np.float64(100.0))
        mock_read_csv.assert_called()
        assert slit_position_column == "slits-x_centre"
        assert detector_columns == "CentroidX"


def test_read_raw_data_baseline_last_scan(
    raw_data: pd.DataFrame, raw_data_pivoted: pd.DataFrame
):
    with (
        patch("bimorph_mirror_analysis.read_file.pd.read_csv") as mock_read_csv,
        patch("bimorph_mirror_analysis.read_file.read_metadata") as mock_read_metadata,
    ):
        mock_read_csv.return_value = raw_data
        mock_read_metadata.return_value = {
            "voltage_increment": -100,
            "dimension": "x",
            "num_slit_positions": 1,
            "channels": 3,
        }
        pivoted, initial_voltages, increment, *_ = read_bluesky_plan_output(
            "input_path", baseline_voltage_scan_index=-1
        )
        pd.testing.assert_frame_equal(pivoted, raw_data_pivoted)
        np.testing.assert_array_equal(initial_voltages, np.array([100, 100, 100]))
        np.testing.assert_equal(increment, np.float64(-100.0))
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
            "slits-x_centre",
            "CentroidX",
        )
        pivoted, initial_voltages, increment, *_ = mock_read_bluesky_plan_output()
        expected_output = pd.read_csv("tests/data/raw_data_pivoted.csv")  # type: ignore
        pd.testing.assert_frame_equal(pivoted, expected_output)
        np.testing.assert_array_equal(initial_voltages, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_equal(increment, np.float64(100.0))


@pytest.mark.parametrize(
    ["filename", "is_filepath"],
    [
        ["tests/data/example_bluesky_output.csv", True],
        [
            """#voltage_increment 200.0
#dimension x
#slit_positions 361
#channels 8
#this will break
slits-x_centre,slits-y_centre,CentroidX,CentroidY
0.0,0.0,0.0,0.0""",
            False,
        ],
    ],
)
def test_read_metadata(filename: str, is_filepath: str | None):
    if not is_filepath:
        with patch("builtins.open", mock_open(read_data=filename)) as mock_file:
            metadata = read_metadata(filename)
            mock_file.assert_called()
    else:
        metadata = read_metadata(filename)

    assert metadata == {
        "voltage_increment": 200.0,
        "dimension": "x",
        "num_slit_positions": 361,
        "channels": 8,
    }
