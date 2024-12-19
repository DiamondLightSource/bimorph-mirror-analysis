from unittest.mock import patch

import numpy as np
import pandas as pd
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


def test_calculate_optimal_voltages_mocked():
    with (
        patch(
            "bimorph_mirror_analysis.__main__.read_bluesky_plan_output"
        ) as mock_read_bluesky_plan_output,
        patch(
            "bimorph_mirror_analysis.__main__.find_voltages"
        ) as mock_calculate_voltages,
    ):
        # create dataframe as mock return
        data = {
            "slit_position_x": [0.0, 2.5, 5.0, 7.5, 10.0],
            "pencil_beam_scan_0": [0.902155, 0.974760, 0.628935, 0.776265, 0.507112],
            "pencil_beam_scan_1": [0.554659, 0.139067, 0.879844, 0.184194, 0.871104],
            "pencil_beam_scan_2": [0.723392, 0.290743, 0.137997, 0.796650, 0.603555],
            "pencil_beam_scan_3": [0.232850, 0.394227, 0.519578, 0.305536, 0.748342],
        }
        df = pd.DataFrame(data)  # type: ignore

        # set the mock return values
        mock_read_bluesky_plan_output.return_value = (
            df,
            np.array([0.0, 2.5, 5.0]),
            2.5,
        )
        mock_calculate_voltages.return_value = np.array([72.14, 50.98, 18.59])
        calculate_optimal_voltages("input_file")
        # assert mock was called
        mock_read_bluesky_plan_output.assert_called()
        mock_calculate_voltages.assert_called()
