from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bimorph_mirror_analysis.app import (
    DataDict,
    calculate_optimal_voltages,
    calculate_voltages,
    change_table,
    download_data,
    export_data_as_csv,
    make_download_button_visible,
    read_file,
    run_server,
)
from bimorph_mirror_analysis.maths import check_voltages_fit_constraints


def test_read_file_no_exception(
    raw_data: pd.DataFrame, data_dict: DataDict, encoded_file: str
):
    with (
        patch("bimorph_mirror_analysis.app.pd.read_csv") as mock_read_csv,
        patch(
            "bimorph_mirror_analysis.app.read_bluesky_plan_output"
        ) as mock_read_blue_sky_plan_output,
        patch("bimorph_mirror_analysis.app.base64.b64decode") as mock_b64decode,
        patch("bimorph_mirror_analysis.app.io.StringIO") as mock_string_io,
    ):
        mock_read_csv.return_value = raw_data
        mock_read_blue_sky_plan_output.return_value = (
            pd.DataFrame(data_dict["pivoted_data_dict"]),  # type: ignore
            data_dict["initial_voltages"],
            data_dict["increment"],
        )

        output = read_file(encoded_file, data_dict["filename"])

        mock_b64decode.assert_called_once_with(encoded_file.split(",")[-1])
        mock_string_io.assert_called_with(mock_b64decode.return_value.decode())
        assert mock_string_io.call_count == 2
        mock_read_csv.assert_called_once_with(mock_string_io.return_value)
        mock_read_blue_sky_plan_output.assert_called_once_with(
            mock_string_io.return_value
        )
        assert output == (
            data_dict,
            data_dict["filename"],
            {"display": "block"},
            {"display": "block"},
        )


def test_read_file_with_exception(data_dict: DataDict):
    output = read_file("encoded_file", data_dict["filename"])

    blank_data_dict: DataDict = {
        "raw_data_dict": {},
        "pivoted_data_dict": {},
        "initial_voltages": np.array([0]),
        "increment": 0,
        "filename": "",
    }
    expected: tuple[DataDict, str, dict[str, str], dict[str, str]] = (
        blank_data_dict,
        "There was an error processing this file: " + data_dict["filename"],
        {"display": "block"},
        {"display": "none"},
    )
    assert output == expected


@pytest.mark.parametrize(
    "ag_grid_selector_value",
    [
        "Raw Data",
        "Pivoted Data",
        "Error",
    ],
)
def test_change_table(data_dict: DataDict, ag_grid_selector_value: str):
    output = change_table(data_dict, ag_grid_selector_value, {"display": "block"})
    if ag_grid_selector_value == "Raw Data":
        df = pd.DataFrame(data_dict["raw_data_dict"])
        columns = [{"headerName": col, "field": col} for col in df.columns]
        data: dict[str, str] = df.to_dict("records")  # type: ignore
        pivot_data_style = {"display": "none"}
        expected = (
            columns,
            data,
            {"fileName": f"{data_dict['filename'].split('.')[0]}_pivoted.csv"},
            pivot_data_style,
        )

    elif ag_grid_selector_value == "Pivoted Data":
        df = pd.DataFrame(data_dict["pivoted_data_dict"])
        pivot_data_style = {"display": "block", "margin": "auto"}
        columns = [{"headerName": col, "field": col} for col in df.columns]
        data: dict[str, str] = df.to_dict("records")  # type: ignore
        expected = (
            columns,
            data,
            {"fileName": f"{data_dict['filename'].split('.')[0]}_pivoted.csv"},
            pivot_data_style,
        )
    elif ag_grid_selector_value == "Error":
        pivot_data_style = {"display": "block", "margin": "auto"}
        expected = ([{}], {}, {}, pivot_data_style)  # type: ignore

    else:
        raise ValueError(
            f"value {ag_grid_selector_value} for ag_grid_selector_value parameter is \
not covered by test"
        )

    assert output == expected


@pytest.mark.parametrize("n_clicks", [1, None])
def test_export_data_as_csv(n_clicks: int | None):
    output = export_data_as_csv(n_clicks)
    if n_clicks:  # only when int
        assert output  # should be true
    else:  # when Nnoe
        assert not output  # should be false


def test_calculate_optimal_voltages(data_dict: DataDict):
    voltages = calculate_optimal_voltages(data_dict, (-1000, 1000), 500, 0)
    voltages = np.round(voltages, 2)
    # assert correct voltages calculated
    np.testing.assert_almost_equal(voltages, np.array([72.14, 50.98, 18.59]))


def test_calculate_optimal_voltages_with_restraints(data_dict: DataDict):
    voltages = calculate_optimal_voltages(data_dict, (-1000, 1000), 10, 0)
    voltages = np.round(voltages, 2)
    # assert correct voltages calculated
    assert check_voltages_fit_constraints(voltages, (-1000, 1000), 10)


def test_calculate_voltages(data_dict: DataDict):
    output = calculate_voltages(1, data_dict, -1000, 1000, 500, 0)

    assert output == "[72.14, 50.98, 18.59]"

    # test that if datadict is not assigned properly, the function returns an empty
    # string
    output2 = calculate_voltages(2, [2], -1000, 1000, 500, 0)  # type: ignore
    assert output2 == ""


def test_make_download_button_visible(data_dict: DataDict):
    output = make_download_button_visible(1, data_dict)
    assert output == {
        "display": "block",
        "justify-content": "center",
        "margin-top": "40px",
    }

    # test that if datadict is not assigned properly, the function returns
    # display: none to the style
    output2 = make_download_button_visible(1, ["data_dict"])  # type: ignore
    assert output2 == {"display": "none"}


def test_download_data(data_dict: DataDict):
    output = download_data(1, "[72.14, 50.98, 18.59]", data_dict)

    assert output == {
        "base64": False,
        "content": ",".join([str(i) for i in [72.14, 50.98, 18.59]]),
        "filename": f"{data_dict['filename'].split('.')[0]}_optimal_voltages.csv",
    }


def test_run_server():
    with patch("bimorph_mirror_analysis.app.Dash.run") as mock_run:
        run_server()
        mock_run.assert_called_once()
