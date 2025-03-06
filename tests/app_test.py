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
    find_voltage_corrections_with_restraints,
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
    "ag_grid_selector_value, data_key, pivot_data_style",
    [
        ("Raw Data", "raw_data_dict", {"display": "none"}),
        ("Pivoted Data", "pivoted_data_dict", {"display": "block", "margin": "auto"}),
        ("Error", None, {"display": "block", "margin": "auto"}),
    ],
)
def test_change_table(
    data_dict: DataDict,
    ag_grid_selector_value: str,
    data_key: str,
    pivot_data_style: dict[str, str],
):
    if data_key:
        df = pd.DataFrame(data_dict.get(data_key))
        columns = [{"headerName": col, "field": col} for col in df.columns]
        data: dict[str, str] = df.to_dict("records")  # type: ignore
        expected = (
            columns,
            data,
            {"fileName": f"{data_dict['filename'].split('.')[0]}_pivoted.csv"},
            pivot_data_style,
        )
    else:  # "Error" case
        expected = ([{}], {}, {}, pivot_data_style)  # type: ignore

    output = change_table(data_dict, ag_grid_selector_value, {"display": "block"})
    assert output == expected


@pytest.mark.parametrize("n_clicks, expected", [(1, True), (None, False)])
def test_export_data_as_csv(n_clicks: int | None, expected: bool):
    assert export_data_as_csv(n_clicks) == expected


def test_calculate_optimal_voltages(data_dict: DataDict):
    voltages = calculate_optimal_voltages(data_dict, (-1000, 1000), 500, 0)
    voltages = np.round(voltages, 2)
    # assert correct voltages calculated
    np.testing.assert_almost_equal(voltages, np.array([72.14, 50.98, 18.59]))


def test_calculate_optimal_voltages_with_restraints(data_dict: DataDict):
    with (
        patch(
            "bimorph_mirror_analysis.app.find_voltage_corrections_with_restraints"
        ) as mock_find_voltage_corrections_with_restraints,
    ):
        mock_find_voltage_corrections_with_restraints.side_effect = (
            find_voltage_corrections_with_restraints
        )
        voltages = calculate_optimal_voltages(data_dict, (-1000, 1000), 10, 0)
        voltages = np.round(voltages, 2)
        # assert correct voltages calculated
        mock_find_voltage_corrections_with_restraints.assert_called()
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
