import base64
import io
from typing import Any, TypedDict

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback, dcc, html  # type: ignore

from bimorph_mirror_analysis.maths import (
    check_voltages_fit_constraints,
    find_voltage_corrections,
    find_voltage_corrections_with_restraints,
)
from bimorph_mirror_analysis.read_file import read_bluesky_plan_output

app = Dash(__name__, use_pages=True)  # type:ignore


nav = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink(page["name"], active=True, href=page["relative_path"]))
        for page in dash.page_registry.values()  # type: ignore,
    ],
    pills=True,
    # fill=True,
    # className="border my-4",
    vertical=False,
    style={"margin": "auto", "display": "block", "align-items": "center"},
)

nav = dcc.Tabs(
    id="pages",
    children=[
        dcc.Tab(label=page["name"], value=page["relative_path"])
        for page in dash.page_registry.values()
    ],
    value="/",
)

app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh="callback-nav"),
        html.H1("Bimorph Mirror Analysis", style={"textAlign": "center"}),
        dcc.Upload(
            html.Button(
                "Upload File",
                style={"font-size": "20px", "background-color": "lightgrey"},
            ),
            id="upload-data",
            style={
                "textAlign": "center",
                "margin": "10px",
                "margin-left": "auto",
                "margin-right": "auto",
            },
            # Allow multiple files to be uploaded
            multiple=False,
        ),
        html.Div(
            id="data-upload-result",
            style={
                "text-align": "center",
                "font-size": "20px",
                "margin-top": "20px",
            },
            children=html.Div(
                id="data-upload-result-children",
                children=[
                    html.I(className="fas fa-file-csv"),  # Icon for CSV file
                    html.Span(
                        id="data-file-name", children="", style={"margin-left": "5px"}
                    ),
                ],
                style={"display": "none"},
            ),
        ),
        nav,
        dash.page_container,
        dcc.Store(id="loaded-data"),
        dcc.Store(id="data-viewer-style", data={"display": "none"}),
    ],
    style={},
)


@callback(
    Output("url", "href"),
    Input("pages", "value"),
    prevent_initial_call=True,
)
def update_url(value: str) -> str:
    return value


class DataDict(TypedDict):
    raw_data_dict: dict["str", "str"]
    pivoted_data_dict: dict["str", "str"]
    initial_voltages: np.typing.NDArray[np.float64]
    increment: float
    filename: str


@callback(
    Output("loaded-data", "data"),
    Output("data-file-name", "children"),
    Output("data-upload-result-children", "style"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def read_file(contents: str, filename: str) -> tuple[DataDict, str, dict[str, str]]:
    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))  # type: ignore

        pivoted, initial_voltages, increment = read_bluesky_plan_output(
            io.StringIO(decoded.decode("utf-8"))
        )

        output_dict: DataDict = {
            "raw_data_dict": df.to_dict(),  # type: ignore
            "pivoted_data_dict": pivoted.to_dict(),  # type: ignore
            "initial_voltages": initial_voltages,
            "increment": increment,
            "filename": filename,
        }

        return (
            output_dict,
            filename,
            {"display": "block"},
        )

    except Exception as e:
        print(e)
        return (
            DataDict(
                raw_data_dict={},
                pivoted_data_dict={},
                initial_voltages=np.array([0]),
                increment=0,
                filename="",
            ),
            "There was an error processing this file: " + filename,
            {"display": "block"},
        )


@callback(
    Output("data-viewer", "style", allow_duplicate=True),
    Output("data-viewer-style", "data"),
    Input("loaded-data", "data"),
    prevent_initial_call=True,
)
def trigger_on_data_upload(data: str) -> tuple[dict[str, str], dict[str, str]]:
    return (
        {"display": "block"},
        {"display": "block"},
    )


@callback(
    Output("data-viewer", "style", allow_duplicate=True),
    Input("url", "pathname"),
    State("data-viewer-style", "data"),
    prevent_initial_call=True,
)
def trigger_on_page_change(url: str, data_viewer_style: dict[str, str]):
    # returns none by default
    print(f"switched to {url}")
    style = data_viewer_style.get("display", "none")
    return {"display": style}


@callback(
    Output("ag-grid", "columnDefs"),
    Output("ag-grid", "rowData"),
    Output("ag-grid", "csvExportParams"),
    Output("download-pivoted-data", "style"),
    State("loaded-data", "data"),
    Input("ag-grid-selector", "value"),
    Input("data-viewer", "style"),
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def change_table(
    data_dict: DataDict, value: str, style: dict[str, str], url: str
) -> tuple[
    list[dict[str, str]],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    if value == "Raw Data":
        df = pd.DataFrame(data_dict["raw_data_dict"])
        pivot_data_style = {"display": "none"}
    elif value == "Pivoted Data":
        df = pd.DataFrame(data_dict["pivoted_data_dict"])
        pivot_data_style = {"display": "block", "margin": "auto"}

    else:
        print("Error: Invalid value for table selector")
        print(value)
        pivot_data_style = {"display": "block", "margin": "auto"}

        return ([{}], {}, {}, pivot_data_style)

    columns = [{"headerName": col, "field": col} for col in df.columns]
    data: dict[str, str] = df.to_dict("records")  # type: ignore

    return (
        columns,
        data,
        {"fileName": f"{data_dict['filename'].split('.')[0]}_pivoted.csv"},
        pivot_data_style,
    )


@callback(
    Output("ag-grid", "exportDataAsCsv"),
    Input("download-pivoted-data", "n_clicks"),
)
def export_data_as_csv(n_clicks: int | None) -> bool:
    if n_clicks:
        return True
    return False


def calculate_optimal_voltages(
    data_dict: DataDict,
    voltage_range: tuple[int, int],
    max_consecutive_voltage_difference: int,
    baseline_voltage_scan_idx: int,
) -> np.typing.NDArray[np.float64]:
    """Calculate the optimal voltages for the bimorph mirror actuators.

    Args:
        data_dict: The DataDict of uploaded data
        voltage_range: The minimum and maximum values a voltage can take
        max_consecutive_voltage_difference: The maximum voltage difference allowed\
 between two consecutive actuators
        baseline_voltage_scan: The index of the pencil beam scan which had no increment\
 applied

    Returns:
        The optimal voltages for the bimorph mirror actuators.
    """
    pivoted = pd.DataFrame(data_dict["pivoted_data_dict"])
    initial_voltages = data_dict["initial_voltages"]
    increment = data_dict["increment"]

    # numpy array of pencil beam scans
    data: np.typing.NDArray[np.float64] = pivoted[pivoted.columns[1:]].to_numpy()  # type: ignore

    voltage_adjustments = find_voltage_corrections(
        data,  # type: ignore
        increment,
        baseline_voltage_scan=baseline_voltage_scan_idx,
    )
    optimal_voltages = initial_voltages + voltage_adjustments
    if check_voltages_fit_constraints(
        optimal_voltages, voltage_range, max_consecutive_voltage_difference
    ):
        return optimal_voltages

    else:
        voltage_adjustments = find_voltage_corrections_with_restraints(
            data,  # type: ignore
            increment,
            initial_voltages,
            voltage_range,
            max_consecutive_voltage_difference,
            baseline_voltage_scan=baseline_voltage_scan_idx,
        )
        optimal_voltages = initial_voltages + voltage_adjustments

        return optimal_voltages


@callback(
    Output("optimal-voltages", "children", allow_duplicate=True),
    Input("calculate-button", "n_clicks"),
    State("loaded-data", "data"),
    State("minimum_voltage_allowed-input", "value"),
    State("maximum_voltage_allowed-input", "value"),
    State("maximum_adjacent_voltage_difference-input", "value"),
    State("baseline_voltage_scan_index-input", "value"),
    prevent_initial_call=True,
)
def calculate_voltages(
    n_clicks: int,
    uploaded_data: DataDict,
    min_v: int,
    max_v: int,
    max_diff: int,
    baseline_voltage_scan_idx: int = 0,
) -> str:
    # prevent running if data not uploaded
    if not getattr(uploaded_data, "keys", None):
        return ""

    # add human readbale save file here

    optimal_voltages = calculate_optimal_voltages(
        uploaded_data,
        (int(min_v), int(max_v)),
        int(max_diff),
        int(baseline_voltage_scan_idx),
    )
    optimal_voltages = np.round(optimal_voltages, 2)

    return f"[{', '.join([str(i) for i in optimal_voltages])}]"


@callback(
    Output("save-file-section", "style"),
    Input("calculate-button", "n_clicks"),
    State("loaded-data", "data"),
    prevent_initial_call=True,
)
def make_download_button_visible(n_clicks: int, data: DataDict) -> dict[str, str]:
    # the below check will evaluate to true if data is of type DataDict
    # checking if type is dict because DataDict does not exist at runtime
    if type(data) is dict:  # type: ignore
        if "filename" in data.keys():
            if data["filename"] != "":
                return {
                    "display": "block",
                    "justify-content": "center",
                    "margin-top": "40px",
                }
    return {"display": "none"}


@callback(
    Output("download-optimal-voltages", "data"),
    Input("save-optimal-voltages-button", "n_clicks"),
    State("optimal-voltages", "children"),
    State("loaded-data", "data"),
    prevent_initial_call=True,
)
def download_data(n_clicks: int, arr_str: str, data: DataDict) -> dict[str, Any]:
    fname = data["filename"]
    arr = np.array(eval(arr_str))
    return {
        "base64": False,
        "content": ",".join([str(i) for i in arr]),
        "filename": f"{fname.split('.')[0]}_optimal_voltages.csv",
    }


def run_server(host_ip: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    app.run(debug=debug, host=host_ip, port=port)  # type:ignore
