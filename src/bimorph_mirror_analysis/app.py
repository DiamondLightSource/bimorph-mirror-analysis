import base64
import io
from typing import Any, TypedDict

import dash_ag_grid as dag  # type: ignore
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback, dcc, html  # type: ignore

from bimorph_mirror_analysis.maths import (
    find_voltage_corrections_with_restraints,
)
from bimorph_mirror_analysis.read_file import read_bluesky_plan_output

external_stylesheets = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
]
app = Dash(__name__, external_stylesheets=external_stylesheets)  # type:ignore


app.layout = html.Div(
    children=[
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
        html.Div(
            id="data-viewer",
            style={"display": "none"},
            children=[
                html.Div(
                    id="ag-grid-container",
                    children=[
                        html.Div(
                            id="ag-grid-buttons",
                            children=[
                                dcc.RadioItems(
                                    id="ag-grid-selector",
                                    options=["Raw Data", "Pivoted Data"],
                                    value="Raw Data",
                                    inline=True,
                                    style={
                                        "margin": "auto",
                                        "margin-top": "20px",
                                        "display": "block",
                                        "text-align": "center",
                                    },
                                ),
                                html.Button(
                                    "Download pivoted data",
                                    id="download-pivoted-data",
                                    n_clicks=0,
                                    style={"display": "block", "margin": "auto"},
                                ),
                            ],
                        ),
                        dag.AgGrid(
                            id="ag-grid",
                            defaultColDef={
                                "sortable": True,
                                "filter": True,
                                "resizable": True,
                            },
                            columnSize="sizeToFit",
                            style={
                                "height": "400px",
                                "width": "80%",
                                "margin": "auto",
                                "margin-top": "5px",
                            },
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="input-variables",
            children=[
                html.Div(
                    [
                        html.Label(
                            "minimum voltage allowed",
                            style={
                                "text-align": "center",
                                "color": "darkgrey",
                                "font-weight": "bold",
                                "font-size": "18px",
                            },
                        ),
                        dcc.Input(
                            id="minimum_voltage_allowed-input",
                            value=-1000,
                            style={
                                "margin-left": "10px",
                                "margin-right": "10px",
                                "display": "block",
                            },
                            size="30",
                        ),
                    ],
                    style={
                        "align-items": "center",
                        "align-content": "center",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "maximum voltage allowed",
                            style={
                                "text-align": "center",
                                "color": "darkgrey",
                                "font-weight": "bold",
                                "font-size": "18px",
                            },
                        ),
                        dcc.Input(
                            id="maximum_voltage_allowed-input",
                            value=1000,
                            style={
                                "margin-left": "10px",
                                "margin-right": "10px",
                                "display": "block",
                            },
                            size="30",
                        ),
                    ],
                    style={
                        "align-items": "center",
                        "align-content": "center",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "maximum adjacent voltage difference",
                            style={
                                "text-align": "center",
                                "color": "darkgrey",
                                "font-weight": "bold",
                                "font-size": "18px",
                            },
                        ),
                        dcc.Input(
                            id="maximum_adjacent_voltage_difference-input",
                            value=500,
                            style={
                                "margin-left": "10px",
                                "margin-right": "10px",
                                "display": "block",
                            },
                            size="30",
                        ),
                    ],
                    style={
                        "align-items": "center",
                        "align-content": "center",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "baseline voltage scan index",
                            style={
                                "text-align": "center",
                                "color": "darkgrey",
                                "font-weight": "bold",
                                "font-size": "18px",
                            },
                        ),
                        dcc.Input(
                            id="baseline_voltage_scan_index-input",
                            value=0,
                            style={
                                "margin-left": "10px",
                                "margin-right": "10px",
                                "display": "block",
                            },
                            size="30",
                        ),
                    ],
                    style={
                        "align-items": "center",
                        "align-content": "center",
                    },
                ),
            ],
            style={
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
                "margin-top": "50px",
            },
        ),
        html.Button(
            "Calculate Voltages",
            id="calculate-button",
            n_clicks=0,
            style={
                "display": "block",
                "margin": "auto",
                "margin-top": "50px",
                "font-size": "20px",
                "background-color": "#84de81",
            },
        ),
        html.Div(
            id="optimal-voltages-section",
            children=[
                html.Label("The optimal voltages are:"),
                html.Span(id="optimal-voltages", children=""),
            ],
            style={
                "text-align": "center",
                "font-size": "20px",
                "margin-top": "20px",
                "display": "block",
                "justify-content": "center",
            },
        ),
        html.Div(
            id="save-file-section",
            children=[
                html.Button(
                    "Download Optimal Voltages",
                    id="save-optimal-voltages-button",
                    n_clicks=0,
                    style={
                        "textAlign": "center",
                        "margin": "10px",
                        "margin-left": "auto",
                        "margin-right": "auto",
                        "font-size": "20px",
                        "background-color": "lightgrey",
                        "display": "inherit",
                        "justify-content": "center",
                        "margin-top": "40px",
                    },
                ),
                dcc.Download(id="download-optimal-voltages"),
            ],
            style={"display": "none"},
        ),
        dcc.Store(id="loaded-data"),
    ],
    style={},
)


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
    Output("data-viewer", "style"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def read_file(
    contents: str, filename: str
) -> tuple[DataDict, str, dict[str, str], dict[str, str]]:
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

        return (output_dict, filename, {"display": "block"}, {"display": "block"})

    except Exception as e:
        print(e)
        return (
            DataDict(
                raw_data_dict={},
                pivoted_data_dict={},
                initial_voltages=np.array([]),
                increment=0,
                filename="",
            ),
            "There was an error processing this file: " + filename,
            {"display": "block"},
            {"display": "none"},
        )


@callback(
    Output("ag-grid", "columnDefs"),
    Output("ag-grid", "rowData"),
    Output("ag-grid", "csvExportParams"),
    Output("download-pivoted-data", "style"),
    State("loaded-data", "data"),
    Input("ag-grid-selector", "value"),
    Input("data-viewer", "style"),
    prevent_initial_call=True,
)
def change_table(
    data_dict: DataDict, value: str, style: dict[str, str]
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
def export_data_as_csv(n_clicks: int) -> bool:
    if n_clicks:
        return True
    return False


def calculate_optimal_voltages(
    data_dict: DataDict,
    min_v: int,
    max_v: int,
    max_diff: int,
    baseline_voltage_scan_idx: int,
) -> np.typing.NDArray[np.float64]:
    pivoted = pd.DataFrame(data_dict["pivoted_data_dict"])
    initial_voltages = data_dict["initial_voltages"]
    increment = data_dict["increment"]

    # numpy array of pencil beam scans
    data: np.typing.NDArray[np.float64] = pivoted[pivoted.columns[1:]].to_numpy()  # type: ignore

    voltage_adjustments = find_voltage_corrections_with_restraints(
        data,
        increment,
        (min_v, max_v),
        max_diff,
        baseline_voltage_scan=baseline_voltage_scan_idx,
    )  # type: ignore

    optimal_voltages = initial_voltages + voltage_adjustments
    return optimal_voltages  # type: ignore


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
        int(min_v),
        int(max_v),
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


if __name__ == "__main__":
    run_server()
