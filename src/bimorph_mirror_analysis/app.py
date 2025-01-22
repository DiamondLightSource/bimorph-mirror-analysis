import base64
import datetime
import io

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html
from dash.dependencies import Input, Output, State

from bimorph_mirror_analysis.maths import (
    find_voltage_corrections_with_restraints,
)
from bimorph_mirror_analysis.read_file import read_bluesky_plan_output

external_stylesheets = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
]
app = Dash(__name__, external_stylesheets=external_stylesheets)

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
                html.Span(id="optimal-voltages"),
            ],
            style={
                "text-align": "center",
                "font-size": "20px",
                "margin-top": "20px",
                "display": "block",
                "justify-content": "center",
            },
        ),
    ],
    style={},
)

uploaded_data = {}


def parse_input(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        pivoted, initial_voltages, increment = read_bluesky_plan_output(
            io.StringIO(decoded.decode("utf-8"))
        )

        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

        # Store the dataframe in the global variable
        uploaded_data["pivoted"] = pivoted
        uploaded_data["initial_voltages"] = initial_voltages
        uploaded_data["increment"] = increment

        # Create a Plotly table
        table = go.Figure(
            data=[
                go.Table(
                    header={
                        "values": list(df.columns),
                        "fill_color": "paleturquoise",
                        "align": "left",
                    },
                    cells={
                        "values": [df[col] for col in df.columns],
                        "fill_color": "lavender",
                        "align": "left",
                    },
                    # columnwidth="auto",
                )
            ]
        )

        return html.Div(
            [
                html.I(className="fas fa-file-csv"),  # Icon for CSV file
                html.Span(f" {filename}"),
                html.Div(dcc.Graph(figure=table)),
            ]
        )
    except Exception as e:
        print(e)
        return html.Div(
            [
                "There was an error processing this file.",
                html.Span(f" {filename}"),
            ]
        )


@callback(
    Output("data-upload-result", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_file(content, name):
    if content is not None:
        child = [parse_input(content, name)]
        return child


def calculate_optimal_voltages(
    data_dict: dict[
        (str, pd.DataFrame), (str, np.typing.NDArray[np.float64]), (str, float)
    ],
    min_v: float,
    max_v: float,
    max_diff: float,
    baseline_voltage_scan_idx: int,
) -> np.typing.NDArray[np.float64]:
    pivoted = data_dict["pivoted"]
    initial_voltages = data_dict["initial_voltages"]
    increment = data_dict["increment"]
    # numpy array of pencil beam scans
    data = pivoted[pivoted.columns[1:]].to_numpy()  # type: ignore

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
    Output("optimal-voltages", "children"),
    Input("calculate-button", "n_clicks"),
    State("minimum_voltage_allowed-input", "value"),
    State("maximum_voltage_allowed-input", "value"),
    State("maximum_adjacent_voltage_difference-input", "value"),
    State("baseline_voltage_scan_index-input", "value"),
)
def calculate_voltages(n_clicks, min_v, max_v, max_diff, baseline_voltage_scan_idx=0):
    # add human readbale save file here

    optimal_voltages = calculate_optimal_voltages(
        uploaded_data,
        float(min_v),
        float(max_v),
        float(max_diff),
        int(baseline_voltage_scan_idx),
    )
    optimal_voltages = np.round(optimal_voltages, 2)

    # save the file here

    return f"[{', '.join([str(i) for i in optimal_voltages])}]"


if __name__ == "__main__":
    app.run_server(debug=True)
