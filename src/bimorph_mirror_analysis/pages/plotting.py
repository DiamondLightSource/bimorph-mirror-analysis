from typing import Any, TypedDict

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

dash.register_page(__name__)  # type: ignore

layout = html.Div(
    [
        dcc.Tabs(
            id="graph-type",
            # primary is the bar above the selected tab
            # background is the main colour of the unselected tabs
            # border is the border of unselected tabs
            colors={"primary": "#000000", "background": "#cad4ff", "border": "#cad4ff"},
            children=[
                dcc.Tab(
                    label="beamline scans", value="beamline-scan", className="plot-tabs"
                ),
                dcc.Tab(
                    label="influcence functions",
                    value="influence-function",
                    className="plot-tabs",
                ),
                dcc.Tab(
                    label="expected correction",
                    value="expected-correction",
                    className="plot-tabs",
                ),
            ],
            value="beamline-scan",
            className="plot-tabs-holder",
        ),
        dcc.Dropdown(
            id="graph-selector",
            style={
                "margin": "auto",
                "margin-top": "20px",
                "display": "block",
                "text-align": "center",
                "width": "50em",
            },
        ),
        html.Div(
            id="plot-input-variables",
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
                            id="plot-minimum_voltage_allowed-input",
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
                            id="plot-maximum_voltage_allowed-input",
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
                            id="plot-maximum_adjacent_voltage_difference-input",
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
                            id="plot-baseline_voltage_scan_index-input",
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
                "display": "none",
                "justify-content": "center",
                "align-items": "center",
            },
        ),
        dcc.Graph(
            id="graph",
            style={"display": "block", "margin": "auto", "height": "70vh"},
        ),
    ],
)


class DataDict(TypedDict):
    raw_data_dict: dict["str", "str"]
    pivoted_data_dict: dict["str", "str"]
    initial_voltages: np.typing.NDArray[np.float64]
    increment: float
    filename: str


@callback(
    Output("graph-selector", "options"),
    Output("graph-selector", "value"),
    Output("graph-selector", "style"),
    Output("plot-input-variables", "style"),
    Input("graph-type", "value"),
    Input("loaded-data", "data"),
)
def update_graph_selector(
    value: str, data: DataDict | None
) -> tuple[list[str], str, dict[str, str], dict[str, str]]:
    if data is None:
        return [], "", {"display": "none"}, {"display": "none"}
    if value == "beamline-scan":
        cols = [col for col in data["pivoted_data_dict"].keys() if "scan" in col]
        return (
            cols,
            cols[0],
            {
                "margin": "auto",
                "margin-top": "20px",
                "display": "block",
                "text-align": "center",
                "width": "50em",
            },
            {"display": "none"},
        )

    elif value == "influence-function":
        num_actuators = len(data["initial_voltages"])
        return (
            [f"actuator {i}" for i in range(num_actuators)],
            "actuator 0",
            {
                "margin": "auto",
                "margin-top": "20px",
                "display": "block",
                "text-align": "center",
                "width": "50em",
            },
            {"display": "none"},
        )

    elif value == "expected-correction":
        return (
            ["expected correction"],
            "expected correction",
            {"display": "none"},
            {
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
                "margin-top": "10px",
            },
        )

    else:
        raise ValueError(f"Invalid value: {value}")


@callback(
    Output("graph", "figure"),
    Input("graph-selector", "value"),
    Input("loaded-data", "data"),
    Input("graph-type", "value"),
)
def graph_data(value: str, data: DataDict, graph_type: str) -> Any:
    if data is None:
        return go.Figure()
    if graph_type == "beamline scans":
        xdata = [
            data["pivoted_data_dict"]["slit_position_x"][str(i)]
            for i in range(len(data["pivoted_data_dict"]["slit_position_x"]))
        ]
        ydata = [
            data["pivoted_data_dict"][value][str(i)]
            for i in range(len(data["pivoted_data_dict"][value]))
        ]
        fig = go.Figure(data=go.Scatter(x=xdata, y=ydata, mode="lines"))
        return fig

    elif graph_type == "influcence functions":
        num = int(value.split(" ")[-1])
        df = pd.DataFrame(data["pivoted_data_dict"])
        xdata = df["slit_position_x"]
        ydata = df[f"pencil_beam_scan_{num + 1}"] - df[f"pencil_beam_scan_{num}"]
        fig = go.Figure(data=go.Scatter(x=xdata, y=ydata, mode="lines"))
        return fig

    elif graph_type == "expected correction":
        df = pd.DataFrame(data["pivoted_data_dict"])
        xdata = df["slit_position_x"]
        initial_y = df["pencil_beam_scan_0"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xdata, y=initial_y, mode="lines"))
        return fig
