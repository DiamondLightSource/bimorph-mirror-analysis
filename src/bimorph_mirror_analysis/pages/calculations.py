import dash  # type: ignore
import dash_ag_grid as dag  # type: ignore
from dash import Input, Output, State, callback, dcc, html  # type: ignore

dash.register_page(__name__, path="/")  # type: ignore

layout = html.Div(
    [
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
    ]
)
