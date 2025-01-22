import base64
import datetime
import io

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html
from dash.dependencies import Input, Output, State

from bimorph_mirror_analysis.maths import find_voltage_corrections
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
            html.Button("Upload File"),
            id="upload-data",
            style={
                # "width": "60%",
                # "height": "60px",
                # "lineHeight": "60px",
                # "borderWidth": "1px",
                # "borderStyle": "dashed",
                # "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
                "margin-left": "auto",
                "margin-right": "auto",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        html.Button(
            "Calculate Voltages",
            id="calculate-button",
            n_clicks=0,
            style={
                "display": "block",
                "margin": "auto",
            },
        ),
        html.Div(
            id="output-data-upload", style={"margin": "auto", "margin-top": "20px"}
        ),
    ],
    style={},
)

uploaded_data = {}


def parse_input(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    if "csv" in filename:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        # Store the dataframe in the global variable
        uploaded_data[filename] = df
        return html.Div(
            [
                html.I(className="fas fa-file-csv"),  # Icon for CSV file
                html.Span(f" {filename}"),
            ]
        )
    else:
        return html.Div(
            [
                html.I(className="fas fa-file"),  # Generic file icon
                html.Span(f" {filename}"),
            ]
        )


@callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [parse_input(c, n) for c, n in zip(list_of_contents, list_of_names)]
        return children


if __name__ == "__main__":
    app.run_server(debug=True)
