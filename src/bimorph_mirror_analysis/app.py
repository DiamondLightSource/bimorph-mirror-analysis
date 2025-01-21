"""import datetime
import io

import dash
import numpy as np
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State

from bimorph_mirror_analysis.maths import find_voltage_corrections
from bimorph_mirror_analysis.read_file import read_bluesky_plan_output

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Bimorph Mirror Analysis"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Div(id="output-data-upload"),
        html.Button("Calculate Voltages", id="calculate-button", n_clicks=0),
        html.Div(id="output-voltage"),
    ]
)


def parse_contents(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            return df
        else:
            return None
    except Exception as e:
        print(e)
        return None


@app.callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_output(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            return html.Div(
                [
                    html.H5(filename),
                    html.H6(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    dcc.Graph(
                        figure={
                            "data": [
                                {
                                    "x": df[df.columns[0]],
                                    "y": df[df.columns[1]],
                                    "type": "line",
                                    "name": filename,
                                },
                            ],
                            "layout": {"title": "Uploaded Data"},
                        }
                    ),
                ]
            )
        else:
            return "There was an error processing the file."


@app.callback(
    Output("output-voltage", "children"),
    Input("calculate-button", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
)
def calculate_voltages(n_clicks, contents, filename):
    if n_clicks > 0 and contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            file_path = filename
            optimal_voltages = calculate_optimal_voltages(file_path)
            optimal_voltages = np.round(optimal_voltages, 2)
            date = datetime.datetime.now().date()
            output_path = f"{file_path.replace('.csv', '')}_optimal_voltages_{date}.csv"
            np.savetxt(output_path, optimal_voltages, fmt="%.2f")
            return f"The optimal voltages have been saved to {output_path}. \
The optimal voltages are: [{', '.join([str(i) for i in optimal_voltages])}]"
        else:
            return "There was an error processing the file."
    return ""


if __name__ == "__main__":
    app.run_server(debug=True)
"""
