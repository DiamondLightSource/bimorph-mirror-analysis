import dash
from dash import html

dash.register_page(__name__)  # type: ignore

layout = html.Div("New Page")
