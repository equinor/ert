import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from datetime import datetime
from functools import partial

import pandas as pd
from ert_shared.storage.storage_api import StorageApi

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


import json

server_url = "http://127.0.0.1:5000"

import requests


def convertdate(dstring):
    return datetime.strptime(dstring, "%Y-%m-%d %H:%M:%S")


def get_axis(data_url):
    resp = requests.get(data_url)
    indexes = resp.content.decode(resp.encoding).split(",")
    if indexes and ":" in indexes[0]:
        return list(map(convertdate, indexes))
    else:
        return list(map(int, indexes))


def get_data(data_url):
    resp = requests.get(data_url)
    data = resp.content.decode(resp.encoding)
    return list(map(float, data.split(",")))


def get_ensembles():
    resp = requests.get("{base}/ensembles".format(base=server_url))
    return resp.json()["ensembles"]


def api_request(api_url):
    resp = requests.get(api_url)
    return resp.json()


ensembles = get_ensembles()
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H5("Ensemble"),
                        dcc.Dropdown(
                            id="ensemble-selector",
                            options=[
                                {
                                    "label": ensemble["name"],
                                    "value": ensemble["ref_url"],
                                }
                                for ensemble in ensembles
                            ],
                            value=ensembles[0]["ref_url"],
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [html.H5("Response"), dcc.Dropdown(id="response-selector",),],
                    style={"width": "48%", "float": "right", "display": "inline-block"},
                ),
            ]
        ),
        dcc.Graph(id="responses-graphic"),
    ]
)


def get_realizations_data(realizations, x_axis):
    realizations_data = list()
    for realization in realizations:
        data = get_data(realization["data_url"])
        realization_data = dict(
            x=x_axis,
            y=data,
            text="Realization {}".format(realization["name"]),
            name="Realization {}".format(realization["name"]),
            mode="line",
            line=dict(color="royalblue"),
        )
        realizations_data.append(realization_data)
    return realizations_data


def get_observations_data(observations, x_axis):
    return_plots = []
    for observation in observations:
        observation = observation["data"]
        data = get_data(observation["values"]["data_url"])
        stds = get_data(observation["std"]["data_url"])
        x_axis_indexes = get_axis(observation["data_indexes"]["data_url"])
        x_axis_tmp = [x_axis[i] for i in x_axis_indexes]
        observation_data = dict(
            x=x_axis_tmp,
            y=data,
            text="Observations",
            name="Observations",
            mode="markers",
            marker=dict(color="red", size=10),
        )
        lower_std_data = dict(
            x=x_axis_tmp,
            y=[d - std for d, std in zip(data, stds)],
            text="Observations std lower",
            name="Observations std lower",
            mode="line",
            line=dict(color="red", dash="dash"),
        )
        upper_std_data = dict(
            x=x_axis_tmp,
            y=[d + std for d, std in zip(data, stds)],
            text="Observations std upper",
            name="Observations std upper",
            mode="line",
            line=dict(color="red", dash="dash"),
        )
        return_plots.extend([observation_data, lower_std_data, upper_std_data])
    return return_plots


@app.callback(
    Output("response-selector", "options"), [Input("ensemble-selector", "value")]
)
def set_response_options(selected_ensemble_id):
    ensemble_schema = api_request(selected_ensemble_id)
    return [
        {"label": response["name"], "value": response["ref_url"]}
        for response in ensemble_schema["responses"]
    ]


@app.callback(
    Output("response-selector", "value"), [Input("response-selector", "options")]
)
def set_responses_value(available_options):
    return available_options[0]["value"]


@app.callback(
    Output("responses-graphic", "figure"),
    [Input("ensemble-selector", "value"), Input("response-selector", "value"),],
)
def update_graph(xaxis_column_name, yaxis_column_name):
    response = api_request(yaxis_column_name)
    x_axis = get_axis(response["axis"]["data_url"])
    plot_lines = get_realizations_data(response["realizations"], x_axis)
    if "observations" in response:
        plot_lines += get_observations_data(response["observations"], x_axis)
    return {
        "data": plot_lines,
        "layout": dict(
            xaxis={"title": "Index",},
            yaxis={"title": "Unit TODO",},
            margin={"l": 40, "b": 40, "t": 10, "r": 0},
            hovermode="closest",
        ),
    }


if __name__ == "__main__":
    app.run_server(debug=True)
