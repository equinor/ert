import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from datetime import datetime
from functools import partial

import pandas as pd
from ert_shared.storage.storage_api import StorageApi
import webviz_subsurface_components as wsc

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


import urllib.request, json
server_url = "http://127.0.0.1:5000/"

def convertdate(dstring):
    return datetime.strptime(dstring, '%Y-%m-%d %H:%M:%S')

def get_axis(data_url):
    with urllib.request.urlopen(data_url) as url:
        data = url.read().decode()
        indexes = data.split(",")
        if indexes and ":" in indexes[0]:
            return list(map(convertdate, indexes))
        else:
            return list(map(int, indexes))

def get_data(data_url):
    with urllib.request.urlopen(data_url) as url:
        data = url.read().decode()
        return list(map(float, data.split(",")))

def get_ensembles():
    with urllib.request.urlopen(server_url+"ensembles") as url:
        data = json.loads(url.read().decode())
    return data["ensembles"]


def api_request(api_url):
    with urllib.request.urlopen(api_url) as url:
        return json.loads(url.read().decode())

def get_parameter_options(ensembles):
    parameter_options = list()
    used_keys = set()
    for ensemble in ensembles:
        for param in api_request(ensemble["ref_url"])['parameters']:
            if param['name'] not in used_keys:
                parameter_options.append({'label': param['name'], 'value': param['name']})
                used_keys.add(param['name'])
    return parameter_options

def set_grid_layout(columns):
    return {
        "display": "grid",
        "alignContent": "space-around",
        "justifyContent": "space-between",
        "gridTemplateColumns": f"{columns}",
    }

def make_buttons(prev_id, next_id):
    return html.Div(
        style=set_grid_layout("1fr 1fr"),
        children=[
            html.Button(
                id=prev_id,
                style={
                    "fontSize": "2rem",
                    "paddingLeft": "5px",
                    "paddingRight": "5px",
                },
                children="⬅",
            ),
            html.Button(
                id=next_id,
                style={
                    "fontSize": "2rem",
                    "paddingLeft": "5px",
                    "paddingRight": "5px",
                },
                children="➡",
            ),
        ],
    )

def prev_value(current_value, options):
    try:
        index = options.index(current_value)
    except ValueError:
        index = None
    if index > 0:
        return options[index - 1]
    return current_value

def next_value(current_value, options):
    try:
        index = options.index(current_value)
    except ValueError:
        index = None
    if index < len(options) - 1:
        return options[index + 1]
    return current_value

ensembles = get_ensembles()
parameter_options = get_parameter_options(ensembles)
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H5("Ensemble"),
                        dcc.Dropdown(
                            id="ensemble-selector",
                            options=[{"label": ensemble["name"], "value": ensemble["ref_url"]} for ensemble in ensembles],
                            value=ensembles[0]["ref_url"],
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.H5("Response"),
                        dcc.Dropdown(
                            id="response-selector",
                        ),
                    ],
                    style={"width": "48%", "float": "right", "display": "inline-block"},
                ),
            ]
        ),
        dcc.Graph(id="responses-graphic"),
        html.Div(
            children=[
                html.Span("Parameter distribution:", style={"font-weight": "bold"}),
                html.Div(
                    style=set_grid_layout("8fr 1fr 2fr"),
                    children=[
                        dcc.Dropdown(
                            id="parameter-selector",
                            options=parameter_options,
                            value=parameter_options[0]['value'],
                            clearable=False,
                        ),
                        make_buttons("prev-btn", "next-btn"),
                    ],
                ),
                wsc.PriorPosteriorDistribution("parameter-graph", data={"iterations": [[]], "values": [[]], "labels": []}),
            ],
        )
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
            line = dict(color='royalblue'),
        )
        realizations_data.append(realization_data)
    return realizations_data

def get_observation_data(observation,x_axis):
    data = get_data(observation["values"]["data_url"])
    stds = get_data(observation["std"]["data_url"])
    x_axis_indexes = get_axis(observation["data_indexes"]["data_url"])
    x_axis = [x_axis[i] for i in x_axis_indexes]
    observation_data = dict(
        x=x_axis,
        y=data,
        text="Observations",
        name="Observations",
        mode="markers",
        marker=dict(color="red", size=10)
    )
    lower_std_data = dict(
        x=x_axis,
        y=[d-std for d, std in zip(data,stds)],
        text="Observations std lower",
        name="Observations std lower",
        mode="line",
        line=dict(color="red", dash="dash")
    )
    upper_std_data = dict(
        x=x_axis,
        y=[d+std for d, std in zip(data,stds)],
        text="Observations std upper",
        name="Observations std upper",
        mode="line",
        line=dict(color="red", dash="dash")
    )
    return [observation_data, lower_std_data, upper_std_data]

def get_parameter_data(realizations, parameter):
    realization_names = list()
    realization_parameters = list()
    for realization in realizations:
        realization_schema = api_request(realization['ref_url'])
        for param in realization_schema['parameters']:
            if param['name'] == parameter:
                realization_names.append(realization['name'])
                realization_parameters.extend(get_data(param['data_url']))
    return (realization_names, realization_parameters)

@app.callback(
    Output('response-selector', 'options'),
    [Input('ensemble-selector', 'value')])
def set_response_options(selected_ensemble_id):
    ensemble_schema = api_request(selected_ensemble_id)
    return [{'label': response["name"], 'value': response["ref_url"]} for response in ensemble_schema["responses"]]

@app.callback(
    Output('response-selector', 'value'),
    [Input('response-selector', 'options')])
def set_responses_value(available_options):
    return available_options[0]['value']

@app.callback(
    Output('parameter-selector', 'options'),
    [Input('ensemble-selector', 'value')])
def set_parameter_options(selected_ensemble_id):
    ensemble_schema = api_request(selected_ensemble_id)
    return [{'label': parameter["name"], 'value': parameter["name"]} for parameter in ensemble_schema["parameters"]]

@app.callback(
    Output("responses-graphic", "figure"),
    [
        Input("ensemble-selector", "value"),
        Input("response-selector", "value"),
    ],
)
def update_graph(
    xaxis_column_name, yaxis_column_name
):
    response = api_request(yaxis_column_name)
    x_axis = get_axis(response["axis"]["data_url"])
    plot_lines = get_realizations_data(response["realizations"], x_axis)
    if "observation" in response:
        plot_lines += get_observation_data(response["observation"]["data"], x_axis)
    return {
        "data": plot_lines,
        "layout": dict(
            xaxis={
                "title": "Index",
            },
            yaxis={
                "title": "Unit TODO",
            },
            margin={"l": 40, "b": 40, "t": 10, "r": 0},
            hovermode="closest",
        ),
    }
@app.callback(
    Output("parameter-selector", "value"),
    [
        Input("prev-btn", "n_clicks"),
        Input("next-btn", "n_clicks"),
    ],
    [
        State("parameter-selector", "value"),
    ],
)
def _set_parameter_from_btn(_prev_click, _next_click, parameter):
    ctx = dash.callback_context.triggered
    if not ctx:
        raise PreventUpdate

    callback = ctx[0]["prop_id"]
    if callback == f"{'prev-btn'}.n_clicks":
        parameter = prev_value(parameter, [option["value"] for option in parameter_options])
    elif callback == f"{'next-btn'}.n_clicks":
        parameter = next_value(parameter, [option["value"] for option in parameter_options])
    return parameter

@app.callback(
    Output("parameter-graph", "data"), 
    [Input("parameter-selector", "value")]
)
def _set_parameter(parameter):
    iterations = []
    values = []
    labels = []
    
    for ensemble in ensembles:
        ensemble_schema = api_request(ensemble['ref_url'])
        (realizations, params) = get_parameter_data(ensemble_schema['realizations'], parameter)
        if realizations:
            iterations.append(ensemble_schema['name'])
            values.append(params)
            labels.append([f"Realization {real}" for real in realizations])
    return {"iterations": iterations, "values": values, "labels": labels}

if __name__ == "__main__":
    app.run_server(debug=True)
