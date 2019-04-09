import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import sys

from TinyErtModel import TinyErtModel
from TinyMockErtModel import TinyMockErtModel
from TinyDashPlot import TinyDashPlot
from TinyDashHistogram import TinyDashHistogram
from TinyDashPlotErrorBar import TinyDashPlotErrorBar
from TinyDashFanChart import TinyDashFanChart

import matplotlib.cm as cm

from webviz_components import Page, Layout, Map
from webviz_plotly.graph_objs import FanChart


external_stylesheets = [
    'https://sibwebvizcdn.blob.core.windows.net/fonts/index.css',
    'https://sibwebvizcdn.blob.core.windows.net/theme/equinor.theme.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

ERT_CONFIG_FILE = '/data/workspace/ert/test-data/local/example_case/example.ert'
# ERT_CONFIG_FILE = sys.argv[1]
tem = TinyErtModel(ERT_CONFIG_FILE)
# tem = TinyMockErtModel(['default0', 'default1', 'default2'], 20)
sel_cases = tem.get_cases()[0]
all_cases = tem.get_cases()
print(all_cases)


COLOR_CM = list(cm.get_cmap('Set1').colors)
COLORS_CB = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
case_color = {case : COLORS_CB[i] for i, case in enumerate(all_cases)}


container= {
    'display':'grid',
    'grid-template-columns':'repeat(3, 1fr)',
    'grid-template-rows':'repeat(3, minmax(100px, auto))',
    'grid-padding': '1em',
}



def get_graph_div(name='func-plot'):
    return html.Div([
                dcc.Graph(id=name,
                          style={'height':'70vh', 'width':'60vw'}
                          )])

def get_table_div(name='table-func-plot', key_name='SUMMARY KEY', keys=[]):
    return html.Div([
                html.H3(
                    children='Select keys for plot:',
                ),
                dt.DataTable(
                    id=name,
                    columns=[{'name':key_name, 'id':'id_key'}],
                    data=[{"id_key": i} for i in keys],
                    style_cell={'textAlign': 'left'},
                    style_table={
                        'maxHeight': '500',
                        'maxWidth': '200',
                        'overflowY': 'scroll'
                    })])

def get_checklist_div(name='case-list', keys=[]):
    return html.Div([
                html.Label('Select case:'),
                dcc.Checklist(
                    id=name,
                    values=[0],
                    options=[{
                        'label': '{}'.format(j),
                        'value': i
                    } for i, j in enumerate(keys)]
                )])

def get_graph_type_div(name='graph_type', keys=[]):
    return html.Div([
                html.Label('Select Graph Type'),
                dcc.RadioItems(
                    options=[{
                        'label': '{}'.format(id),
                        'value': id
                    } for id in keys],
                    value=keys[0],
                    id=name
                )])

def get_textarea_div(name='text-plot', rows=30):
    return html.Div([
                dcc.Textarea(
                    id=name,
                    placeholder='',
                    rows=rows,
                    readOnly=True,
                    style={'maxHeight': '600', 'height': 600}
                )])

def serve_layout():
    return Layout(children = [
        Page(
            id='ertviz',
            title='Tiny ERT Visualization',
            children=[
                html.Div(id='output-1'),
                html.H1(
                    children='Tiny ERT Visualization',
                    style={
                        'textAlign': 'center',
                    }
                ),
                html.H3(
                    children='Ert config file: {}'.format(ERT_CONFIG_FILE),
                ),
                get_checklist_div(name='cases', keys=tem.get_cases()),
                get_graph_type_div(name='graph_type', keys = ['Function plot', 'FanChart', 'Stat. function plot']),
                html.Div([
                    get_table_div(name='table-func-plot', key_name='SUMMARY KEY', keys=tem.get_summary_keys()),
                    get_graph_div(name='func-plot'),
                    get_textarea_div(name='func-plot-area')
                ], style=container),
                html.Div([
                    get_table_div(name='table-hist-plot', key_name='GEN KW KEY', keys=tem.get_gen_kw_keys()),
                    get_graph_div(name='hist-plot'),
                    get_textarea_div(name='hist-plot-area')
                ], style=container)
            ]
        )])

app.layout = serve_layout()


@app.callback(
    Output('func-plot-area', 'value'),
    [Input('cases', 'values')])
def update_cases(values):
    global sel_cases
    if values is not None:
        sel_cases = []
        for i in values:
            if all_cases[i] is not None:
                sel_cases.append(all_cases[i])
    return sel_cases

@app.callback(
    Output('func-plot', 'figure'),
    [Input('table-func-plot', 'active_cell')],
    state = [State('graph_type', 'value')])
def update_func_plot(active_cell, graph_type_value):
    global tem
    case_data = {}
    title = 'Not key selected'
    if active_cell is not None:
        key = tem.get_summary_keys()[active_cell[0]]
        case_data = {case: tem.get_summary_data(case, key)
                     for case in sel_cases}
        title = 'Function plot for {} in cases {}'.format(key, ', '.join(sel_cases))

    if graph_type_value == 'Function plot':
        return TinyDashPlot().get_figure(case_data, title, case_color)
    elif graph_type_value == 'FanChart':
        return FanChart(data=TinyDashFanChart().get_figure(case_data, title, case_color), colors=COLOR_CM)
    return TinyDashPlotErrorBar().get_figure(case_data, title, case_color)

@app.callback(
    Output('hist-plot', 'figure'),
    [Input('table-hist-plot', 'active_cell')])
def update_func_plot(active_cell):
    global tem
    case_data = {}
    title = 'Not key selected'
    if active_cell is not None:
        key = tem.get_gen_kw_keys()[active_cell[0]]
        case_data = {case: tem.get_gen_kw_data(case, key)
                     for case in sel_cases}
        title = 'Histogram for {} in cases {}'.format(key, ', '.join(sel_cases))
    return TinyDashHistogram().get_figure(case_data, title, case_color)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)