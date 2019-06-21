# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import flask
import networkx as nx
import pandas as pd
import numpy as np
from ast import literal_eval
import optimizer_app

#bos_graph = nx.read_gpickle('bos_graph')
restaurants = pd.read_csv('bos_restaurants',index_col=0, 
    converters = {"categories": literal_eval, "location.display_address":literal_eval})
#grid_points = np.loadtxt('grid_points')
#grid_to_rest = np.loadtxt('gridpts_to_restaurants')

mapbox_access_token = 'pk.eyJ1IjoiY2hldW5nYWhjIiwiYSI6ImNqd3ZscnRyZjAxZjQzeXM1c3hxdml0aDkifQ.cyUjrtUZ01q5isW4UG9-VQ'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([html.Div([
	html.H1('Where2Eat'),
	html.Div(children='''
        Enter addresses for your restaurant search:
    ''', style={'textAlign': 'left',
        }),
	html.Div(dcc.Textarea(placeholder='Enter address 1',
		value='',
		style={'width': '80%'}, id='loc_1')),

	html.Div(dcc.Textarea(placeholder='Enter address 2',
		value='',
		style={'width': '80%'}, id='loc_2')),

	html.Div(dcc.Textarea(placeholder='Enter address 3',
		value='',
		style={'width': '80%'}, id='loc_3')),

	html.Div(dcc.Textarea(placeholder='Enter address 4',
		value='',
		style={'width': '80%'}, id='loc_4')), 

	html.Div('Select price range:'),
	dcc.Checklist(
    options=[
        {'label': '$', 'value': '$'},
        {'label': '$$', 'value': '$$'},
        {'label': '$$$', 'value': '$$$'},
        {'label': '$$$$', 'value': '$$$$'}
    ],
    values=['$','$$','$$$','$$$$'], id='price_range'),

	html.Div('Preferred restaurant types:'),
	dcc.Dropdown(
    options=[
        {'label': 'American (Traditional)', 'value': 'tradamerican'},
        {'label': 'Italian', 'value': 'italian'},
        {'label': 'Amerian (New)', 'value': 'newamerican'},
        {'label': 'Bars', 'value': 'bars'},
        {'label': 'Pizza', 'value': 'pizza'},
        {'label': 'Seafood', 'value': 'seafood'},
        {'label': 'Mexican', 'value': 'mexican'},
        {'label': 'Japanese', 'value': 'japanese'},
        {'label': 'Chinese', 'value': 'chinese'},
        {'label': 'Burgers', 'value': 'burgers'},
        {'label': 'Thai', 'value': 'thai'},
        {'label': 'Mediterranean', 'value': 'mediterranean'},
        {'label': 'Cafes', 'value': 'cafes'},
        {'label': 'Indian', 'value': 'indpak'}],
    multi=True, id='categories'),

	html.Div('Importance of restaurant type:'),
    dcc.Slider(
    min=0,
    max=1,
    step=0.1,
    value=1,
    marks={0.1*i: '{:.1f}'.format(0.1*i) for i in range(10)
    }, id='category_weight'),
    html.Div('',style={'padding': 10}),

    html.Div('Importance of restaurant proximity:'),
    dcc.Slider(
    min=0,
    max=1,
    step=0.1,
    value=1,
    marks={0.1*i: '{:.1f}'.format(0.1*i) for i in range(10)
    }, id='distance_weight'),
    html.Div('',style={'padding': 10}),

    html.Div('Importance of restaurant rating:'),
    dcc.Slider(
    min=0,
    max=1,
    step=0.1,
    value=1,
    marks={0.1*i: '{:.1f}'.format(0.1*i) for i in range(10)
    }, id='rating_weight'),
    html.Div('',style={'padding': 10}),
    html.Button(id='submit', children='Find Restaurants'),
    html.Div('',style={'padding': 30})],
	style={'width': '800px', 'margin-right': 'auto', 'margin-left': 'auto','columnCount': 2}),
	html.Div(id="rated_table"),
	dcc.Graph(id="my-graph",figure = {"layout": go.Layout(title='Restaurant locations', autosize=True, hovermode='closest', showlegend=False, 
            height=550, mapbox={'accesstoken': mapbox_access_token, 'bearing': 0, 'zoom': 15,
            'center': {'lat': 42.3499334, 'lon':  -71.0786254}, "style": 'mapbox://styles/mapbox/light-v9'})})
])


@app.callback(
	[Output(component_id='rated_table', component_property='children'),
	Output(component_id='my-graph', component_property='figure')],
	[Input(component_id='submit', component_property='n_clicks')],
	[State('loc_1', 'value'),
	State('loc_2', 'value'),
	State('loc_3', 'value'),
	State('loc_4', 'value'),
    State('categories', 'value'),
    State('category_weight', 'value'),
    State('distance_weight','value'),
    State('rating_weight', 'value'),
    State('price_range','values')])

def update_table_fig(n, loc1, loc2, loc3, loc4, cat, cat_weight, dist_weight, rate_weight, price_range):

    # addresses = [loc1, loc2, loc3, loc4]
    # weights: categories of restaurant, rating, distance
    # price_range, categories = conditions
    # conditions = [price_range, cat]

    if n is not None:
        addresses = []

        if not loc1=="":
            addresses.append(loc1)
        if not loc2=="":
            addresses.append(loc2)
        if not loc3=="":
            addresses.append(loc3)
        if not loc4=="":
            addresses.append(loc4)

        weights = [cat_weight, rate_weight, dist_weight]
        conditions = [price_range, cat]
        points = [optimizer_app.address_to_coords(address) for address in addresses]
        lats = [point[0] for point in points]
        lons = [point[1] for point in points]

        top_10_info, top_10_display = optimizer_app.get_restaurants(conditions, weights, addresses, restaurants, bos_graph, grid_to_rest, grid_points)

        table = dt.DataTable(
            columns=[{"name": i, "id": i} for i in top_10_display.columns],
            data=top_10_display.to_dict('records')
            )

        trace1 = go.Scattermapbox(lat= top_10_info["coordinates.latitude"], lon=top_10_info["coordinates.longitude"], 
            mode='markers', hoverinfo='text',
            text=top_10_info['name'])

        trace2 = go.Scattermapbox(lat= lats, lon=lons, 
            mode='markers', hoverinfo='text', marker={'symbol':"marker-15",'size': 10},
            text= addresses)

        layout = go.Layout(title='Restaurant locations', autosize=True, hovermode='closest', showlegend=False, 
            height=550, mapbox={'accesstoken': mapbox_access_token, 'bearing': 0, 'zoom': 15,
            "style": 'mapbox://styles/mapbox/light-v9'})

        figure = {"data": [trace1, trace2], "layout": layout}
        return (table, figure)
    else:

        table = dt.DataTable()
        trace = [go.Scattermapbox(lat= restaurants["coordinates.latitude"], lon=restaurants["coordinates.longitude"], 
            mode='markers', hoverinfo='text',
            text=restaurants['name'])]

        layout = go.Layout(title='Restaurant locations', autosize=True, hovermode='closest', showlegend=False, 
            height=550, mapbox={'accesstoken': mapbox_access_token, 'bearing': 0, 'zoom': 15,
            'center': {'lat': 42.3499334, 'lon':  -71.0786254}, "style": 'mapbox://styles/mapbox/light-v9'})

        figure = {"data": trace, "layout": layout}
        return [table, figure]

if __name__ == '__main__':
    app.run_server(debug=True)

