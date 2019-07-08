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

bos_graph = nx.read_gpickle('transit_graph')
restaurants = pd.read_csv('bos_restaurants_1',index_col=0, 
    converters = {"categories": literal_eval, "location.display_address":literal_eval})

mapbox_access_token = 'pk.eyJ1IjoiY2hldW5nYWhjIiwiYSI6ImNqd3ZscnRyZjAxZjQzeXM1c3hxdml0aDkifQ.cyUjrtUZ01q5isW4UG9-VQ'
#external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/milligram/1.3.0/milligram.css']

#https://cdnjs.cloudflare.com/ajax/libs/milligram/1.3.0/milligram.css

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
        html.Div('', style={'padding':20}),
        html.Div([
	html.H1('Eat Together'),
        html.H4('Helping friends meet in the middle'),
        html.Div('', style={'padding':15}),
	html.Div(children='''
        Enter addresses for your restaurant search:
    ''', style={'textAlign': 'left',
        }),
	html.Div(dcc.Textarea(placeholder='Enter address 1',
		value='Prudential Center, Boston',
		style={'width': '80%','font-size': '14px'}, id='loc_1')),

	html.Div(dcc.Textarea(placeholder='Enter address 2',
		value='',
		style={'width': '80%','font-size': '14px'}, id='loc_2')),

	html.Div(dcc.Textarea(placeholder='Enter address 3',
		value='',
		style={'width': '80%','font-size': '14px'}, id='loc_3')),

	html.Div(dcc.Textarea(placeholder='Enter address 4',
		value='',
		style={'width': '80%','font-size': '14px'}, id='loc_4')), 

        html.Div('', style={'padding':100}),
	html.Div('Select price range:'),
	dcc.Checklist(
    options=[
        {'label': '$', 'value': '$'},
        {'label': '$$', 'value': '$$'},
        {'label': '$$$', 'value': '$$$'},
        {'label': '$$$$', 'value': '$$$$'}
    ], labelStyle={'display':'inline'},
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
    html.Div(dcc.Slider(
    min=0,
    max=1,
    step=0.1,
    value=1,
    marks={0.1*i: '{:.1f}'.format(0.1*i) for i in range(10)
    }, id='category_weight'),style={'width':'90%','margin-left':'auto', 'margin-right':'auto'}),

    html.Div('',style={'padding': 10}),

    html.Div('Importance of restaurant proximity:'),
    html.Div(dcc.Slider(
    min=0,
    max=1,
    step=0.1,
    value=1,
    marks={0.1*i: '{:.1f}'.format(0.1*i) for i in range(10)
    }, id='distance_weight'),style={'width':'90%','margin-left':'auto', 'margin-right':'auto'}),

    html.Div('',style={'padding': 10}),

    html.Div('Importance of restaurant rating:'),
    html.Div(dcc.Slider(
    min=0,
    max=1,
    step=0.1,
    value=1,
    marks={0.1*i: '{:.1f}'.format(0.1*i) for i in range(10)
    }, id='rating_weight'),style={'width':'90%','margin-left':'auto', 'margin-right':'auto'}),

    html.Div('',style={'padding': 10}),
    html.Button(id='submit', children='Find Restaurants'),
    html.Div('',style={'padding': 30})],
	style={'width': '800px', 'margin-right': 'auto', 'margin-left': 'auto','columnCount': 2}),
html.Div(id="rated_table",style={'width':'1000px', 'margin-left':'auto', 'margin-right':'auto'}),

    html.Div(dcc.Graph(id="my-graph",figure = {"layout": go.Layout(title='Restaurant locations', autosize=True, hovermode='closest', showlegend=False, 
                    height=550, mapbox={'accesstoken': mapbox_access_token, 'bearing': 0, 'zoom': 15,
                                    'center': {'lat': 42.3499334, 'lon':  -71.0786254}, "style": 'mapbox://styles/mapbox/light-v9'})}),
                    style={'width':'1200px', 'margin-left':'auto', 'margin-right':'auto'}),
	html.Div([html.H2('About:'),
	html.Div("Eat Together recommends restaurants that optimize for total travel time \
		and equality in travel time from multiple locations. This project was built by Anthea Cheung at Insight Data Science \
		during the Summer 2019 Boston session."),
	html.A("Slides", href='https://docs.google.com/presentation/d/1_KEnNJX6ppCQNbTdn_mF7463BQ7xDDJCTLmSHkp6mVs', target="_blank"),
	html.Div(""),
	html.A("Source Code", href='https://github.com/ahccheung/eat-coast', target="_blank"),
        html.Div('', style={'padding':40})
	],style={'width': '800px', 'margin-right': 'auto', 'margin-left': 'auto'})
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
        points = [optimizer_app.address_to_coords(address)[0] for address in addresses]
        lats = [point[0] for point in points]
        lons = [point[1] for point in points]
        centroid = optimizer_app.get_centroid(points)  

        top_10_info, top_10_display = optimizer_app.get_restaurants(conditions, weights, addresses, restaurants, bos_graph)

        table = dt.DataTable(
            columns=[{"name": i, "id": i} for i in top_10_display.columns],
            data=top_10_display.to_dict('records'),
            style_cell = {'font_size': '13px', 'font_family':'sans-serif', 
            'minWidth': '0px', 'maxWidth':'150px', 
            'whiteSpace':'normal'},
            css=[{'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'}]
            )

        trace1 = go.Scattermapbox(lat= top_10_info["coordinates.latitude"], lon=top_10_info["coordinates.longitude"], 
            mode='markers+text', text = top_10_info['name'], hoverinfo = "text", textposition = 'top center',
            marker={'size': 9}, hovertext=top_10_info['name'] + '\r' + top_10_info['price'], name='Restaurants')

        traces = [go.Scattermapbox(lat= [points[i][0]], lon=[points[i][1]], 
            mode='markers', hoverinfo='text', marker={'size': 9},
            text= addresses[i], name=addresses[i]) for i in range(len(addresses))]

        layout = go.Layout(title='Restaurant locations', autosize=True, hovermode='closest', showlegend=True, 
            height=550, mapbox={'accesstoken': mapbox_access_token, 'bearing': 0, 
            'center':{'lat':centroid[0], 'lon': centroid[1]}, 'zoom': optimizer_app.calc_zoom(points, top_10_info),
            "style": 'mapbox://styles/mapbox/light-v9'})

        figure = {"data": [trace1] + traces, "layout": layout}
        return (table, figure)

    else:

        table = dt.DataTable()
        trace = [go.Scattermapbox(lat= restaurants["coordinates.latitude"], lon=restaurants["coordinates.longitude"], 
            mode='markers', hoverinfo='text',
            text=[restaurants['name'],restaurants['price']])]

        layout = go.Layout(title='Restaurant locations', autosize=True, hovermode='closest', showlegend=False, 
            height=550, mapbox={'accesstoken': mapbox_access_token, 'bearing': 0, 'zoom': 13,
            'center': {'lat': 42.3499334, 'lon':  -71.0786254}, "style": 'mapbox://styles/mapbox/light-v9'})

        figure = {"data": trace, "layout": layout}
        return [table, figure]

if __name__ == '__main__':
    app.run_server(debug=True)

