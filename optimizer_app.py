import numpy as np
import pandas as pd
import requests
import json
from pandas.io.json import json_normalize
import networkx as nx
import requests
import matplotlib.cm as cm
import matplotlib.colors as colors
from descartes import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon
import pandana as pdna
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.distance import distance

## Cluster restaurants?
## For multiple users: find shortest path for each pair, then take centroid?

def travel_times(locs, gridrest_mat, gridpts):

    closest_gridpt = lambda loc: np.argmin([distance(x, loc).m for x in gridpts])
    inds = [closest_gridpt(loc) for loc in locs]
    rows = np.array([gridrest_mat[ind] for ind in inds])
    total = np.sum(rows,axis=0)
    normalized = 0.5*(1 - (total/ np.max(total)))
    min_dist = np.apply_along_axis(lambda x: min(x), 0, rows)
    max_dist = np.apply_along_axis(lambda x:max(x), 0, rows)
    ratio = 0.5*(np.divide(min_dist, max_dist))

    return normalized + ratio

def get_restaurants(conditions, weights, addresses, restaurants, graph, gridrest_mat, gridpts):
	max_radius = 5000
	sources = [address_to_coords(address) for address in addresses]
	centroid = get_centroid(sources)
	top_ratings = score_restaurants(conditions, weights, sources, max_radius, restaurants, gridrest_mat, gridpts)

	#travel_times = top_10.apply(lambda x: get_travel_time(sources,x, graph), axis = 1 )

	top_10 = restaurants.loc[top_ratings]
	return top_10, format_df(top_10)

def format_df(df):

	formatted = df[['name','price','rating', 'review_count']]
	get_titles = lambda x: [d['title'] for d in x.categories]
	into_string = lambda y: ', '.join(y)

	cats= df.apply(lambda x: into_string(get_titles(x)), axis=1)
	addresses= df.apply(lambda x: into_string(x['location.display_address']), axis=1)

	cats = cats.reset_index(drop=True)
	addresses = addresses.reset_index(drop=True)

	formatted = formatted.reset_index(drop=True)
	formatted = formatted.join([ cats, addresses])
	formatted.columns = ['Name', 'Price', 'Yelp Rating', 'Number of Reviews', 'Categories','Address']

	return formatted

def score_restaurants(conditions, weights, sources, max_radius, restaurants,gridrest_mat, gridpts):

	price_range, categories = conditions
	#centroid = get_centroid(sources)
	#loc_restaurants = filter_by_location(centroid, max_radius, restaurants)
	#print(len(loc_restaurants))

	fit_func = lambda x: restaurant_fit(categories, weights, sources, max_radius, x)
	rating_scores = restaurants.apply(fit_func, axis = 1)
	travel_scores = travel_times(sources, gridrest_mat, gridpts)
	scores = rating_scores + travel_scores
	price_index = filter_by_price(price_range, restaurants).index
	filtered_scores = scores.loc[price_index]
	top_ratings = filtered_scores.sort_values(ascending=False).index[:10]

	#return top_ratings
	return top_ratings
	
def remove_null_prices(restaurants):
	restaurants = restaurants[~restaurants.price.isnull()]
	return restaurants 

def filter_by_location(centroid, max_radius, restaurants):
	bbox = ox.bbox_from_point(centroid, max_radius)
	loc_filter = lambda x: in_bounding_box(bbox, x['coordinates.latitude'],x['coordinates.longitude'])

	return restaurants[restaurants.apply(loc_filter, axis = 1)]

def filter_by_price(price_range, restaurants):
	print(price_range)
	if len(price_range)==4:
		filtered_restaurants = restaurants

	else:
		not_null = lambda x: not x.price==''
		not_null_restaurants = restaurants[restaurants.apply(not_null, axis=1)]
		range_filter = lambda x: x.price in price_range
		filtered_restaurants = restaurants[not_null_restaurants.apply(range_filter, axis = 1)]

	return filtered_restaurants

def in_bounding_box(bbox, latitude, longitude):
	n, s, e, w = bbox
	return (s < latitude ) & (latitude < n) & (w < longitude) & (longitude < e)

def get_centroid(points):

	n = len(points)
	xsum = sum([p[0] for p in points])
	ysum = sum([p[1] for p in points])
	return (xsum/n, ysum/n)

def address_to_coords(address):
	geolocator = Nominatim(user_agent="optimizer_app")
	location = geolocator.geocode(address)

	return (location.latitude, location.longitude)



def restaurant_fit(categories, weights, sources, max_radius, restaurant):

	# weights: categories of restaurant, rating, distance

	#distance_score = get_distance_score(weights[2], sources, max_radius, restaurant)

	category_score = get_category_score(weights[0], categories, restaurant)

	rating_score = get_rating_score(weights[1], restaurant)
	#print("Rating score = %.3f , Price score = %.3f, Distance score = %.3f, Category score = %.3f" %(rating_score, price_score, distance_score, category_score))

	return category_score + rating_score

def get_distance_score(weight, sources, radius, restaurant):

	rest_point = (restaurant['coordinates.latitude'],restaurant['coordinates.longitude'])
	dist = np.mean([distance(source, rest_point).m for source in sources])

	score = (radius - dist)/ radius

	return weight * score

def get_price_score(weight, price_range, restaurant):

	score = restaurant.price in price_range

	return weight * score

def get_rating_score(weight, restaurant):
	restaurant_rating = restaurant.rating
	num_reviews = restaurant.review_count

	min_reviews = 10
	max_reviews = 300

	num_reviews = min(num_reviews, max_reviews)
	if num_reviews < min_reviews:
		num_reviews = 0

	# normalize by 4.5 stars * 300 reviews = 1350	
	score = num_reviews * restaurant_rating / 1350

	return weight * score

def get_category_score(weight, categories, restaurant):
	#restaurant_categories = list of dictionaries with aliases and titles

	restaurant_categories = [d['alias'] for d in restaurant.categories]
	if categories is None:
		score = 1
	else:
		score = np.sum([cat in restaurant_categories for cat in categories ])

	return weight * score

def get_travel_time(sources, target, graph):
	# Returns time in minutes. Estimate walking speed of 12 minutes per kilometer
	target_coords = (target['coordinates.latitude'], target['coordinates.longitude'])
	orig_nodes = [ox.get_nearest_node(graph, source) for source in sources]
	target_node = ox.get_nearest_node(graph, target_coords)
	lengths = [nx.shortest_path_length(graph, orig_node, target_node, weight='length')  for orig_node in orig_nodes]
	# 12 minutes per kilometer = 3/250 min per meter
	times = [3 * length/ 250 for length in lengths]

	return times
