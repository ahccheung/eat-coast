import numpy as np
import pandas as pd
import math
import networkx as nx
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.distance import distance

def calc_zoom(points, df):

    lats = list(df['coordinates.latitude'].values) + [point[0] for point in points]
    lons = list(df['coordinates.longitude'].values) + [point[1] for point in points]
    min_lat = min(lats)
    max_lat = max(lats)
    min_lng = min(lons)
    max_lng = max(lons)

    width_y = max_lat - min_lat
    width_x = max_lng - min_lng
    zoom_y = -1.446*math.log(width_y) + 7
    zoom_x = -1.415*math.log(width_x) + 7

    return min(round(zoom_y,2),round(zoom_x,2))

def travel_times(weight, locs, restaurants, graph):

    restaurant_nodes = restaurants['closest_node']
    nodes = [ox.get_nearest_node(graph, loc) for loc in locs]
    path_lengths = [nx.shortest_path_length(graph, target=node, weight='time') for node in nodes]
    frames =  [pd.DataFrame.from_dict(path_length,orient='index')/60 for path_length in path_lengths]
    df = frames[0]
    N = len(frames)

    for i in range(1, N):
        df = df.merge(frames[i], left_index=True, right_index=True)
    df = df.loc[restaurant_nodes]

    total = df.apply(lambda x: sum(x), axis=1)
    df_min = df.apply(lambda x: min(x), axis=1)
    df_max = df.apply(lambda x: max(x), axis=1)
    ratio = weight * 0.4 * (df_min / df_max)
    normalized = weight * 0.6 * (1 - (total/ max(total)))

    travel_scores = pd.Series(ratio + normalized, name='travel')
    scores = restaurants['closest_node'].to_frame().join(travel_scores, on='closest_node')

    score_df = restaurants[['id','closest_node']].join(travel_scores, on='closest_node')
    score_df = score_df.join(df, on='closest_node')
    score_df.drop_duplicates(inplace=True)
    travel = score_df['travel'].sort_index()
    dropped = score_df.drop(['id', 'closest_node', 'travel'],axis=1)

    return travel, dropped

def get_restaurants(conditions, weights, addresses, restaurants, graph):

	geocodes = [address_to_coords(address) for address in addresses]
	sources = [geocode[0] for geocode in geocodes]
	addresses = [geocode[1] for geocode in geocodes]

	top_ratings, travel_df = score_restaurants(conditions, weights, sources, restaurants, graph)
	top_10 = restaurants.loc[top_ratings]
	#format_df(top_10, travel_df)

	return top_10, format_df(top_10, travel_df)

def format_df(df, travel_df):

	formatted = df[['name','price','rating', 'review_count']]
	formatted.columns = ['Name', 'Price', 'Yelp Rating', 'Number of Reviews']
	get_titles = lambda x: [d['title'] for d in x.categories]
	into_string = lambda y: ', '.join(y)

	cats= df.apply(lambda x: into_string(get_titles(x)), axis=1)
	cats = cats.rename('Categories')
	addresses= df.apply(lambda x: into_string(x['location.display_address']), axis=1)
	addresses = addresses.rename('Address')

	num_users = travel_df.shape[1]
	distance_columns = ['Est. travel from location ' + str(i+1) for i in range(num_users)]
	lower = travel_df.apply(lambda x:  x - 0.2*x ).astype('int')
	upper = travel_df.apply(lambda x: x + 0.2*x ).astype('int')
	travel_range = lower.astype('str') + '-' + upper.astype('str') + " mins"
	travel_range.columns = distance_columns

	formatted = formatted.join([cats, addresses, travel_range])

	return formatted

def dist_to_address(df, source):
	#labels = ['< 10', '10-15', '15-20', '20-30', '30-40', '40-50', '50-60', '60-80', '80-100', '100-120', '>120']


	#return time_to_address
	pass


def score_restaurants(conditions, weights, sources,restaurants, graph):

	price_range, categories = conditions

	fit_func = lambda x: restaurant_fit(categories, weights, sources, x)
	rating_scores = restaurants.apply(fit_func, axis = 1)

	travel_scores, travel_df = travel_times(weights[2], sources, restaurants, graph)
	travel_scores = 1.25 * travel_scores
	scores = rating_scores + travel_scores

	price_index = filter_by_price(price_range, restaurants).index
	filtered_scores = scores.loc[price_index]

	N = min(10, len(filtered_scores))
	top_ratings = filtered_scores.sort_values(ascending=False).index[:N]
	travel_df = travel_df.loc[top_ratings]

	return top_ratings, travel_df

def filter_by_price(price_range, restaurants):
	#print(price_range)
	if len(price_range)==4:
		filtered_restaurants = restaurants

	else:
		not_null = lambda x: not x.price==''
		not_null_restaurants = restaurants[restaurants.apply(not_null, axis=1)]
		range_filter = lambda x: x.price in price_range
		filtered_restaurants = restaurants[not_null_restaurants.apply(range_filter, axis = 1)]

	return filtered_restaurants

def address_to_coords(address):
	geolocator = Nominatim(user_agent="optimizer_app")
	location = geolocator.geocode(address)

	return (location.latitude, location.longitude), location.address

def restaurant_fit(categories, weights, sources, restaurant):

	# weights: categories of restaurant, rating, distance

	category_score = get_category_score(weights[0], categories, restaurant)

	rating_score = get_rating_score(weights[1], restaurant)

	return category_score + rating_score

def get_rating_score(weight, restaurant):
	restaurant_rating = restaurant.rating
	num_reviews = restaurant.review_count

	min_reviews = 10
	max_reviews = 150

	num_reviews = min(num_reviews, max_reviews)
	if num_reviews < min_reviews:
		num_reviews = 0

	# normalize by 4.5 stars * 100 reviews = 675
	score = num_reviews * restaurant_rating / 675

	return weight * score

def get_category_score(weight, categories, restaurant):
	#restaurant_categories = list of dictionaries with aliases and titles

	restaurant_categories = [d['alias'] for d in restaurant.categories]
	if categories is None:
		score = 1
	else:
		score = np.sum([cat in restaurant_categories for cat in categories ])

	return weight * score

def get_centroid(points):

	n = len(points)

	if n==0:
		centroid = (42.3499334,-71.0786254)

	else: 	
		xsum = sum([p[0] for p in points])
		ysum = sum([p[1] for p in points])
		centroid = (xsum/n, ysum/n)

	return centroid