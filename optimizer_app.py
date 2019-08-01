import numpy as np
import pandas as pd
import math
import networkx as nx
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.distance import distance

def get_restaurants(conditions, weights, addresses, restaurants, graph):

    """Get top recommended restaurants given user-inputted preferences, a dataframe of restaurants, and a 
    transit travel time network graph.

    Args:
        conditions (list): price range and category preferences
        weights (list of floats): weights for the importance of category, restaurant rating, and distance
        addresses (list): user-inputted addreses
        restaurants (pandas DataFrame): dataframe of restaurants
        graph (networkx graph): transit travel time graph

    Returns:
        top_10 (pandas DataFrame): dataframe of top 10 recommended restaurants
        formatted_top_10 (pandas DataFrame): table of top 10 restaurants to be displayed on web app
    """

    # Getted gps coordinates of inputted addresses
    sources = [address_to_coords(address) for address in addresses]

    # Score restaurants based on preferences
    top_ratings, travel_times = score_restaurants(conditions, weights, sources, restaurants, graph)

    # Get top 10 restaurants as ranked by top_ratings
    top_10 = restaurants.loc[top_ratings]
    top_10_display = format_df(top_10, travel_times)

    return top_10, top_10_display

def address_to_coords(address):
    """Get gps coordinates given an address.

    Args:
        address (str): user-inputted address

    Returns:
        (tuple of floats): latitude and longitude coordinates
    """

    geolocator = Nominatim(user_agent="optimizer_app")
    location = geolocator.geocode(address)

    return (location.latitude, location.longitude)

def score_restaurants(conditions, weights, sources,restaurants, graph):

    """Score and rank restaurants given conditions, weights, and starting locations.

    Args:
        conditions (list): price range and category preferences
        weights (list): weights for importance of categories, ratings, and proximity
        sources (list): list of gps coordinates of starting locations
        restaurant (pandas DataFrame): data frame of restaurants parsed from yelp api.
        graph (networkx graph): graph of public transit times

    Returns:
        top_ratings: series of top rated restaurants
        travel_times: dataframe of travel times from restaurants to each starting location
    """

    price_range, categories = conditions

    # rate restaurants based on category and rating
    fit_func = lambda x: restaurant_fit(categories, weights, x)
    rating_scores = restaurants.apply(fit_func, axis = 1)

    # get travel scores and travel time from starting locations to restaurants
    travel_scores, travel_times = get_travel_scores(weights[2], sources, restaurants, graph)
    scores = rating_scores + travel_scores

    #filter out restaurants by price range
    price_index = filter_by_price(price_range, restaurants).index
    filtered_scores = scores.loc[price_index]

    # get top ten restaurants
    N = min(10, len(filtered_scores))
    top_ratings = filtered_scores.sort_values(ascending=False).index[:N]
    travel_times = travel_times.loc[top_ratings]

    return top_ratings, travel_times

def restaurant_fit(categories, weights, restaurant):

    """Score and rank restaurants given conditions, weights, and starting locations.

    Args:
        categories (list): list of desired categories
        weights (list): weights for importance of categories, ratings, and proximity
        restaurant (pandas DataFrame row): dataframe row of restaurant

    Returns:
        score (float): score based on category and ratings of restaurant 
    """

    # weights: categories of restaurant, rating, distance

    category_score = get_category_score(weights[0], categories, restaurant)

    rating_score = get_rating_score(weights[1], restaurant)

    return category_score + rating_score

def get_category_score(weight, categories, restaurant):

    """Score restaurants given category preferences.
    
    Args:
        weight (float): weight for importance of category
        categories (list): list of desired categories
        restaurant (pandas DataFrame row): dataframe row of a restaurant.

    Returns:
        score (float): category score for restaurant 
    """

    # restaurant_categories = list of dictionaries with aliases and titles
    restaurant_categories = [d['alias'] for d in restaurant.categories]

    if categories is None: # restaurants with no category are ranked 1.
        score = 1
    else:
        # score = 1 if restaurant categories belong in preferences
        score = np.sum([cat in restaurant_categories for cat in categories ])

    return weight * score

def get_rating_score(weight, restaurant):

    """Score restaurants given weight for the importance of ratings.
    
    Args:
        weight (float): weight for importance of ratings
        restaurant (pandas DataFrame row): dataframe row of a restaurant.

    Returns:
        score (float): rating score for restaurant 
    """
    restaurant_rating = restaurant.rating
    num_reviews = restaurant.review_count

    min_reviews = 10
    max_reviews = 150

    # clip number of reviews at 150
    num_reviews = min(num_reviews, max_reviews)
    if num_reviews < min_reviews:
        num_reviews = 0

    # normalize by 4.5 stars * 150 reviews = 675
    score = num_reviews * restaurant_rating / 675

    return weight * score

def get_travel_scores(weight, locs, restaurants, graph):

    """Create and return dataframe with updated travel time scores

    Args: 
        weight (float): user-inputted weight for importance of travel time
        locs (list of tuples): latitude and longitude coordinates of addresses
        restaurants (DataFrame): dataframe of restaurants
        graph (networkx graph): transportation and walking network graph

    Returns:
        travel_scores (pandas Series): series of travel scored (between 0 and 1) for each restaurant
        travel_times (pandas DataFrame): Dataframe of travel time information for each restaurant
    """

    # Get shortest travel time from each address to every restaurant
    restaurant_nodes = restaurants['closest_node']
    nodes = [ox.get_nearest_node(graph, loc) for loc in locs]
    path_lengths = [nx.shortest_path_length(graph, target=node, weight='time') for node in nodes]

    # Create dataframe of travel time to restaurants for each address
    frames =  [pd.DataFrame.from_dict(path_length,orient='index')/60 for path_length in path_lengths]
    travel_times = frames[0]
    N = len(frames)

    # Merge travel time dataframes together
    for i in range(1, N):
        travel_times = travel_times.merge(frames[i], left_index=True, right_index=True)
    travel_times = travel_times.loc[restaurant_nodes]

    # Create columns for total, minimum, and maximum travel time for each restaurant
    total = travel_times.apply(lambda x: sum(x), axis=1)
    min_time = travel_times.apply(lambda x: min(x), axis=1)
    max_time = travel_times.apply(lambda x: max(x), axis=1)

    # Calculate scores:

    # weighted ratio of shortest travel time / longest travel time (measure of disparity in travel time)
    ratio = weight * 0.4 * (min_time / max_time)  

    # normalized weighted measure of total travel time from each location 
    # restaurants that have longer travel times have smaller scores)
    normalized = weight * 0.6 * (1 - (total/ max(total)))
    travel_scores = pd.Series(ratio + normalized, name='travel')

    # create dataframe with restaurant id, closest graph node, travel score, and travel times of each restaurant
    score_df = restaurants[['id','closest_node']].join(travel_scores, on='closest_node')
    score_df = score_df.join(travel_times, on='closest_node')
    score_df.drop_duplicates(inplace=True)

    # get travel scores to each restaurant
    travel_scores = score_df['travel'].sort_index()
    # get travel times to each restaurant
    travel_times = score_df.drop(['id', 'closest_node', 'travel'],axis=1)

    return travel_scores, travel_times

def filter_by_price(price_range, restaurants):
    """Filter restaurants according to price range preferences.
    
    Args:
        price_range (list): list of price range preferences ('$','$$','$$$','$$$$')
        restaurants (pandas DataFrame): dataframe of restaurants from yelp api.
    Returns:
        filtered_restaurants (pandas DataFrame): restaurants filtered out by price range preferences
    """

    if len(price_range)==4:
        filtered_restaurants = restaurants

    else:
        # get rid of restaurants with no indicated price range
        not_null = lambda x: not x.price==''
        not_null_restaurants = restaurants[restaurants.apply(not_null, axis=1)]

        # filter by desired price range
        range_filter = lambda x: x.price in price_range
        filtered_restaurants = restaurants[not_null_restaurants.apply(range_filter, axis = 1)]

    return filtered_restaurants

def format_df(df, travel_times):

    """ Format DataFrame for Dash app
    
    Args:
        df (pandas DataFrame): original restaurant dataframe
        travel_times (pandas DataFrame): dataframe with travel times

    Returns:
        display (pandas DataFrame): dataframe formatted for Dash app
    """    

    display = df[['name','price','rating', 'review_count']]
    display.columns = ['Name', 'Price', 'Yelp Rating', 'Number of Reviews']

    # format the categories column of df
    get_titles = lambda x: [d['title'] for d in x.categories]
    into_string = lambda y: ', '.join(y)
    cats= df.apply(lambda x: into_string(get_titles(x)), axis=1)
    cats = cats.rename('Categories')

    # format address column of df
    addresses= df.apply(lambda x: into_string(x['location.display_address']), axis=1)
    addresses = addresses.rename('Address')

    # Create columns for estimated travel times from each source location
    num_users = travel_times.shape[1]
    distance_columns = ['Est. travel from location ' + str(i+1) for i in range(num_users)]

    # travel time range = prediction plus/minus 20% 
    lower = travel_times.apply(lambda x:  x - 0.2*x ).astype('int')
    upper = travel_times.apply(lambda x: x + 0.2*x ).astype('int')
    travel_range = lower.astype('str') + '-' + upper.astype('str') + " mins"
    travel_range.columns = distance_columns

    # return display dataframe for app
    display = display.join([cats, addresses, travel_range])

    return display

