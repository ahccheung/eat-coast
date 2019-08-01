# eat-coast

Eat Together recommends restaurants that optimize for total travel time and equality in travel time from multiple locations. This project was built by Anthea Cheung at Insight Data Science during the Summer 2019 Boston session. You can view this project at [eat-coast.com](http://eat-coast.com).

Files:
  * transit_graph is a custom built networkx graph encoding public transit travel times in the Boston area
  * bos_restaurants_1 is a json file containing information for >1400 Boston area restaurants compiled from the Yelp API.
  * optimizer_app.py is a python file that performs all the restaurant rankings.
  * app.py is a python file containing code for the Dash app.
  
  
To run locally, download files in the repository and simply run
`python app.py`.
