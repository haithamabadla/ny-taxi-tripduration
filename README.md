# ny-taxi-tripduration

The aim of this project is to predict New York taxi trip duration. Model was trained on 750,000 record, evaluated on 350,000 and tested on 150,000 records using Keras/ Tensorflow

Pass pickup and drop-off latitude and longitude points to https://ny-trip-duration-flaskapi.herokuapp.com/parameters?points=[POINTS] - i.e. https://ny-trip-duration-flaskapi.herokuapp.com/parameters?points=40.719158,-73.981743,40.829182,-73.938828

API will return the following in dictionary format:

1- Date 
2- Time 
3- Descriptive addresses for both From/ To 
4- Weather information - does it snow? does it rain? 
5- Whether the trip in a weekend or a weekday 
6- Whether the trip is during rush hours or not. 
7- Predicted trip duration

