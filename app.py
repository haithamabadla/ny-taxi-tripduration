# import numpy as np
import pytz
from flask import Flask, jsonify, request
from datetime import datetime as dt

# Read web API, weather API in this case
import requests

# Calculate distance and get the actual address
from geopy.geocoders import Nominatim
from geopy.distance import great_circle

# Import and run models
import joblib as j
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import exp, reshape
# from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


# Load model and scaler
model = load_model('keras_model.h5')
scaler = j.load('standard_scaler_model.sav')


# Get weather details, whether it is raining or snowing
def get_weather():
    api_key = "400d640018f1a837e621e50dcf941329"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    city_name = 'New York City, US'
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)

    main = response.json()['weather'][0]['main'].lower()
    description = response.json()['weather'][0]['description'].lower()
    weather = main + ' ' + description

    rain_value = 0
    snow_value = 0

    if 'rain' in weather:
        rain_value = 1
    if 'snow' in weather:
        snow_value = 1

    return rain_value, snow_value


# Calculate distance between pickup/ dropoff points
def get_distance(pickup_point, dropoff_point):
    # points must be tuples (Latitude, Longitude) for each point
    return round(great_circle(pickup_point, dropoff_point).km, 1)


# Extract whether it is rush hour or not
def rush_hours(x):
    return 1 if (x >= 7 and x <= 10) or (x >= 16 and x <= 19) else 0


# Extract what part of the day
def part_of_the_day(x):
    afternoon = 0
    evening = 0
    midnight = 0
    morning = 0
    night = 0
    noon = 0

    if (x >= 5 and x < 12): morning = 1
    if x == 12: noon = 1
    if (x > 12 and x < 17): afternoon = 1
    if (x >= 17 and x < 20): evening = 1
    if (x >= 20 and x <= 23): night = 1
    if (x >= 0 and x <= 4): midnight = 1

    return afternoon, evening, midnight, morning, night, noon


# Extract datetime related features
def get_datetime_features(date):

    month = date.month
    weekday = date.weekday()
    hour = date.hour
    is_rushhour = rush_hours(hour)
    is_weekend = 1 if weekday in [5,6] else 0 # int(np.where((weekday == 5) | (weekday == 6), 1, 0))

    return date, month, weekday, hour, is_rushhour, is_weekend


# Extract quarter
def get_quarter(month):
    quarter = (month - 1) // 3 + 1

    quarter_1 = 0
    quarter_2 = 0
    quarter_3 = 0
    quarter_4 = 0

    if quarter == 1: quarter_1 = 1
    if quarter == 2: quarter_2 = 1
    if quarter == 3: quarter_3 = 1
    if quarter == 4: quarter_4 = 1

    return quarter_1, quarter_2, quarter_3, quarter_4


# Extract weekday
def get_weekday(weekday):
    weekday_0 = 0
    weekday_1 = 0
    weekday_2 = 0
    weekday_3 = 0
    weekday_4 = 0
    weekday_5 = 0
    weekday_6 = 0

    if weekday == 0: weekday_0 = 1
    if weekday == 1: weekday_1 = 1
    if weekday == 2: weekday_2 = 1
    if weekday == 3: weekday_3 = 1
    if weekday == 4: weekday_4 = 1
    if weekday == 5: weekday_5 = 1
    if weekday == 6: weekday_6 = 1

    return weekday_0, weekday_1, weekday_2, weekday_3, weekday_4, weekday_5, weekday_6


# Get the actual address based on the lat/lon points
def get_location_address(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    geolocator = Nominatim(user_agent="ny-taxi-student-project")

    from_point = geolocator.reverse((pickup_latitude, pickup_longitude), timeout=5)
    to_point = geolocator.reverse((dropoff_latitude, dropoff_longitude), timeout=5)

    try:
        from_point = ','.join(from_point[0].split(',')[0:6])
        to_point = ','.join(to_point[0].split(',')[0:6])
    except:
        from_point = ','.join(from_point[0].split(',')[0:3])
        to_point = ','.join(to_point[0].split(',')[0:3])

    return from_point, to_point


# Get predicted trip duration
def get_trip_duration(features):
    # Transform features
    features_scaled = scaler.transform(features)
    # Predict trip duration - value is logged
    predicted_duration = model.predict(features_scaled)
    # Convert predicted logged value to original value using exponential function
    predicted_duration = int(round(exp(predicted_duration).numpy()[0][0])) # int(np.ceil(np.exp(predicted_duration)))

    return predicted_duration


@app.route('/parameters', methods=['GET'])
def parameters():

    # Pass lat/lon points as list from the URL
    points = request.args.get('points')
    points = points.split(',')
   
    # Extract each point
    pickup_latitude = float(points[0] )
    pickup_longitude = float(points[1])
    dropoff_latitude = float(points[2])
    dropoff_longitude = float(points[3])

    # Define timezone
    ny_timezone = pytz.timezone('America/New_York')
    
    # Retrieve weather feature
    is_rain, is_snow = get_weather()
    
    # Retrieve datetime features
    date, month, weekday, hour, is_rushhour, is_weekend = get_datetime_features(dt.now(tz = ny_timezone))
    quarter_1, quarter_2, quarter_3, quarter_4 = get_quarter(month)
    weekday_0, weekday_1, weekday_2, weekday_3, weekday_4, weekday_5, weekday_6 = get_weekday(weekday)
    afternoon, evening, midnight, morning, night, noon = part_of_the_day(hour)
    
    # Retrieve the actual address
    from_point, to_point = get_location_address(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
    
    pickup_point = (pickup_latitude, pickup_longitude)
    dropoff_point = (dropoff_latitude, dropoff_longitude)
    distance = get_distance(pickup_point, dropoff_point)
    
    # Collect all feature inputs
    features = [distance, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, is_rushhour,
                is_weekend, is_rain, is_snow, afternoon, evening, midnight, morning, night, noon, weekday_0, weekday_1,
                weekday_2, weekday_3, weekday_4, weekday_5, weekday_6, quarter_1, quarter_2, quarter_3, quarter_4]
    
    # Convert feature inputs into array, then reshape it
    #features = np.array(features)
    #features = features.reshape(1, -1)
    
    # Convert features into array using keras reshape function
    features = reshape(features, shape=(1,-1)).numpy()
    
    # Retrieve predicted trip duration
    predicted_duration = get_trip_duration(features)
    
    result = {'from_point': from_point, # Locations
              'to_point': to_point,
              'date': dt.now(tz = ny_timezone).strftime("%d %B, %Y"), # Datetime related
              'hour': dt.now(tz = ny_timezone).time().strftime('%I:%M %p'),
              'is_rushhour': is_rushhour,
              'is_weekend': is_weekend,
              'distance': distance, # Distance
              'is_rain': is_rain, # Weather
              'is_snow': is_snow,
              'predicted_duration': predicted_duration}
    
    return result

if __name__ == '__name__':
    app.run(threaded=True)