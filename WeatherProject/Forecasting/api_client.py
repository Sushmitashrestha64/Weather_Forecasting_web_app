import requests
import os
import joblib
import pandas as pd
from datetime import datetime
import pytz

API_KEY = '70ff9507484aa74214bb3005bc88a1bd'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200 or ('cod' in data and data['cod'] != 200):
            raise ValueError(f"API error: {data.get('message', 'Unknown error')}")
        
        precipitation = 0
        if 'rain' in data and '1h' in data['rain']:
            precipitation = float(data['rain']['1h'])

        utc_dt = datetime.fromtimestamp(data['dt'], tz=pytz.UTC)
        api_timestamp = utc_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"API Data Timestamp: {api_timestamp}")

        weather_data = {
            'city': data['name'],
            'current_temp': round(float(data['main'].get('temp'))),
            'feels_like': round(float(data['main'].get('feels_like'))),
            'temp_min': round(float(data['main'].get('temp_min'))),
            'temp_max': round(float(data['main'].get('temp_max'))),
            'humidity': round(float(data['main'].get('humidity'))),
            'description': data['weather'][0].get('description', 'unknown'),
            'country': data['sys'].get('country', 'Unknown'),
            'wind_dir': float(data['wind'].get('deg')),
            'wind_speed': float(data['wind'].get('speed')),
            'pressure': float(data['main'].get('pressure')),
            'precipitation': precipitation,
            'clouds': int(data['clouds'].get('all')),
            'visibility': int(data.get('visibility'))
        }
        last_row = pd.DataFrame([{
            'Temp': weather_data['current_temp'],
            'MinTemp': weather_data['temp_min'],
            'MaxTemp': weather_data['temp_max'],
            'Humidity': weather_data['humidity'],
            'Pressure': weather_data['pressure'],
            'WindSpeed': weather_data['wind_speed'] * 3.6,
            'WindDir': weather_data['wind_dir'],
            'Precipitation': weather_data['precipitation'],
            'RainTomorrow': None
        }])
        joblib.dump(last_row, os.path.join(MODELS_DIR, 'models', 'last_day_data.joblib'))

        print(f"Successfully fetched API Data for {city}.")
        return weather_data

    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"API Request Failed for {city}: {e}. Using fallback data.")
        try:
            last_row = joblib.load(os.path.join(MODELS_DIR, 'models', 'last_day_data.joblib')).iloc[0]
            print("Using fallback data from the last day of the dataset.")
            return {
                'city': city,
                'current_temp': round(float(last_row['Temp'])),
                'feels_like': round(float(last_row['Temp'])),
                'temp_min': round(float(last_row['MinTemp'])),
                'temp_max': round(float(last_row['MaxTemp'])),
                'humidity': round(float(last_row['Humidity'])),
                'description': 'Data unavailable (fallback)',
                'country': 'Unknown',
                'wind_dir': float(last_row['WindDir']),
                'wind_speed': float(last_row['WindSpeed']) / 3.6, 
                'pressure': float(last_row['Pressure']),
                'precipitation': float(last_row['Precipitation']),
                'clouds': 50,
                'visibility': 10000
            }
        except Exception as fallback_e:
            print(f"Fallback data error: {fallback_e}")
            return None