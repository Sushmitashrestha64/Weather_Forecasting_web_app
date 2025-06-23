from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
import os
import joblib
import pytz
import pandas as pd
from datetime import datetime, timedelta
from .ML_model import predict_future
from .api_client import get_current_weather
from .model_training import train_and_save_models

MODELS_DIR = os.path.join(settings.BASE_DIR, 'Forecasting', 'models')

def index_view(request):
    return render(request, 'index.html')

def weather_view(request):
    if request.method != 'POST':
        return redirect('index_view')

    city = request.POST.get('city', '').strip()
    if not city:
        messages.error(request, "Error: City name cannot be empty.")
        return redirect('index_view')

    success, result = train_and_save_models(city)
    if not success:
        messages.error(request, f"Model training failed: {result}")
        return redirect('index_view')

    try:
        rain_model = joblib.load(os.path.join(MODELS_DIR, 'rain_model.joblib'))
        temp_model = joblib.load(os.path.join(MODELS_DIR, 'temp_model.joblib'))
        humidity_model = joblib.load(os.path.join(MODELS_DIR, 'humidity_model.joblib'))
        features_loaded = joblib.load(os.path.join(MODELS_DIR, 'features.joblib'))
    except FileNotFoundError:
        messages.error(request, "Error: Model files not found after training.")
        return redirect('index_view')

    if not all([rain_model, features_loaded, temp_model, humidity_model]):
        messages.error(request, "Cannot run view; models are not loaded.")
        return redirect('index_view')

    current_weather = get_current_weather(city)
    if not current_weather:
        messages.error(request, "Error: Could not fetch weather data.")
        return redirect('index_view')
    

    timezone = pytz.timezone('Asia/Kathmandu')
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute = 0, second =0, microsecond = 0)
    future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]
    

    current_data_for_prediction = {
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'Precipitation': current_weather.get('precipitation', 0),
        'WindDir': current_weather['wind_dir'],
        'Pressure': current_weather['pressure'],
        'WindSpeed': current_weather['wind_speed'] * 3.6,  
        'Temp': current_weather['current_temp'],
        'Humidity': current_weather['humidity'],
        'day_of_year'  : now.timetuple().tm_yday,
        'month'        : now.month,
        'day_of_week'  : now.weekday()
    }
    current_df = pd.DataFrame([current_data_for_prediction])[features_loaded]
    current_df = current_df.fillna(0)

    rain_prediction = rain_model.predict(current_df.values)[0]
    temp_prediction = temp_model.predict(current_df.values)[0]
    humidity_prediction = humidity_model.predict(current_df.values)[0]

    future_temp_preds = predict_future(temp_model, current_data_for_prediction, features_loaded)
    future_humidity_preds = predict_future(humidity_model, current_data_for_prediction, features_loaded)

    

    context = {
        'city': current_weather['city'],
        'country': current_weather['country'],
        'date': now.strftime('%Y-%m-%d'),
        'time': now.strftime('%H:%M:%S %Z'),
        'description': current_weather['description'].capitalize(),
        'current_temp': current_weather['current_temp'],
        'feels_like': current_weather['feels_like'],
        'clouds': current_weather['clouds'],
        'humidity': current_weather['humidity'],
        'visibility': current_weather['visibility'],
        'wind_speed': round(current_weather['wind_speed'], 1),
        'wind_dir': current_weather['wind_dir'],
        'pressure': current_weather['pressure'],
        'rain_prediction': 'a chance of rain' if rain_prediction == 1 else 'no chance of rain',
        'temp_prediction': round(temp_prediction, 1),
        'humidity_prediction': round(humidity_prediction, 1),
        'time1': future_times[0],
        'temp1': round(future_temp_preds[0], 1),
        'hum1': round(future_humidity_preds[0], 1),

        'time2': future_times[1],
        'temp2': round(future_temp_preds[1], 1),
        'hum2': round(future_humidity_preds[1], 1),

        'time3': future_times[2],
        'temp3': round(future_temp_preds[2], 1),
        'hum3': round(future_humidity_preds[2], 1),

        'time4': future_times[3],
        'temp4': round(future_temp_preds[3], 1),
        'hum4': round(future_humidity_preds[3], 1),

        'time5': future_times[4],
        'temp5': round(future_temp_preds[4], 1),
        'hum5': round(future_humidity_preds[4], 1),
        }
    return render(request, 'weather.html', context) 