import os
import joblib
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from .data_preprocessing import outlier_check, preprocess_data, features
from .ML_model import RandomForestClassifier, RandomForestRegressor
from .api_client import get_current_weather

def train_and_save_models(city=None):
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(APP_DIR, 'weather.csv')
    MODELS_DIR = os.path.join(APP_DIR, 'models')

    print(f"Attempting to read data from: {DATA_PATH}")
    print(f"Attempting to save models to: {MODELS_DIR}")

    FEATURES = [
        'MinTemp', 'MaxTemp', 'Precipitation', 'WindDir', 'Pressure',
        'WindSpeed', 'Temp', 'Humidity',
        'day_of_year', 'month', 'day_of_week'
    ]
    TARGET_RAIN = 'RainTomorrow'
    TARGET_TEMP = 'Temp'
    TARGET_HUMIDITY = 'Humidity'

    try:
        df = pd.read_csv(DATA_PATH)
        if df.empty:
            return False, "Dataset is empty. Cannot train models."
    except FileNotFoundError:
        print(f"ERROR: weather.csv not found at {DATA_PATH}")
        return False, "Dataset not found."

    if city:
        current_weather = get_current_weather(city)
        if current_weather:
            live_data = pd.DataFrame([{
                'Date': datetime.now(pytz.timezone('Asia/Kathmandu')).strftime('%Y-%m-%d'),
                'Temp': current_weather['current_temp'],
                'MinTemp': current_weather['temp_min'],
                'MaxTemp': current_weather['temp_max'],
                'Precipitation': current_weather['precipitation'],
                'WindDir': current_weather['wind_dir'],
                'WindSpeed': current_weather['wind_speed'] * 3.6,  # Convert to km/h
                'Pressure': current_weather['pressure'],
                'Humidity': current_weather['humidity'],
                'RainTomorrow': None  # Prediction will be made later
            }])
            df = pd.concat([df, live_data], ignore_index=True)
            print(f"Appended live data for {city} to the dataset.")

    df_no_outliers = outlier_check(df)
    processed_df = preprocess_data(df_no_outliers)
    featured_df = features(processed_df)

    if featured_df.empty:
        return False, "No data available for training after preprocessing."

    X = featured_df[FEATURES].values
    y_rain = featured_df[TARGET_RAIN].values
    y_temp = featured_df[TARGET_TEMP].values
    y_humidity = featured_df[TARGET_HUMIDITY].values

    if city and pd.isna(featured_df[TARGET_RAIN]).any():
        featured_df[TARGET_RAIN] = featured_df[TARGET_RAIN].fillna(0) 

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train_rain, y_test_rain = y_rain[:train_size], y_rain[train_size:]
    y_train_temp, y_test_temp = y_temp[:train_size], y_temp[train_size:]
    y_train_humidity, y_test_humidity = y_humidity[:train_size:], y_humidity[train_size:]

    print("\nTraining Models")
    rain_model = RandomForestClassifier(n_trees=50, max_depth=10)
    rain_model.fit(X_train, y_train_rain)
    print("Rain model trained.")

    temp_model = RandomForestRegressor(n_trees=50, max_depth=10)
    temp_model.fit(X_train, y_train_temp)
    print("Temperature model trained.")

    humidity_model = RandomForestRegressor(n_trees=50, max_depth=10)
    humidity_model.fit(X_train, y_train_humidity)
    print("Humidity model trained.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(rain_model, os.path.join(MODELS_DIR, 'rain_model.joblib'))
    joblib.dump(temp_model, os.path.join(MODELS_DIR, 'temp_model.joblib'))
    joblib.dump(humidity_model, os.path.join(MODELS_DIR, 'humidity_model.joblib'))
    joblib.dump(FEATURES, os.path.join(MODELS_DIR, 'features.joblib'))

    # Evaluate models
    print("\n Model Evaluation on Unseen Test Data ")
    print("\n Rain Model Evaluation ")
    rain_preds = rain_model.predict(X_test)
    accuracy = accuracy_score(y_test_rain, rain_preds)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test_rain, rain_preds, target_names=['No Rain', 'Rain']))

    print("\nTemperature Model Evaluation ")
    temp_preds = temp_model.predict(X_test)
    temp_mae = mean_absolute_error(y_test_temp, temp_preds)
    temp_mse = mean_squared_error(y_test_temp, temp_preds)
    temp_rmse = np.sqrt(temp_mse)
    print(f"Mean Absolute Error (MAE):   {temp_mae:.2f} °C")
    print(f"Mean Squared Error (MSE):    {temp_mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {temp_rmse:.2f}")

    print("\nHumidity Model Evaluation")
    humidity_preds = humidity_model.predict(X_test)
    humidity_mae = mean_absolute_error(y_test_humidity, humidity_preds)
    humidity_mse = mean_squared_error(y_test_humidity, humidity_preds)
    humidity_rmse = np.sqrt(humidity_mse)
    print(f"Mean Absolute Error (MAE):   {humidity_mae:.2f} %")
    print(f"Mean Squared Error (MSE):    {humidity_mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {humidity_rmse:.2f}")

    print("\nAll models saved successfully.")
    return True, {"rain_model": rain_model, "temp_model": temp_model, "humidity_model": humidity_model,
                  "X_test": X_test, "y_test_rain": y_test_rain,
                  "y_test_temp": y_test_temp, "y_test_humidity": y_test_humidity}

