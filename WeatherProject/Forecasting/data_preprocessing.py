import pandas as pd
import numpy as np

def preprocess_data(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df.set_index('Date', inplace=True)
    df['Precipitation'] = df['Precipitation'].fillna(0)
    
    if df['RainTomorrow'].dtype == 'object':
        df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    
    df.dropna(subset=['RainTomorrow'], inplace=True)
    df['RainTomorrow'] = df['RainTomorrow'].astype(int)
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def outlier_check(df):
    df_no_outliers = df.copy()
    outlier_check_cols = ['Temp', 'MinTemp', 'MaxTemp', 'WindSpeed', 'Humidity', 'Pressure']
    
    initial_rows = len(df_no_outliers)
    for col in outlier_check_cols:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]
    
    print(f"Rows removed by outlier check: {initial_rows - len(df_no_outliers)}")
    return df_no_outliers


def features(df):
    featured_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(featured_df.index):
        featured_df.index = pd.to_datetime(featured_df.index)
    if 'RainTomorrow' in featured_df.columns and featured_df['RainTomorrow'].dtype == 'object':
        featured_df['RainTomorrow'] = featured_df['RainTomorrow'].map({'Yes': 1, 'No': 0})

    featured_df['day_of_year'] = featured_df.index.dayofyear
    featured_df['month'] = featured_df.index.month
    featured_df['day_of_week'] = featured_df.index.dayofweek

    featured_df.dropna(inplace=True)
    return featured_df

