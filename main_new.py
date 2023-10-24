import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings
import random

def read_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("Файл не найден")
        exit()

def create_date_features(df):
    df['time'] = df['time'].astype(str).str.zfill(2)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str) + ':00:00')
    df['datetime'] = pd.to_datetime(df['datetime'])

    df["Year"] = pd.to_datetime(df["datetime"]).dt.year
    df["Month"] = pd.to_datetime(df["datetime"]).dt.month
    df["Day"] = pd.to_datetime(df["datetime"]).dt.day
    df["Hour"] = pd.to_datetime(df["datetime"]).dt.hour
    df["Quarter"] = pd.to_datetime(df["datetime"]).dt.month.apply(lambda x: (x - 1) // 3 + 1)
    df["DayOfWeek"] = pd.to_datetime(df["datetime"]).dt.dayofweek
    df["WeekOfYear"] = pd.to_datetime(df["datetime"]).dt.week
    df["IsWeekend"] = pd.to_datetime(df["datetime"]).dt.dayofweek // 5
    df["DayOfYear"] = pd.to_datetime(df["datetime"]).dt.dayofyear
    df['time'] = df['time'].astype(int)

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def create_season_feature(df):
    df["Season"] = pd.to_datetime(df["datetime"]).dt.month.apply(get_season)

def get_time_of_day(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 23:
        return "Evening"
    else:
        return "Night"

def create_time_of_day_feature(df):
    df["TimeOfDay"] = pd.to_datetime(df["datetime"]).dt.hour.apply(get_time_of_day)

def encode_categorical_features(df):
    df = pd.get_dummies(df, columns=["Season", "TimeOfDay"])
    return df

def create_lags(df, column_name, start_day, step, max_lags):
    for lag in range(start_day, start_day + step * max_lags + 1, step):
        df[f'lag_{lag - 1}'] = df[column_name].shift(lag * 24)
    df.dropna(inplace=True)

def load_model(model_file):
    try:
        return joblib.load(model_file)
    except FileNotFoundError:
        print("Модель не найдена")
        exit()

def filter_and_predict(test_df, model_rf):
    test = test_df.query('datetime >= "2023-08-01 00:00:00"')
    X_test = test.drop(columns=['target', 'datetime', 'date', 'temp', 'weather_pred', 'weather_fact'])
    y_pred = model_rf.predict(X_test)
    test['predict'] = y_pred
    return test

def save_results(result, output_filename):
    output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_filename)
    result.to_csv(output_path, index=False)
    print(result)
    print(f'Результаты сохранены в корневой директории: {output_path}')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    random.seed(42)

    csv_file_path_for_prediction = input("Введите путь к данным для прогноза: ")
    csv_file_path_for_prediction = csv_file_path_for_prediction.strip('""')

    test_df = read_csv_file(csv_file_path_for_prediction)
    create_date_features(test_df)
    create_season_feature(test_df)
    create_time_of_day_feature(test_df)
    test_df = encode_categorical_features(test_df)

    create_lags(test_df, 'target', start_day=64, step=7, max_lags=9)

    main_py_path = os.path.abspath(__file__)
    model_file = os.path.join(os.path.dirname(main_py_path), "model.pkl")

    model_rf = load_model(model_file)

    result = filter_and_predict(test_df, model_rf)
    output_filename = 'predict_команда_8.csv'
    save_results(result, output_filename)
