import pandas as pd 
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings
warnings.filterwarnings('ignore') 
import random
random.seed(42)

# запрашиваем путь к файлу для прогноза
csv_file_path_for_prediction = input("Введите путь к данным для прогноза: ")
csv_file_path_for_prediction = csv_file_path_for_prediction.strip('""')

try:
    test_df = pd.read_csv(csv_file_path_for_prediction)
except FileNotFoundError:
    print("Файл не найден")
    exit()

# формируем полную дату из столбцов 'date' и 'time'
test_df['time'] = test_df['time'].astype(str).str.zfill(2)  
test_df['datetime'] = pd.to_datetime(test_df['date'] + ' ' + test_df['time'].astype(str) + ':00:00')
test_df['datetime'] = pd.to_datetime(test_df['datetime'])

# создаем календарные признаки
test_df["Year"] = pd.to_datetime(test_df["datetime"]).dt.year
test_df["Month"] = pd.to_datetime(test_df["datetime"]).dt.month
test_df["Day"] = pd.to_datetime(test_df["datetime"]).dt.day
test_df["Hour"] = pd.to_datetime(test_df["datetime"]).dt.hour 
test_df["Quarter"] = pd.to_datetime(test_df["datetime"]).dt.month.apply(lambda x: (x-1)//3 + 1)
test_df["DayOfWeek"] = pd.to_datetime(test_df["datetime"]).dt.dayofweek
test_df["WeekOfYear"] = pd.to_datetime(test_df["datetime"]).dt.week
test_df["IsWeekend"] = pd.to_datetime(test_df["datetime"]).dt.dayofweek // 5
test_df["DayOfYear"] = pd.to_datetime(test_df["datetime"]).dt.dayofyear
test_df['time'] = test_df['time'].astype(int)

# функция для определения времени года
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"
    
test_df["Season"] = pd.to_datetime(test_df["datetime"]).dt.month.apply(get_season)

# функция для определения части суток
def get_time_of_day(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 23:
        return "Evening"
    else:
        return "Night"
    
test_df["TimeOfDay"] = pd.to_datetime(test_df["datetime"]).dt.hour.apply(get_time_of_day)

# кодируем категориальные признаки
test_df = pd.get_dummies(test_df, columns=["Season", "TimeOfDay"])

# создаем лаги с шагом 7 дней, начиная с 64-го дня
def create_lags(df, column_name, start_day, step, max_lags):
    for lag in range(start_day, start_day + step * max_lags + 1, step):
        df[f'lag_{lag - 1}'] = df[column_name].shift(lag * 24)

    df.dropna(inplace=True)

create_lags(test_df, 'target', start_day=64, step=7, max_lags=9)

# определяем абсолютный путь к файлу main.py
main_py_path = os.path.abspath(__file__)

# определяем путь к файлу модели в том же каталоге, что и main.py
model_file = os.path.join(os.path.dirname(main_py_path), "model.pkl")

try:
    model_rf = joblib.load(model_file)
except FileNotFoundError:
    print("Модель не найдена")
    exit()

# фильтруем данные для прогноза в указанном периоде
test = test_df.query('datetime >= "2023-08-01 00:00:00"')
X_test = test.drop(columns=['target', 'datetime', 'date', 'temp', 'weather_pred', 'weather_fact'])
y_pred = model_rf.predict(X_test)
test['predict'] = y_pred

# результаты
result = test.groupby('date', as_index=False).agg({'predict': 'sum'})
output_filename = 'predict_команда_8.csv'
output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), output_filename)
result.to_csv(output_path, index=False)
print(result)

print(f'Результаты сохранены в корневой директории: {output_path}')

