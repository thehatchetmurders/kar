import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import os
import sys
import re

dat = pd.read_csv(r'C:\Users\Наргис\AppData\Local\Programs\Python\Python310\Scripts\karaganda.csv', encoding = 'cp1251', delimiter = ';')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    pd.set_option('display.max_columns', None)
    df = df.drop(columns=['Улица','Дата последнего осмотра','Дата вложение файла специалистом банка','Причины перерассмотрения','Стоимость по ценовой зоне за единицу','Стоимость по ценовой зоне','Стоимость по модели крыши','Стоимость по модели крыши за единицу:','Отклонение','Целевое назначение земельного участка'])
    df = df.dropna(subset=['Сумма сделки'])
    df.dropna(subset=['Село/Перекресток/Улица'])
    df.drop(columns=['Подход оценки','Метод оценки'])
    df['Этаж']=df['Этаж'].astype(int)
    df['Сумма сделки'] = df['Сумма сделки'].str.replace(',', '.').astype(float).astype(int)
    df['Общая площадь'] = df['Общая площадь'].str.replace(',', '.').astype(float).astype(int)
    repl = {
    'Кирпич кирпичные': 'Кирпич',
    'Панель': 'Ж/б панели',
    'Газоблок облицовка кирпичом': 'Газоблок',
    'Газоблок обл. кирпич': 'Газоблок',
    'Газоблок облицовка кирпичом': 'Газоблок',
    'Монолит бетон':'Монолит',
    'Газоблок обложенные кирпичом': 'Газоблок',
    'Железобетон':'Ж/б панели',
    'Пескоблок СКЦ':'Шлакоблок',
    'Пеноблочный монолитный бетон., пеноблочные':'Пеноблочный',
    'Ж/б блок':'Ж/б панели',
    'Ж/б панели ж/б панель':'Ж/б панели',
    'Пенобетон пенобетонные блоки':'Пеноблочный',
    'Монолит пеноблоки':'Монолит пеноблок'
    }

    df['Материал стен'] = df['Материал стен'].str.strip().replace(repl)
    return df

dat = preprocess_data(dat)

def calculate_distances(df: pd.DataFrame, significant_points: dict) -> pd.DataFrame:
    # Convert the coordinates to radians
    df['latitude_rad'] = np.radians(df['Широта'])
    df['longitude_rad'] = np.radians(df['Долгота'])

    # Convert significant points to radians
    significant_points_rad = {k: [radians(float(i)) for i in v.split(', ')] for k, v in significant_points.items()}

    for district, coords in significant_points_rad.items():
        # Repeat the district's coordinates for each row in the DataFrame
        repeated_coords = np.repeat(np.array([coords]), len(df), axis=0)

        # Calculate haversine distances
        df[district] = haversine_distances(
            df[['latitude_rad', 'longitude_rad']], repeated_coords
        ).diagonal() * 6371000 / 1000  # Multiply by Earth radius to get kilometers
    df['LoanFromDate'] = pd.to_datetime(df['Дата сделки'])
    return df

yerke_significant_points_almaty = {
        'bokeyhanov_street': "49.833189, 73.182199",
        "kazybek_bee": "49.809328, 73.090728"
    }
dat = calculate_distances(dat, yerke_significant_points_almaty)

dat.to_csv("karag.csv",index=False, encoding = 'cp1251')


