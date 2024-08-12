# import os
# import sys
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import haversine_distances
# from math import radians
# import mlflow.pyfunc
# import pickle
#
#
# def find_project_root(current_dir):
#     while not os.path.exists(os.path.join(current_dir, '.project_root')):
#         parent_dir = os.path.dirname(current_dir)
#         if parent_dir == current_dir:
#             raise Exception("Project root not found.")
#         current_dir = parent_dir
#     return current_dir
#
# project_root = find_project_root(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(project_root, 'data')
#
# print("Project root directory:", project_root)
# print("Data directory:", DATA_DIR)
#
# pipelines_dir = os.path.join(project_root, 'src', 'pipelines')
# if pipelines_dir not in sys.path:
#     sys.path.append(pipelines_dir)
#
# print("Project root directory:", project_root)
# print("Data directory:", DATA_DIR)
#
# from scraping_stat_gov import process
#
# model_name = "Kar_catboost_2024_Feb.cbm"
# model_version = 0  # replace with the correct version number
# loaded_model = mlflow.catboost.load_model(r"C:\Users\Наргис\PycharmProjects\kargo\pipelines")
# loaded_model
# #loaded_model.save_model("Almaty_catboost_2023_Nov_version_4.cbm", format="cbm")
# export_path = r"C:\Users\Наргис\Desktop\models11"
#
#
# # Save the model
# # mlflow.pyfunc.save_model(path=export_path, python_model=loaded_model)
# #
# # feature_names = loaded_model.feature_names_


import pandas as pd
import numpy as np
import mlflow.pyfunc
import datetime

model_uri = r'C:\kar\pipelines\mlruns\478528617975616027\1cd6570f4781493e96b553e487ba0cfa\artifacts\model'
model = mlflow.pyfunc.load_model(model_uri)


def predict(input_data):
    data = pd.DataFrame(input_data)


    data = data.astype({
        'Общая площадь': 'int32',
        'Этажность': 'int32',
        'Этаж': 'int32',
        'Год постройки': 'int32',
        'Широта': 'float32',
        'Долгота': 'float32'
    })


    data['Timestamp'] = data['Дата сделки'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())

    predictions = model.predict(data)
    return predictions


if __name__ == "__main__":
    input_data = {
        'Общая площадь': [50, 75],
        'Этажность': [5, 9],
        'Этаж': [3, 6],
        'Год постройки': [1990, 2000],
        'Материал стен': ['Кирпич', 'Ж/б панели'],
        'Широта': [50.4501, 51.4501],
        'Долгота': [30.5234, 31.5234],
        'Дата сделки': ['2024-07-01 00:00:00', '2024-07-02 00:00:00']
    }

    predictions = predict(input_data)
    print("Predictions:", predictions)
