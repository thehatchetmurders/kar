import pandas as pd
import numpy as np
import mlflow
from mlflow.entities import run
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.tracking import MlflowClient
import datetime
import optuna
import pickle
import mlflow.pyfunc
import matplotlib.pyplot as plt
from catboost import Pool
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException
import re
import os

def evaluate_prediction(y_test: pd.Series, y_pred: pd.Series) -> float:
    y_test = y_test.reset_index(drop=True)
    y_pred = pd.Series(y_pred, name='y_pred').reset_index(drop=True)

    diff = np.abs(y_test - y_pred)
    diff_percentage = diff / y_test

    result_df = pd.concat(
        [y_test.rename('y_test'), y_pred, diff.rename('diff'), diff_percentage.rename('diff_percentage')], axis=1)

    print(result_df.describe())

    percentage_diff = (diff > 0.1 * y_test).mean() * 100
    print(f"The percentage of y_pred values differing by more than 10% from y_test is: {percentage_diff:.2f}%")

    return percentage_diff

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rlmse = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    r2 = r2_score(y_true, y_pred)
    custom_metric = evaluate_prediction(y_true, y_pred)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'RLMSE': rlmse,
        'R2': r2,
        'Custom Metric': custom_metric
    }

dat = pd.read_csv(r'C:\Users\Наргис\AppData\Local\Programs\Python\Python310\Scripts\karaganda.csv', encoding = 'cp1251', delimiter = ';')
pd.set_option('display.max_columns', None)
dat = dat.drop(columns=['Улица','Дата последнего осмотра','Дата вложение файла специалистом банка','Причины перерассмотрения','Стоимость по ценовой зоне за единицу','Стоимость по ценовой зоне','Стоимость по модели крыши','Стоимость по модели крыши за единицу:','Отклонение','Целевое назначение земельного участка'])
dat = dat.dropna(subset=['Сумма сделки'])
dat.dropna(subset=['Село/Перекресток/Улица'])
dat.drop(columns=['Подход оценки','Метод оценки'])
dat['Этаж']=dat['Этаж'].astype(int)
dat['Сумма сделки'] = dat['Сумма сделки'].str.replace(',', '.').astype(float).astype(int)
dat['Общая площадь'] = dat['Общая площадь'].str.replace(',', '.').astype(float).astype(int)

def extract_floor(value):
    match = re.search(r'\d+', str(value))
    if match:
        return int(match.group())
    else:
        return None

dat['Этажность'] = dat['Этажность'].apply(extract_floor)

repl = {
    'Кирпич кирпичные': 'Кирпич',
    'Панель': 'Ж/б панели',
    'Газоблок облицовка кирпичом': 'Газоблок',
    'Газоблок обл. кирпич': 'Газоблок',
    'Газоблок облицовка кирпичом': 'Газоблок',
    'Монолит бетон': 'Монолит',
    'Газоблок обложенные кирпичом': 'Газоблок',
    'Железобетон': 'Ж/б панели',
    'Пескоблок СКЦ': 'Шлакоблок',
    'Пеноблочный монолитный бетон., пеноблочные': 'Пеноблочный',
    'Ж/б блок': 'Ж/б панели',
    'Ж/б панели ж/б панель': 'Ж/б панели',
    'Пенобетон пенобетонные блоки': 'Пеноблочный',
    'Монолит пеноблоки': 'Монолит пеноблок'
}

dat['Материал стен'] = dat['Материал стен'].str.strip().replace(repl)

cat_features = ['Материал стен']

dat['Timestamp'] = dat['Дата сделки'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())

y = dat['Сумма сделки']
X = dat[['Общая площадь', 'Этажность', 'Этаж', 'Год постройки', 'Материал стен', 'Широта', 'Долгота', 'Timestamp']]

test_mask = dat['Timestamp'] >= datetime.datetime.strptime('2023-09-20', '%Y-%m-%d').timestamp()
X_train, y_train = X[~test_mask], y[~test_mask]
X_test, y_test = X[test_mask], y[test_mask]

client = mlflow.tracking.MlflowClient()
mlflow.get_tracking_uri()

def find_project_root(current_dir):
    while not os.path.exists(os.path.join(current_dir, '.project_root')):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise Exception("Project root not found.")
        current_dir = parent_dir
    return current_dir

project_root = find_project_root(os.path.dirname(os.path.abspath(__file__)))
mlruns_path = os.path.join(project_root, "mlruns")

mlflow.set_tracking_uri(f"file://{mlruns_path}")

experiment_name = "karg_house_prediction_" + datetime.datetime.today().strftime('%Y-%m-%d')
mlflow.set_experiment(experiment_name)

model = CatBoostRegressor(iterations=200, depth=7, learning_rate=0.08, loss_function='RMSE', verbose=200)

current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
algo_name = "CatBoost"

with mlflow.start_run(run_name=f"{current_date_time}_{algo_name}") as active_run:
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), early_stopping_rounds=30)

    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)

    mlflow.log_param("features", ', '.join(X.columns))
    mlflow.log_param("algorithm", "CatBoost")

    mlflow.log_param("train_start_date", dat['Timestamp'][~test_mask].min())
    mlflow.log_param("test_start_date", dat['Timestamp'][test_mask].min())

    mlflow.log_param("num_rows_train", len(X_train))
    mlflow.log_param("num_rows_test", len(X_test))
    mlflow.log_param("categorical_features", ', '.join(cat_features))

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    feature_importances = model.get_feature_importance(Pool(X_test, label=y_test, cat_features=cat_features))
    feature_names = X.columns
    for feature, importance in zip(feature_names, feature_importances):
        mlflow.log_metric(f"feature_importance_{feature}", importance)

    signature = infer_signature(X_train, predictions)
    mlflow.catboost.log_model(model, "model", signature=signature)

    run_id = active_run.info.run_id

model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

required_columns = X_train.columns.tolist()
missing_columns = [col for col in required_columns if col not in X_test.columns]
if missing_columns:
    raise ValueError(f"Missing columns in test data: {missing_columns}")

predictions = loaded_model.predict(X_test)
print(predictions)
