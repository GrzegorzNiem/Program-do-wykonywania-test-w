import time
import tracemalloc
import pandas as pd
import logging
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'dane_czyste_ml.csv')


class ModelBenchmarker:
    def __init__(self, data_path=DEFAULT_DATA_PATH):
        self.data_path = data_path
        self.models = {
            "Regresja Liniowa (Skalowana)": make_pipeline(StandardScaler(), LinearRegression()),
            "Drzewo Decyzyjne": DecisionTreeRegressor(max_depth=12, random_state=42),
            "Las Losowy": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
        }

    def load_and_split_data(self):
        if not os.path.exists(self.data_path):
            return None, None, None, None

        df = pd.read_csv(self.data_path)

        feature_cols = ['Powierzchnia', 'Pokoje', 'Pietro', 'Rok_budowy', 'Miejsce_parkingowe', 'Stan']
        X = df[feature_cols]
        y = df['Cena']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def benchmark(self, model, X_train, X_test, y_train, y_test):
        tracemalloc.start()
        start_train_time = time.perf_counter()

        model.fit(X_train, y_train)

        train_time = time.perf_counter() - start_train_time
        _, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        start_inf_time = time.perf_counter()
        y_pred = model.predict(X_test)
        inference_time = time.perf_counter() - start_inf_time

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "MAE [PLN]": round(mae, 2),
            "R^2 Score": round(r2, 4),
            "Czas trenowania [s]": round(train_time, 5),
            "Czas predykcji [s]": round(inference_time, 5),
            "Max RAM [MB]": round(peak_ram / (1024 * 1024), 4)
        }