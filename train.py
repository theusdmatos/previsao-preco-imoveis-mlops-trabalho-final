import numpy as np
import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "house_price_prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

def load_processed_data():
    print("Carregando dados processados...")
    
    X_train = np.load('processed_data/X_train_processed.npy')
    X_test = np.load('processed_data/X_test_processed.npy')
    y_train = np.load('processed_data/y_train.npy')
    y_test = np.load('processed_data/y_test.npy')
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    print("Avaliando modelo...")
    y_pred = model.predict(X_test)
    y_pred_exp = np.exp(y_pred)
    y_test_exp = np.exp(y_test)
    rmse_log = math.sqrt(mean_squared_error(y_test, y_pred))
    mae_log = mean_absolute_error(y_test, y_pred)
    r2_log = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    r2 = r2_score(y_test_exp, y_pred_exp)
    mape = np.mean(np.abs((y_test_exp - y_pred_exp) / y_test_exp)) * 100
    
    return {
        'rmse_log': rmse_log,
        'mae_log': mae_log,
        'r2_log': r2_log,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test, params=None):
    print(f"Treinando modelo: {model_name}")
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
       
        if params:
            mlflow.log_params(params)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"{metric_name}: {metric_value:.4f}")
        
        mlflow.sklearn.log_model(model, model_name)
        model_uri = f"runs:/{run.info.run_id}/{model_name}"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        print(f"Modelo registrado com sucesso: {registered_model.name} versão {registered_model.version}")
        return run.info.run_id, metrics

def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    print(f"Dados carregados - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {}
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(n_estimators=100, random_state=42),
            "params": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "random_state": 42
            }
        },
        "GradientBoostingRegressor": {
            "model": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            }
        }
    }
    
    results = {}
    for model_name, model_info in models.items():
        print(f"\n{'-'*50}")
        print(f"Treinando modelo: {model_name}")
        
        run_id, metrics = train_and_log_model(
            model_name=model_name,
            model=model_info["model"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=model_info["params"]
        )
        
        results[model_name] = {
            "run_id": run_id,
            "metrics": metrics
        }
    
    # Encontrar o melhor modelo
    best_model = None
    best_score = 0
    for model_name, result in results.items():
        r2 = result["metrics"]["r2"]
        if r2 > best_score:
            best_score = r2
            best_model = model_name
    
    print(f"\n{'-'*50}")
    print(f"Melhor modelo: {best_model} com R² de {best_score:.4f}")
    print(f"Experimento registrado no MLflow: {EXPERIMENT_NAME}")
    print(f"URI de tracking do MLflow: {MLFLOW_TRACKING_URI}")

if __name__ == "__main__":
    main() 