import pandas as pd
import numpy as np
import json
import os
import glob
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
import subprocess

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

if not os.path.exists('monitoring_reports'):
    os.makedirs('monitoring_reports')

def get_reference_data():
    print("Carregando dados de referência...")
    X_train = pd.read_csv('processed_data/X_train_original.csv')
    y_train = pd.read_csv('processed_data/y_train.csv')
    
    reference_data = X_train.copy()
    reference_data['price_log'] = y_train.values
    
    reference_data['price'] = np.exp(reference_data['price_log'])
    return reference_data

def get_production_model_details():
    best_model_name = None
    best_model_version = None
    best_r2 = 0
    
    for model in client.search_registered_models():
        model_name = model.name
        
        for model_version in client.search_model_versions(f"name='{model_name}'"):
            # Verificar se é o modelo em produção
            try:
                if hasattr(model_version, 'current_stage') and model_version.current_stage == 'Production':
                    run_id = model_version.run_id
                    run = client.get_run(run_id)
                    metrics = run.data.metrics
                    
                    if "r2" in metrics:
                        r2 = metrics["r2"]
                        if r2 > best_r2:
                            best_r2 = r2
                            best_model_name = model_name
                            best_model_version = model_version.version
            except Exception as e:
                print(f"Erro ao verificar modelo {model_name} versão {model_version.version}: {e}")
    
    if best_model_name is None:
        for model in client.search_registered_models():
            model_name = model.name
            
            for model_version in client.search_model_versions(f"name='{model_name}'"):
                run_id = model_version.run_id
                run = client.get_run(run_id)
                metrics = run.data.metrics
                
                if "r2" in metrics:
                    r2 = metrics["r2"]
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_name = model_name
                        best_model_version = model_version.version
    
    if best_model_name is None:
        print("Nenhum modelo encontrado.")
        return None, None, 0
        
    print(f"Melhor modelo encontrado: {best_model_name} versão {best_model_version} (R²={best_r2:.4f})")
    return best_model_name, best_model_version, best_r2

def collect_inference_data(days=1):
    print(f"Coletando dados de inferência dos últimos {days} dias...")
    start_date = datetime.now() - timedelta(days=days)
    log_files = glob.glob('inference_logs/inference_*.json')
    inference_requests = []
    inference_predictions = []
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
                log_timestamp = datetime.fromisoformat(log_data['timestamp'].replace('Z', '+00:00'))
                
                if log_timestamp >= start_date:
                    inference_requests.append(log_data['request'])
                    inference_predictions.append(log_data['prediction'])
            except Exception as e:
                print(f"Erro ao processar arquivo de log {log_file}: {e}")
    
    print(f"Coletados {len(inference_requests)} registros de inferência.")
    
    if len(inference_requests) == 0:
        return None
    
    current_data = pd.DataFrame(inference_requests)
    current_data['price'] = inference_predictions
    current_data['price_log'] = np.log(current_data['price'])
    return current_data

def analyze_data_drift(reference_data, current_data):
    print("Analisando data drift...")
    numeric_columns = reference_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    exclude_columns = ['price', 'price_log']
    analyze_columns = [col for col in numeric_columns if col not in exclude_columns]
    column_mapping = ColumnMapping()
    column_mapping.target = 'price_log'
    column_mapping.prediction = 'price_log'
    column_mapping.numerical_features = analyze_columns
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    try:
        data_drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        report_path = f"monitoring_reports/data_drift_report_{timestamp}.html"
        data_drift_report.save_html(report_path)
        results = data_drift_report.as_dict()
        dataset_drift = results.get('data_drift', {}).get('data_drift_score', 0.2)
        drift_by_columns = {}
        num_columns_drifted = 0
        
        if 'data_drift' in results and 'data_drift_by_feature' in results['data_drift']:
            for col, data in results['data_drift']['data_drift_by_feature'].items():
                drift_detected = data.get('drift_detected', False)
                drift_by_columns[col] = {'drift_detected': drift_detected}
                if drift_detected:
                    num_columns_drifted += 1
        
        print(f"Dataset drift: {dataset_drift}")
        print(f"Número de colunas com drift: {num_columns_drifted}")
        
        return {
            'report_path': report_path,
            'dataset_drift': dataset_drift,
            'num_columns_drifted': num_columns_drifted,
            'drift_by_columns': drift_by_columns,
            'timestamp': timestamp
        }
    
    except Exception as e:
        print(f"Erro na análise de data drift: {e}")
        import traceback
        traceback.print_exc()
        return None

def retrain_if_needed(drift_results, drift_threshold=0.3, column_threshold=3):
    if drift_results is None:
        print("Sem resultados de drift para análise. Retreinamento não necessário.")
        return False
    
    dataset_drift = drift_results['dataset_drift']
    num_columns_drifted = drift_results['num_columns_drifted']
    needs_retraining = (dataset_drift > drift_threshold) or (num_columns_drifted >= column_threshold)
    
    if needs_retraining:
        print(f"Drift detectado: {dataset_drift:.4f} (threshold: {drift_threshold})")
        print(f"Colunas com drift: {num_columns_drifted} (threshold: {column_threshold})")
        print("Executando script de treinamento...")
        try:
            subprocess.run(["python", "train.py"], check=True)
            print("Treinamento concluído, executando promoção de modelos...")
            subprocess.run(["python", "promote_model.py"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Erro durante o retreinamento: {e}")
            return False
    else:
        print(f"Drift não ultrapassou os limiares de retreinamento (drift: {dataset_drift:.4f}, colunas: {num_columns_drifted}).")
        print("Retreinamento não necessário.")
        return False

def log_monitoring_results(drift_results, retreinament_triggered, model_name, model_version):
    if drift_results is None:
        return
    
    monitoring_log = {
        'timestamp': str(datetime.now()),
        'dataset_drift': drift_results['dataset_drift'],
        'num_columns_drifted': drift_results['num_columns_drifted'],
        'report_path': drift_results['report_path'],
        'retreinament_triggered': retreinament_triggered,
        'current_model': {
            'name': model_name,
            'version': model_version
        }
    }
    
    log_file = f"monitoring_reports/monitoring_log_{drift_results['timestamp']}.json"
    with open(log_file, 'w') as f:
        json.dump(monitoring_log, f, indent=2)
    
    print(f"Resultados do monitoramento salvos em: {log_file}")

def main():
    print(f"Iniciando monitoramento: {datetime.now()}")
    
    # Garantir que os diretórios necessários existam
    if not os.path.exists('inference_logs'):
        os.makedirs('inference_logs')
        
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
        
    # Verificar se os arquivos necessários existem
    if not os.path.exists('processed_data/X_train_original.csv') or not os.path.exists('processed_data/y_train.csv'):
        print("Arquivos de dados de treinamento não encontrados. Execute primeiro o script de preparação de dados.")
        print("Criando dados de exemplo para teste...")
        try:
            raw_data = pd.read_csv('dataset/kc_house_data.csv')
            sample_size = min(1000, len(raw_data))
            raw_data_sample = raw_data.sample(sample_size, random_state=42)
            # Separar features e target
            X = raw_data_sample.drop('price', axis=1)
            y = np.log(raw_data_sample['price'])
            if not os.path.exists('processed_data'):
                os.makedirs('processed_data')
            X.to_csv('processed_data/X_train_original.csv', index=False)
            pd.DataFrame(y, columns=['price_log']).to_csv('processed_data/y_train.csv', index=False)
            print("Dados de exemplo criados com sucesso.")
        except Exception as e:
            print(f"Erro ao criar dados de exemplo: {e}")
            return
    
    try:
        model_info = get_production_model_details()
        if model_info[0] is None:
            print("Nenhum modelo em produção encontrado.")
            print("Monitoramento concluído.")
            return
            
        model_name, model_version, model_r2 = model_info
        print(f"Modelo atual em produção: {model_name} versão {model_version} (R²={model_r2:.4f})")
        reference_data = get_reference_data()
        current_data = collect_inference_data(days=7)  # Ajustar conforme necessário
        
        if current_data is not None and not current_data.empty:
            drift_results = analyze_data_drift(reference_data, current_data)
            
            # Verificar necessidade de retreinamento
            retreinament_triggered = retrain_if_needed(drift_results)
            log_monitoring_results(drift_results, retreinament_triggered, model_name, model_version)
        else:
            print("Dados de inferência insuficientes para análise de drift.")
            print("Criando dados de exemplo para teste...")
            sample_size = min(100, len(reference_data))
            sample_data = reference_data.sample(sample_size).copy()
            for col in sample_data.select_dtypes(include=['number']).columns:
                if col not in ['price', 'price_log']:
                    sample_data[col] = sample_data[col] * np.random.uniform(0.9, 1.1, size=len(sample_data))
            drift_results = analyze_data_drift(reference_data, sample_data)
            
            retreinament_triggered = False
            if drift_results:
                print(f"Demonstração de drift: {drift_results['dataset_drift']:.4f}")
                print(f"Demonstração de colunas com drift: {drift_results['num_columns_drifted']}")
            
            log_monitoring_results(drift_results, retreinament_triggered, model_name, model_version)
    
    except Exception as e:
        print(f"Erro durante o monitoramento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 