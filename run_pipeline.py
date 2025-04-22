import subprocess
import os
import time
import argparse
from datetime import datetime

def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_command(command, description):
    print_section(description)
    print(f"Executando: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Comando executado com sucesso!")
        if result.stdout:
            print("\nSaída do comando:")
            print(result.stdout)
    else:
        print(f"Erro ao executar o comando. Código de saída: {result.returncode}")
        if result.stderr:
            print("\nErro:")
            print(result.stderr)
        exit(1)
    
    return result

def run_mlflow_server():
    print_section("Iniciando Servidor MLflow")
    
    if not os.path.exists('mlruns'):
        os.makedirs('mlruns')
    
    cmd = ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", 
           "--default-artifact-root", "./mlruns", "--host", "0.0.0.0", "--port", "5000"]
    
    print(f"Executando: {' '.join(cmd)}")
    print("MLflow será iniciado em segundo plano. Acesse em http://localhost:5000")
    
    if os.name == 'nt':  # Windows
        process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:  # Linux/Mac
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(5)
    return process

def run_dashboard():
    print_section("Iniciando Dashboard Streamlit")
    cmd = ["streamlit", "run", "dashboard.py", "--server.port", "8501"]
    
    print(f"Executando: {' '.join(cmd)}")
    print("Dashboard será iniciado em segundo plano. Acesse em http://localhost:8501")
    
    if os.name == 'nt':  # Windows
        process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:  # Linux/Mac  
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(5)
    return process

def main():
    parser = argparse.ArgumentParser(description="Executa o pipeline MLOps completo ou etapas específicas.")
    parser.add_argument('--prep', action='store_true', help='Executar apenas a preparação de dados')
    parser.add_argument('--train', action='store_true', help='Executar apenas o treinamento de modelos')
    parser.add_argument('--promote', action='store_true', help='Executar apenas a promoção de modelos')
    parser.add_argument('--api', action='store_true', help='Iniciar apenas a API')
    parser.add_argument('--monitor', action='store_true', help='Executar apenas o monitoramento')
    parser.add_argument('--mlflow', action='store_true', help='Iniciar apenas o servidor MLflow')
    parser.add_argument('--dashboard', action='store_true', help='Iniciar apenas o dashboard Streamlit')
    args = parser.parse_args()
 
    run_all = not (args.prep or args.train or args.promote or args.api or args.monitor or args.mlflow or args.dashboard)
    
    print_section("Pipeline MLOps para Previsão de Preços de Imóveis")
    print(f"Data e Hora de Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    mlflow_process = None
    dashboard_process = None
    
    if args.mlflow or run_all:
        mlflow_process = run_mlflow_server()
    
    try:
        if args.prep or run_all:
            run_command(["python", "data_prep.py"], "Preparação de Dados")
        if args.train or run_all:
            run_command(["python", "train.py"], "Treinamento de Modelos")
        if args.promote or run_all:
            run_command(["python", "promote_model.py"], "Promoção de Modelos")
        if args.monitor or run_all:
            run_command(["python", "monitor.py"], "Monitoramento de Drift")
        
        if args.dashboard or run_all:
            dashboard_process = run_dashboard()
            
            if args.dashboard and not (args.api or run_all):
                print("\nDashboard Streamlit iniciado. Pressione Ctrl+C para encerrar.")
                while True:
                    time.sleep(1)
        
        if args.api or run_all:
            print_section("Iniciando API FastAPI")
            print("A API será iniciada em: http://localhost:8000")
            print("Para acessar a documentação: http://localhost:8000/docs")
            print("Pressione Ctrl+C para encerrar...")
            
            subprocess.run(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
    
    except KeyboardInterrupt:
        print("\nOperação interrompida pelo usuário.")
    
    finally:
        if mlflow_process:
            print_section("Encerrando Servidor MLflow")
            mlflow_process.terminate()
            mlflow_process.wait()
            print("Servidor MLflow encerrado.")
        
        if dashboard_process:
            print_section("Encerrando Dashboard Streamlit")
            dashboard_process.terminate()
            dashboard_process.wait()
            print("Dashboard Streamlit encerrado.")
        
        print_section("Pipeline Concluído")
        print(f"Data e Hora de Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 