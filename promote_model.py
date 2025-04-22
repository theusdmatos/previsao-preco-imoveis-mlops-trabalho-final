import mlflow
from mlflow.tracking import MlflowClient
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

def get_best_model():
    best_model_name = None
    best_model_version = None
    best_r2 = -float('inf')
    
    for model in client.search_registered_models():
        model_name = model.name
        for model_version in client.search_model_versions(f"name='{model_name}'"):
            run_id = model_version.run_id
            run = client.get_run(run_id)
            if "r2" in run.data.metrics:
                r2 = run.data.metrics["r2"]
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = model_name
                    best_model_version = model_version.version
    
    if best_model_name is None:
        logger.warning("Nenhum modelo com métrica R² encontrado!")
        return None, None
    
    logger.info(f"Melhor modelo: {best_model_name} versão {best_model_version} com R² = {best_r2:.4f}")
    return best_model_name, best_model_version

def set_model_to_staging(model_name, model_version):
   
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging"
        )
        logger.info(f"Modelo {model_name} versão {model_version} definido como Staging")
    except Exception as e:
        logger.error(f"Erro ao definir modelo como Staging: {e}")
        raise

def set_model_to_production(model_name, model_version):
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production"
        )
        logger.info(f"Modelo {model_name} versão {model_version} definido como Production")
    except Exception as e:
        logger.error(f"Erro ao definir modelo como Production: {e}")
        raise

def get_production_model():
    production_model_name = None
    production_model_version = None
    
    for model in client.search_registered_models():
        model_name = model.name
        try:
            for model_version in client.search_model_versions(f"name='{model_name}' AND stage='Production'"):
                production_model_name = model_name
                production_model_version = model_version.version
                logger.info(f"Modelo em produção: {production_model_name} versão {production_model_version}")
                return production_model_name, production_model_version
        except Exception as e:
            logger.warning(f"Erro ao buscar modelos em produção: {e}")
    
    if production_model_name is None:
        logger.warning("Nenhum modelo em produção encontrado.")
    return production_model_name, production_model_version

def main():
    print("Promovendo modelos...")
    models = client.search_registered_models()
    if not models:
        logger.error("Nenhum modelo registrado encontrado. Execute primeiro o treinamento.")
        exit(1)
    best_model_name, best_model_version = get_best_model()
    
    if best_model_name is None:
        logger.error("Não foi possível encontrar o melhor modelo.")
        exit(1)
    
    set_model_to_staging(best_model_name, best_model_version)
    production_model_name, production_model_version = get_production_model()
    if production_model_name is None:
        set_model_to_production(best_model_name, best_model_version)
        logger.info(f"Modelo {best_model_name} versão {best_model_version} promovido para produção.")
    else:
        best_run = client.get_run(client.get_model_version(best_model_name, best_model_version).run_id)
        prod_run = client.get_run(client.get_model_version(production_model_name, production_model_version).run_id)
        
        best_r2 = best_run.data.metrics.get("r2", 0)
        prod_r2 = prod_run.data.metrics.get("r2", 0)
        
        improvement = (best_r2 - prod_r2) / prod_r2 * 100 if prod_r2 > 0 else float('inf')
        
        if improvement > 1.0:
            set_model_to_production(best_model_name, best_model_version)
            logger.info(f"Modelo {best_model_name} versão {best_model_version} promovido para produção.")
            logger.info(f"Melhoria de R²: {improvement:.2f}% (de {prod_r2:.4f} para {best_r2:.4f})")
        else:
            logger.info(f"Modelo atual em produção ({production_model_name} v{production_model_version}) mantido.")
            logger.info(f"Melhoria insuficiente: {improvement:.2f}% (menor que limite de 1%)")
    
    #Exibir o modelo atual em produção
    production_model_name, production_model_version = get_production_model()
    if production_model_name:
        prod_run = client.get_run(client.get_model_version(production_model_name, production_model_version).run_id)
        prod_r2 = prod_run.data.metrics.get("r2", 0)
        logger.info(f"Modelo atual em produção: {production_model_name} v{production_model_version}, R² = {prod_r2:.4f}")

if __name__ == "__main__":
    main() 