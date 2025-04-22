import os
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='api_logs.log'
)
logger = logging.getLogger("house-price-api")


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


app = FastAPI(
    title="House Price Prediction API",
    description="API para previsão de preços de imóveis usando modelos MLOps",
    version="1.0.0"
)

# Diretório para armazenamento dos logs de inferência
if not os.path.exists('inference_logs'):
    os.makedirs('inference_logs')
preprocessor = joblib.load('processed_data/preprocessor.pkl')

# Modelo para solicitação de previsão individual
class HousePredictionRequest(BaseModel):
    bedrooms: int = Field(..., example=3, description="Número de quartos")
    bathrooms: float = Field(..., example=2.5, description="Número de banheiros")
    sqft_living: int = Field(..., example=2100, description="Área habitável em pés quadrados")
    sqft_lot: int = Field(..., example=5000, description="Área do terreno em pés quadrados")
    floors: float = Field(..., example=2.0, description="Número de andares")
    waterfront: int = Field(..., example=0, description="Se tem vista para água (0=não, 1=sim)")
    view: int = Field(..., example=0, description="Índice de vista (0-4)")
    condition: int = Field(..., example=3, description="Condição do imóvel (1-5)")
    grade: int = Field(..., example=8, description="Nota do imóvel (1-13)")
    sqft_above: int = Field(..., example=1800, description="Área acima do solo em pés quadrados")
    sqft_basement: int = Field(..., example=300, description="Área do porão em pés quadrados")
    yr_built: int = Field(..., example=1995, description="Ano de construção")
    yr_renovated: int = Field(..., example=0, description="Ano de renovação (0=nunca renovada)")
    zipcode: int = Field(..., example=98038, description="CEP da região")
    lat: float = Field(..., example=47.5396, description="Latitude")
    long: float = Field(..., example=-122.0431, description="Longitude")
    sqft_living15: int = Field(..., example=1890, description="Área habitável dos 15 vizinhos mais próximos")
    sqft_lot15: int = Field(..., example=6000, description="Área do terreno dos 15 vizinhos mais próximos")
    year: Optional[int] = Field(None, example=2023, description="Ano de venda (padrão: atual)")
    month: Optional[int] = Field(None, example=7, description="Mês de venda (padrão: atual)")
    day_of_week: Optional[int] = Field(None, example=2, description="Dia da semana (0-6) (padrão: atual)")

# Modelo para solicitação de previsão em lote
class BatchPredictionRequest(BaseModel):
    houses: List[HousePredictionRequest]

# Modelo para resposta de previsão individual
class HousePredictionResponse(BaseModel):
    price: float = Field(..., example=450000.0, description="Preço previsto em dólares")

# Modelo para resposta de previsão em lote
class BatchPredictionResponse(BaseModel):
    predictions: List[float] = Field(..., description="Lista de preços previstos em dólares")

def get_production_model():
    client = mlflow.tracking.MlflowClient()
    best_model_name = None
    best_model_version = None
    best_r2 = -float('inf')
    
    for model in client.search_registered_models():
        model_name = model.name
        
        for model_version in client.search_model_versions(f"name='{model_name}'"):
            if hasattr(model_version, 'current_stage') and model_version.current_stage == 'Production':
                run_id = model_version.run_id
                run = client.get_run(run_id)
                metrics = run.data.metrics
                
                if "r2" in metrics and metrics["r2"] > best_r2:
                    best_r2 = metrics["r2"]
                    best_model_name = model_name
                    best_model_version = model_version.version
    
    if best_model_name is None or best_model_version is None:
        # Se não encontrar nenhum modelo em produção, pegar o melhor modelo disponível
        for model in client.search_registered_models():
            model_name = model.name
            
            for model_version in client.search_model_versions(f"name='{model_name}'"):
                run_id = model_version.run_id
                run = client.get_run(run_id)
                metrics = run.data.metrics
                
                if "r2" in metrics and metrics["r2"] > best_r2:
                    best_r2 = metrics["r2"]
                    best_model_name = model_name
                    best_model_version = model_version.version
    
    if best_model_name is None or best_model_version is None:
        raise Exception("Nenhum modelo em produção ou modelo alternativo encontrado. Execute o treinamento primeiro.")
    
    # Carregando o modelo
    model = mlflow.pyfunc.load_model(f"models:/{best_model_name}/{best_model_version}")
    return model, best_model_name, best_model_version

def preprocess_input(house_data):
    if isinstance(house_data, list):
        df = pd.DataFrame([h.dict() for h in house_data])
    else:
        df = pd.DataFrame([house_data.dict()])
    
    current_date = datetime.now()
    if 'year' not in df.columns or df['year'].isnull().any():
        df['year'] = current_date.year
    if 'month' not in df.columns or df['month'].isnull().any():
        df['month'] = current_date.month
    if 'day_of_week' not in df.columns or df['day_of_week'].isnull().any():
        df['day_of_week'] = current_date.weekday()
    
    df['house_age'] = df['year'] - df['yr_built']
    df['renovated'] = (df['yr_renovated'] > 0).astype(int)
    df['basement_ratio'] = df['sqft_basement'] / df['sqft_living']
    df['basement_ratio'] = df['basement_ratio'].fillna(0)
    
    processed_data = preprocessor.transform(df)
    return processed_data

def log_inference(request_data, prediction):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_entry = {
        "timestamp": str(datetime.now()),
        "request": request_data,
        "prediction": prediction
    }
    
    log_file = f"inference_logs/inference_{timestamp}.json"
    with open(log_file, "w") as f:
        json.dump(log_entry, f)

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API. Use /predict para fazer previsões."}

@app.post("/predict", response_model=HousePredictionResponse)
def predict(house: HousePredictionRequest):
    try:
        # Carregar o modelo em produção
        model, _, _ = get_production_model()
        processed_data = preprocess_input(house)
        log_prediction = model.predict(processed_data)[0]
        prediction = float(np.exp(log_prediction))
        log_inference(house.dict(), prediction)
        return {"price": prediction}
    
    except Exception as e:
        logger.error(f"Erro na previsão: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest):
    try:
    
        model, _, _ = get_production_model()
        processed_data = preprocess_input(request.houses)
        # Fazer as previsões
        log_predictions = model.predict(processed_data)
        predictions = [float(np.exp(p)) for p in log_predictions]
        for i, house in enumerate(request.houses):
            log_inference(house.dict(), predictions[i])
        return {"predictions": predictions}
    
    except Exception as e:
        logger.error(f"Erro na previsão em lote: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 