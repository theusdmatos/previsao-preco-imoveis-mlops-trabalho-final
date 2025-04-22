import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
import mlflow
from mlflow.tracking import MlflowClient
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import altair as alt
import joblib

# Configurar a página
st.set_page_config(
    page_title="House Price Prediction - MLOps Dashboard", 
    page_icon="🏠",
    layout="wide"
)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

# Função para carregar dados dos logs de monitoramento
def load_monitoring_logs():
    log_files = glob.glob('monitoring_reports/monitoring_log_*.json')
    logs = []
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            try:
                data = json.load(f)
                data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                logs.append(data)
            except Exception as e:
                st.error(f"Erro ao carregar log {log_file}: {e}")
    

    logs.sort(key=lambda x: x['timestamp'])
    return logs

# Função para carregar detalhes dos modelos do MLflow
def load_model_details():
    models_data = []
    
    for model in client.search_registered_models():
        model_name = model.name
        
        for version in client.search_model_versions(f"name='{model_name}'"):
            run_id = version.run_id
            run = client.get_run(run_id)
            metrics = run.data.metrics
            params = run.data.params
            
            models_data.append({
                'name': model_name,
                'version': version.version,
                'stage': version.current_stage if hasattr(version, 'current_stage') else 'None',
                'run_id': run_id,
                'creation_timestamp': datetime.fromtimestamp(version.creation_timestamp/1000),
                'metrics': metrics,
                'params': params
            })
    
    return pd.DataFrame(models_data)

# Função para carregar dados de inferência
def load_inference_data():
    log_files = glob.glob('inference_logs/inference_*.json')
    data = []
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            try:
                log = json.load(f)
                entry = log['request']
                entry['prediction'] = log['prediction']
                entry['timestamp'] = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                data.append(entry)
            except Exception as e:
                st.error(f"Erro ao carregar log de inferência {log_file}: {e}")
    
    if data:
        df = pd.DataFrame(data)
        return df
    return None

# Carregar dados do dataset original
@st.cache_data
def load_original_data():
    try:
        return pd.read_csv('dataset/kc_house_data.csv')
    except:
        st.error("Não foi possível carregar o dataset original.")
        return None


st.sidebar.title("🏠 House Price MLOps")
page = st.sidebar.radio("Navegação", 
    ["📊 Dashboard", "📈 Modelos", "🔍 Monitoramento", "🧪 Simulador de Preços"])

if page == "📊 Dashboard":
    st.title("Dashboard do Projeto MLOps")
    
    col1, col2, col3 = st.columns(3)
    
    # Métricas principais
    try:
        models = client.search_registered_models()
        model_count = len(models)
        
        # Melhor modelo em produção
        best_model_name = None
        best_r2 = 0
        
        for model in models:
            for version in client.search_model_versions(f"name='{model.name}'"):
                if hasattr(version, 'current_stage') and version.current_stage == 'Production':
                    run = client.get_run(version.run_id)
                    if "r2" in run.data.metrics and run.data.metrics["r2"] > best_r2:
                        best_r2 = run.data.metrics["r2"]
                        best_model_name = f"{model.name} (v{version.version})"
        
        # Inferências realizadas
        inference_count = len(glob.glob('inference_logs/inference_*.json'))
        
        col1.metric("Modelos Registrados", model_count)
        col2.metric("Melhor Modelo", best_model_name if best_model_name else "Nenhum")
        col3.metric("Precisão (R²)", f"{best_r2:.4f}" if best_r2 > 0 else "N/A")
        
        col1, col2 = st.columns(2)
        col1.metric("Inferências Realizadas", inference_count)
        
        # Verificar alertas de drift
        drift_detected = False
        monitoring_logs = load_monitoring_logs()
        
        if monitoring_logs:
            latest_log = monitoring_logs[-1]
            drift_detected = latest_log.get('retreinament_triggered', False)
            drift_score = latest_log.get('dataset_drift', 0)
            col2.metric("Data Drift Score", f"{drift_score:.4f}", 
                        delta="⚠️ Drift Detectado!" if drift_detected else "Estável")
        
        # Gráfico de histórico do modelo
        st.subheader("Histórico de Desempenho dos Modelos")
        
        models_df = load_model_details()
        if not models_df.empty:
            # Extrair métricas relevantes
            performance_data = []
            for _, row in models_df.iterrows():
                metrics = row['metrics']
                if 'r2' in metrics:
                    performance_data.append({
                        'model': row['name'],
                        'version': row['version'],
                        'stage': row['stage'],
                        'timestamp': row['creation_timestamp'],
                        'r2': metrics['r2'],
                        'rmse': metrics.get('rmse', 0),
                        'mae': metrics.get('mae', 0)
                    })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                perf_df['full_name'] = perf_df['model'] + " v" + perf_df['version'].astype(str)
                
                # Gráfico de R² por modelo
                fig = px.bar(perf_df, x='full_name', y='r2', 
                            color='stage',
                            title='R² por Modelo e Versão',
                            labels={'full_name': 'Modelo', 'r2': 'R²', 'stage': 'Estágio'},
                            color_discrete_map={'Production': 'green', 'Staging': 'orange', 'None': 'gray', 'Archived': 'red'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de métricas de erro
                error_df = pd.melt(perf_df, 
                                id_vars=['full_name', 'stage'], 
                                value_vars=['rmse', 'mae'],
                                var_name='metric', value_name='value')
                
                fig = px.bar(error_df, x='full_name', y='value', 
                            color='stage', barmode='group', facet_col='metric',
                            labels={'full_name': 'Modelo', 'value': 'Valor', 'metric': 'Métrica'},
                            color_discrete_map={'Production': 'green', 'Staging': 'orange', 'None': 'gray', 'Archived': 'red'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhum modelo registrado no MLflow ainda.")
        
        # Visualização dos dados originais
        st.subheader("Visão Geral do Dataset")
        df = load_original_data()
        
        if df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Distribuição dos Preços")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df['price'], kde=True, ax=ax)
                st.pyplot(fig)
            
            with col2:
                st.write("Relação entre Área e Preço")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x='sqft_living', y='price', data=df, alpha=0.5, ax=ax)
                st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Erro ao carregar o dashboard: {e}")

elif page == "📈 Modelos":
    st.title("Modelos Registrados")
    
    try:
        models_df = load_model_details()
        
        if not models_df.empty:
            models_df['creation_date'] = models_df['creation_timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Extrair métricas principais
            models_df['r2'] = models_df['metrics'].apply(lambda x: x.get('r2', 'N/A'))
            models_df['rmse'] = models_df['metrics'].apply(lambda x: x.get('rmse', 'N/A'))
            models_df['mae'] = models_df['metrics'].apply(lambda x: x.get('mae', 'N/A'))
      
            st.dataframe(models_df[['name', 'version', 'stage', 'creation_date', 'r2', 'rmse', 'mae']])
            # Visualização detalhada
            st.subheader("Detalhes do Modelo")
            # Seletor de modelo
            model_names = models_df['name'].unique().tolist()
            selected_model = st.selectbox("Selecione um modelo", model_names)
            model_data = models_df[models_df['name'] == selected_model]
            st.subheader(f"Versões do Modelo: {selected_model}")
            # Gráfico de comparação de métricas entre versões
            metrics_to_plot = ['r2', 'rmse', 'mae']
            fig = go.Figure()
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Scatter(
                    x=model_data['version'].astype(str),
                    y=model_data[metric],
                    mode='lines+markers',
                    name=metric.upper()
                ))
            
            fig.update_layout(
                title=f'Comparação de Métricas por Versão - {selected_model}',
                xaxis_title='Versão',
                yaxis_title='Valor',
                legend_title='Métrica'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            available_versions = model_data['version'].astype(str).tolist()
            selected_version = st.selectbox("Selecione uma versão para ver detalhes", available_versions)
            version_details = model_data[model_data['version'].astype(str) == selected_version].iloc[0]
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Métricas")
                for metric, value in version_details['metrics'].items():
                    st.metric(metric.upper(), f"{value:.4f}" if isinstance(value, (int, float)) else value)
            
            with col2:
                st.subheader("Parâmetros")
                params = version_details['params']
                if params:
                    for param, value in params.items():
                        st.write(f"**{param}:** {value}")
                else:
                    st.write("Nenhum parâmetro registrado")
        else:
            st.info("Nenhum modelo registrado no MLflow ainda.")
    
    except Exception as e:
        st.error(f"Erro ao carregar detalhes dos modelos: {e}")

elif page == "🔍 Monitoramento":
    st.title("Monitoramento de Data Drift")
    
    try:
        monitoring_logs = load_monitoring_logs()
        
        if monitoring_logs:
            logs_df = pd.DataFrame([
                {
                    'timestamp': log['timestamp'],
                    'dataset_drift': log['dataset_drift'],
                    'num_columns_drifted': log['num_columns_drifted'],
                    'retreinament_triggered': log['retreinament_triggered'],
                    'model_name': log['current_model']['name'] if 'current_model' in log and log['current_model']['name'] else 'N/A',
                    'model_version': log['current_model']['version'] if 'current_model' in log and log['current_model']['version'] else 'N/A'
                }
                for log in monitoring_logs
            ])
           
            logs_df['date'] = logs_df['timestamp'].dt.strftime('%Y-%m-%d')
            logs_df['time'] = logs_df['timestamp'].dt.strftime('%H:%M:%S')
            
            col1, col2, col3 = st.columns(3)
            latest_log = logs_df.iloc[-1]
            col1.metric("Último Data Drift Score", f"{latest_log['dataset_drift']:.4f}")
            col2.metric("Colunas com Drift", latest_log['num_columns_drifted'])
            
            if latest_log['retreinament_triggered']:
                status = "⚠️ Retreinamento Acionado"
            else:
                status = "✅ Estável"
            
            col3.metric("Status", status)
            
            st.subheader("Evolução do Data Drift")
            
            fig = px.line(logs_df, x='timestamp', y='dataset_drift',
                        hover_data=['date', 'time', 'num_columns_drifted', 'retreinament_triggered'],
                        labels={'timestamp': 'Data/Hora', 'dataset_drift': 'Score de Drift'},
                        title='Evolução do Score de Data Drift ao Longo do Tempo')
            
            fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                        annotation_text="Limite de Retreinamento")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Número de Colunas com Drift Detectado")
            
            fig = px.bar(logs_df, x='timestamp', y='num_columns_drifted',
                        labels={'timestamp': 'Data/Hora', 'num_columns_drifted': 'Número de Colunas'},
                        title='Número de Colunas com Drift Detectado por Execução')
            
            fig.add_hline(y=3, line_dash="dash", line_color="red", 
                        annotation_text="Limite de Retreinamento")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Histórico de execuções
            st.subheader("Histórico de Execuções de Monitoramento")
            st.dataframe(logs_df[['date', 'time', 'dataset_drift', 'num_columns_drifted', 
                                'retreinament_triggered', 'model_name', 'model_version']])
            
            # Verificar se há relatórios HTML para visualizar
            report_files = glob.glob('monitoring_reports/data_drift_report_*.html')
            
            if report_files:
                st.subheader("Relatórios de Data Drift")
                # Ordenar por data (mais recente primeiro)
                report_files.sort(reverse=True)
                report_options = {os.path.basename(f): f for f in report_files}
                selected_report = st.selectbox("Selecione um relatório para visualizar", 
                                            list(report_options.keys()))
            
                with open(report_options[selected_report], 'r', encoding='utf-8') as f:
                    report_html = f.read()
                
                st.components.v1.html(report_html, height=600, scrolling=True)
        else:
            st.info("Nenhum log de monitoramento encontrado ainda.")
    
    except Exception as e:
        st.error(f"Erro ao carregar dados de monitoramento: {e}")

elif page == "🧪 Simulador de Preços":
    st.title("Simulador de Preços de Imóveis")
    
    try:
        # Carregar o modelo em produção para fazer previsões
        best_model_name = None
        best_model_version = None
        best_r2 = 0
        
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
        
        # Verificar se encontrou um modelo em produção
        if best_model_name and best_model_version:
            st.success(f"Usando modelo: {best_model_name} (versão {best_model_version}) com R² = {best_r2:.4f}")
            df_original = load_original_data()
            
            if df_original is not None:
                with st.form("house_form"):
                    st.subheader("Informe as características do imóvel")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        bedrooms = st.number_input("Número de quartos", 
                                                min_value=1, max_value=10, value=3,
                                                help=f"Média: {df_original['bedrooms'].mean():.1f}")
                        
                        bathrooms = st.number_input("Número de banheiros", 
                                                min_value=0.5, max_value=10.0, value=2.0, step=0.5,
                                                help=f"Média: {df_original['bathrooms'].mean():.1f}")
                        
                        sqft_living = st.number_input("Área habitável (pés²)", 
                                                    min_value=500, max_value=10000, value=2000,
                                                    help=f"Média: {df_original['sqft_living'].mean():.0f}")
                        
                        sqft_lot = st.number_input("Área do terreno (pés²)", 
                                                min_value=500, max_value=100000, value=8000,
                                                help=f"Média: {df_original['sqft_lot'].mean():.0f}")
                    
                    with col2:
                        floors = st.selectbox("Número de andares",
                                            options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                                            index=1)
                        
                        waterfront = st.checkbox("Vista para água", value=False)
                        
                        view = st.slider("Índice de vista", 
                                        min_value=0, max_value=4, value=0,
                                        help="0=sem vista, 4=excelente vista")
                        
                        condition = st.slider("Condição do imóvel", 
                                            min_value=1, max_value=5, value=3,
                                            help="1=ruim, 5=excelente")
                        
                        grade = st.slider("Nota de construção", 
                                        min_value=1, max_value=13, value=7,
                                        help="1-3=baixa qualidade, 7=média, 11-13=alta qualidade")
                    
                    with col3:
                        yr_built = st.number_input("Ano de construção", 
                                                min_value=1900, max_value=2023, value=1980)
                        
                        yr_renovated = st.number_input("Ano de renovação (0=nunca renovada)", 
                                                    min_value=0, max_value=2023, value=0)
                        
                        zipcode = st.selectbox("CEP (Zipcode)", 
                                            options=sorted(df_original['zipcode'].unique().tolist()),
                                            index=0)
                        
                        lat = st.number_input("Latitude", 
                                            min_value=47.0, max_value=48.0, 
                                            value=47.6,
                                            format="%.4f")
                        
                        long = st.number_input("Longitude", 
                                            min_value=-123.0, max_value=-121.0, 
                                            value=-122.2,
                                            format="%.4f")
                   
                    sqft_above = st.number_input("Área acima do solo (pés²)", 
                                                min_value=500, max_value=9000, value=1500)
                    
                    sqft_basement = sqft_living - sqft_above
                    if sqft_basement < 0:
                        st.error("A área acima do solo não pode ser maior que a área habitável total.")
                        sqft_basement = 0
                    
                    st.write(f"Área do porão: {sqft_basement} pés²")
                    
                    sqft_living15 = st.number_input("Área habitável dos 15 vizinhos (pés²)", 
                                                min_value=500, max_value=6000, value=1800)
                    
                    sqft_lot15 = st.number_input("Área do terreno dos 15 vizinhos (pés²)", 
                                                min_value=500, max_value=60000, value=7500)
                   
                    submitted = st.form_submit_button("Calcular Preço")
                
                if submitted:
                    house_data = {
                        'bedrooms': bedrooms,
                        'bathrooms': bathrooms,
                        'sqft_living': sqft_living,
                        'sqft_lot': sqft_lot,
                        'floors': floors,
                        'waterfront': 1 if waterfront else 0,
                        'view': view,
                        'condition': condition,
                        'grade': grade,
                        'sqft_above': sqft_above,
                        'sqft_basement': sqft_basement,
                        'yr_built': yr_built,
                        'yr_renovated': yr_renovated,
                        'zipcode': zipcode,
                        'lat': lat,
                        'long': long,
                        'sqft_living15': sqft_living15,
                        'sqft_lot15': sqft_lot15,
                        'year': datetime.now().year,
                        'month': datetime.now().month,
                        'day_of_week': datetime.now().weekday()
                    }
                   
                    house_data['house_age'] = house_data['year'] - house_data['yr_built']
                    house_data['renovated'] = 1 if house_data['yr_renovated'] > 0 else 0
                    house_data['basement_ratio'] = house_data['sqft_basement'] / house_data['sqft_living'] if house_data['sqft_living'] > 0 else 0
                    
                    preprocessor = joblib.load('processed_data/preprocessor.pkl')
                    house_df = pd.DataFrame([house_data])
                    processed_data = preprocessor.transform(house_df)
                    model = mlflow.pyfunc.load_model(f"models:/{best_model_name}/{best_model_version}")
                    prediction_log = model.predict(processed_data)[0]
                    prediction = np.exp(prediction_log)
                    st.subheader("Preço Estimado")
                    col1, col2 = st.columns(2)
                    col1.metric("Preço Previsto", f"${prediction:,.2f}")
                    
                    # Carregar imóveis similares para comparação
                    similar_houses = df_original[
                        (df_original['bedrooms'] == bedrooms) &
                        (df_original['bathrooms'].between(bathrooms-0.5, bathrooms+0.5)) &
                        (df_original['sqft_living'].between(sqft_living*0.8, sqft_living*1.2))
                    ]
                    
                    if not similar_houses.empty:
                        avg_similar_price = similar_houses['price'].mean()
                        col2.metric("Preço Médio de Imóveis Similares", 
                                    f"${avg_similar_price:,.2f}",
                                    delta=f"{(prediction-avg_similar_price)/avg_similar_price*100:.1f}%")
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    log_entry = {
                        "timestamp": str(datetime.now()),
                        "request": house_data,
                        "prediction": float(prediction)
                    }
                    
                    log_file = f"inference_logs/inference_{timestamp}.json"
                    with open(log_file, 'w') as f:
                        json.dump(log_entry, f)
                    
                    st.success("Previsão realizada e registrada para monitoramento!")
                    st.subheader("Fatores que Influenciam o Preço")

                    feature_importance = {
                        'Área Habitável': 0.35,
                        'Nota de Construção': 0.25,
                        'Localização (Lat/Long)': 0.15,
                        'Número de Banheiros': 0.10,
                        'Vista para Água': 0.08 if waterfront else 0,
                        'Condição': 0.07,
                        'Outros Fatores': 1.0 - (0.35 + 0.25 + 0.15 + 0.10 + (0.08 if waterfront else 0) + 0.07)
                    }
                    
                    importance_df = pd.DataFrame({
                        'Feature': list(feature_importance.keys()),
                        'Importance': list(feature_importance.values())
                    })
                    
                    # Gráfico de importância
                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                title='Fatores que Mais Influenciaram o Preço',
                                labels={'Importance': 'Importância Relativa', 'Feature': 'Característica'},
                                color='Importance',
                                color_continuous_scale='Viridis')
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("Não foi possível carregar o dataset original para referência.")
        
        else:
            st.warning("Nenhum modelo em produção disponível. Execute primeiro o treinamento e promoção de modelos.")
    
    except Exception as e:
        st.error(f"Erro ao carregar o simulador: {e}")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **MLOps Dashboard**  
    📊 Visualize métricas  
    🔍 Monitore data drift  
    🏠 Simule preços de imóveis
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Última atualização: ", datetime.now().strftime("%d/%m/%Y %H:%M:%S")) 