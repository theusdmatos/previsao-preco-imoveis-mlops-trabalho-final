# Pipeline MLOps para Previsão de Preços de Imóveis

Este projeto implementa um pipeline de MLOps completo para previsão de preços de imóveis usando dados do King County, EUA (https://www.kaggle.com/datasets/harlfoxem/housesalesprediction). Desenvolvi este sistema como trabalho final da disciplina de MLOps cursada na UFRGS do Curso de Especialização em Engenharia de Software para Aplicações de Ciência de Dados, incluindo todas as etapas desde a preparação de dados até o monitoramento em produção.

## Estrutura do Projeto
```
├── dataset/                   # Dados brutos
│   └── kc_house_data.csv      # Dataset de preços de imóveis
├── processed_data/            # Dados processados (Gerada na primeira execução ex: python run_pipeline.py)
├── monitoring_reports/        # Relatórios de monitoramento (Gerada na primeira execução ex: python run_pipeline.py)
├── inference_logs/            # Logs de inferência  (Gerada na primeira execução ex: python run_pipeline.py)
├── data_prep.py               # Script para preparação de dados
├── train.py                   # Treinamento de modelos com MLflow
├── promote_model.py           # Promoção de modelos
├── api.py                     # API FastAPI para servir o modelo
├── monitor.py                 # Monitoramento de drift e gatilho de retreinamento
├── dashboard.py               # Dashboard interativo com Streamlit
├── requirements.txt           # Dependências do projeto
├── Dockerfile                 # Configuração para conteinerização
├── run_pipeline.py            # Script de automação do pipeline
└── README.md                  # Este arquivo
```

## Requisitos

- Python 3.10+
- Bibliotecas listadas em `requirements.txt`

## Instalação

### Opção 1: Ambiente Virtual com venv

1. Clone o repositório:
```bash
git clone https://github.com/theusdmatos/previsao-preco-imoveis-mlops-trabalho-final.git
cd previsao-preco-imoveis-mlops
```

2. Crie e ative um ambiente virtual (se quiser):
```bash
python -m venv venv
# No Windows
venv\Scripts\activate
# No Linux/Mac
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Usar

### Execução Completa do Pipeline

Para executar todo o pipeline de uma vez:

```bash
python run_pipeline.py
```

Isso executa tudo sequencialmente.

### Ou executar cada step

### 1. Preparação dos Dados

Para processar o dataset:

```bash
python data_prep.py
```

Isso vai carregar o dataset bruto, fazer pré-processamento e dividir em treino/teste.

### 2. Treinamento de Modelos

Para treinar os modelos e salvá-los no MLflow:

```bash
python train.py
```

Isso treina três modelos diferentes (Linear, RandomForest e GradientBoosting).

### 3. Promoção de Modelos

Para promover o melhor modelo para produção:

```bash
python promote_model.py
```

### 4. API de Inferência

Para iniciar a API que serve o modelo:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Acesse a documentação da API em: http://localhost:8000/docs

### 5. Monitoramento e Retreinamento

Para detectar data drift e iniciar retreinamento se necessário:

```bash
python monitor.py
```

Pode ser agendado para execução periódica.

### 6. Dashboard Interativo

Visualize métricas e monitore o sistema com o dashboard:

```bash
python -m streamlit run dashboard.py
```

Ou usando:

```bash
python run_pipeline.py --dashboard
```

O dashboard fica disponível em: http://localhost:8501


## Contêinerização com Docker

Para facilitar a implantação em ambientes isolados, criei um Dockerfile completo:

### Construir a imagem

```bash
docker build -t house-price-mlops .
```

### Executar o contêiner

```bash
docker run -p 8000:8000 -p 5000:5000 -p 8501:8501 house-price-mlops
```

Após a execução, você pode acessar:
- API FastAPI: http://localhost:8000/docs
- Servidor MLflow: http://localhost:5000
- Dashboard Streamlit: http://localhost:8501

## Funcionalidades Implementadas

1. **Dashboard Interativo**: Criei uma interface amigável onde é possível visualizar as métricas e o desempenho do modelo de forma intuitiva. Gastei um bom tempo ajustando este dashboard para ter uma boa experiência de usuário.

2. **Simulador de Preços**: Adicionei uma seção no dashboard onde o usuário pode ajustar características de um imóvel e ver a previsão em tempo real. Isso dá mais transparência ao modelo.

3. **Detecção Automática de Drift**: Implementei um sistema de monitoramento que verifica quando os dados começam a mudar em relação aos dados de treino. Foi um desafio entender os conceitos de drift e implementá-los na prática.

4. **Retreinamento Automático**: Quando o drift é detectado, o sistema pode iniciar automaticamente um novo ciclo de treinamento. Tive que pensar bastante na lógica para garantir que isso não criasse loops infinitos.

5. **Visualização das Features Importantes**: O dashboard mostra quais características mais influenciam o preço, o que ajuda na interpretabilidade do modelo.

6. **Rastreabilidade Completa**: Todo o sistema mantém logs detalhados de previsões, experimentos e métricas, o que foi essencial para depurar durante o desenvolvimento.
