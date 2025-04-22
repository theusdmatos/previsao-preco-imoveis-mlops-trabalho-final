FROM python:3.10

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sqlalchemy==1.4.46

COPY . .
RUN mkdir -p processed_data mlruns monitoring_reports inference_logs

EXPOSE 8000 5000 8501

RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &' >> /app/start.sh && \
    echo 'sleep 5' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo 'if [ ! -f "processed_data/preprocessor.pkl" ]; then' >> /app/start.sh && \
    echo '  python data_prep.py' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo 'if [ ! -f "mlflow.db" ] || [ ! -s "mlflow.db" ]; then' >> /app/start.sh && \
    echo '  python train.py' >> /app/start.sh && \
    echo '  python promote_model.py' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo 'streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 &' >> /app/start.sh && \
    echo 'uvicorn api:app --host 0.0.0.0 --port 8000' >> /app/start.sh && \
    chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"] 