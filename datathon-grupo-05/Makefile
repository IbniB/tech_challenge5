# Datathon Grupo 05 - MLOps Analytics

.PHONY: install test api agent data train mlflow all

# Configura o ambiente
install:
	poetry install

# Executa cobertura de testes
test:
	poetry run python -m pytest tests/ --cov=src

# Sobe a API FastAPI
api:
	poetry run uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# Executa terminal interativo do Agente ReAct
agent:
	poetry run python src/agent/react_agent.py

# Processa features dos ativos e RAG
data:
	poetry run python src/features/feature_engineering.py
	poetry run python src/agent/rag_pipeline.py

# Treina o modelo LSTM
train:
	poetry run python src/models/train.py

# Sobe interface do MLflow UI
mlflow:
	poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Pipeline Completo (End-To-End)
all: data train test
