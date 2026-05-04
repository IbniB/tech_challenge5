from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_health_check_no_model():
    """O Health Check deve responder, mas indicará 'degraded' se não houver modelo pré-carregado no MLflow."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["version"] == "2.0.0"

def test_predict_endpoint_rejection():
    """Garante que a API processa ou rejeita o payload de predição adequadamente (se modelo estiver em fallback)."""
    payload = {
        "data": [[[0.5] * 12] * 60], # Formato Tensor 3D Correto: 1 sample, 60 timesteps, 12 features
        "asset_id": "PETR4.SA"
    }
    response = client.post("/infer", json=payload)
    # 503 ocorre se MLFlow não estiver rodando ou não tiver modelo (Comportamento de Fallback Seguro - GAP 03)
    # 200 se a máquina rodando o teste tiver o modelo em cache.
    assert response.status_code in (503, 200)

def test_predict_validation_error():
    """Garante o Data Contract (Pydantic): payload mal formatado."""
    payload = {
        "dados_errados": [0.5] * 30, # Estrutura inválida intencional
        "asset_id": "PETR4.SA"
    }
    response = client.post("/infer", json=payload)
    assert response.status_code == 422 # Unprocessable Entity (Rejeição Pydantic correta)
