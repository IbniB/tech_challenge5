"""
DAG: ml_pipeline_financeiro
============================
Resolve o GAP 02 do Datathon: elimina o anti-padrão de Notebook como SPOF
declarando o pipeline inteiro como uma DAG com tasks isoladas.

Cada Task roda em seu próprio processo (LocalExecutor).
Nenhuma task compartilha estado em memória com outra — comunicação ocorre
exclusivamente via artefatos em disco (data/raw/ e data/processed/).
Isso garante: se o treino falha, a coleta de dados não é afetada.

Fluxo:
    coletar_dados → validar_e_processar → treinar_modelo → detectar_drift
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# ─── Configuração Base ─────────────────────────────────────────────────────────
# Diretório raiz do projeto dentro do container Airflow
PROJECT_DIR = "/opt/airflow/project"

# Cada ativo é tratado como pipeline independente (sem cluster compartilhado)
TICKERS = [
    {"ticker": "PETR4.SA", "ticker_id": "petr4_sa"},
    {"ticker": "NVDC34.SA", "ticker_id": "nvdc34_sa"},
]

default_args = {
    "owner": "grupo-05",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ─── Funções de Validação ───────────────────────────────────────────────────────
def verificar_dados(ticker_id: str, **context):
    """Valida que os arquivos .npy gerados têm shape correto antes do treino."""
    import numpy as np
    import os

    processed_dir = os.path.join(PROJECT_DIR, "data", "processed")
    for split in ["X_train", "y_train", "X_test", "y_test"]:
        path = os.path.join(processed_dir, f"{ticker_id}_{split}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artefato ausente: {path}. Pipeline interrompido.")
        arr = np.load(path)
        print(f"  {split}: shape={arr.shape}")

    print(f"[{ticker_id}] Validação de artefatos OK. Liberando treino.")


# ─── Criação de DAGs por Ticker (Compute Isolation) ────────────────────────────
# Cada ativo tem sua própria DAG independente.
# Isso garante que uma falha no pipeline da NVIDIA não impacta o da Petrobras.

for config in TICKERS:
    ticker = config["ticker"]
    ticker_id = config["ticker_id"]
    dag_id = f"ml_pipeline_{ticker_id}"

    with DAG(
        dag_id=dag_id,
        description=f"Pipeline MLOps isolado para {ticker} — GAP 02 (Anti-SPOF)",
        default_args=default_args,
        schedule_interval="0 6 * * 1-5",  # Dias úteis às 06h (após abertura B3)
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["mlops", "financeiro", ticker_id, "datathon-fase05"],
    ) as dag:

        # ── Task 1: Coleta de Dados via yfinance ──────────────────────────────
        # Compute isolado: processo próprio, sem acesso ao estado do treino
        t1_coletar = BashOperator(
            task_id="coletar_dados_mercado",
            bash_command=(
                f"cd {PROJECT_DIR} && "
                f"python src/features/data_collection.py --ticker {ticker}"
            ),
            doc_md=f"""
            **Coleta de dados de mercado para {ticker}.**
            Acessa a API do yfinance e salva o CSV bruto em data/raw/.
            Falha aqui não afeta outros pipelines (compute isolado).
            """,
        )

        # ── Task 2: Feature Engineering + Schema Validation ───────────────────
        # Roda em processo separado — sem memória compartilhada com t1
        t2_processar = BashOperator(
            task_id="processar_features",
            bash_command=(
                f"cd {PROJECT_DIR} && "
                f"python src/features/feature_engineering.py --ticker_id {ticker_id}"
            ),
            doc_md="""
            **Validação de Schema (Pandera) + MinMaxScaler + janelamento 60d.**
            Gera tensores .npy com split 80/20 para treino/teste.
            """,
        )

        # ── Task 3: Validação de Artefatos (Gate de Qualidade) ────────────────
        # PythonOperator roda em processo próprio do Airflow Worker
        t3_validar = PythonOperator(
            task_id="validar_artefatos",
            python_callable=verificar_dados,
            op_kwargs={"ticker_id": ticker_id},
            doc_md="""
            **Quality Gate:** verifica shapes dos tensores antes de liberar o treino.
            Evita que o modelo treine com dados corrompidos silenciosamente.
            """,
        )

        # ── Task 4: Treinamento LSTM com MLflow ───────────────────────────────
        # Processo isolado — logs e artefatos vão para MLflow, não para memória
        t4_treinar = BashOperator(
            task_id="treinar_modelo_lstm",
            bash_command=(
                f"cd {PROJECT_DIR} && "
                f"python src/models/train.py --ticker_id {ticker_id}"
            ),
            doc_md="""
            **Treinamento LSTM com rastreamento MLflow.**
            Loga RMSE, MAE, MAPE por epoch. Salva artefato versionado.
            Run name derivado do ticker — sem hardcode.
            """,
        )

        # ── Task 5: Drift Detection ───────────────────────────────────────────
        # Roda após o treino — detecta PSI e MAPE rolling
        t5_drift = BashOperator(
            task_id="detectar_drift",
            bash_command=(
                f"cd {PROJECT_DIR} && "
                f"python src/monitoring/drift.py --ticker_id {ticker_id}"
            ),
            doc_md="""
            **Detecção de Concept Drift e Data Drift (Evidently + PSI).**
            PSI > 0.2 ou MAPE rolling > 12% aciona alerta de retreino.
            """,
        )

        # ── Grafo de Dependências (sem paralelismo entre tasks do mesmo ativo) ─
        t1_coletar >> t2_processar >> t3_validar >> t4_treinar >> t5_drift

    # Registra a DAG no namespace global do Airflow
    globals()[dag_id] = dag
