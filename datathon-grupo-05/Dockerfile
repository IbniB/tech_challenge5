# Imagem enxuta baseada em Python Oficial (Solução do MLOps Gap: AMBIENTES DEV vs PROD INSTAVEL)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=5
ENV POETRY_INSTALLER_PARALLEL=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.2

RUN poetry config virtualenvs.create false && \
    poetry config installer.max-workers 1 && \
    poetry config installer.modern-installation false

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root --no-interaction --no-ansi

# chromadb necessário para src/agent/rag_pipeline.py (langchain_chroma)
# instalado via pip pois ainda não está no poetry.lock
RUN pip install --no-cache-dir "chromadb>=0.5.0,<0.6.0"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
