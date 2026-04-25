# Imagem enxuta baseada em Python Oficial (Solução do MLOps Gap: AMBIENTES DEV vs PROD INSTAVEL)
FROM python:3.11-slim

# Evita o buffering de Python injetando logs diretamente no console (Ideal para Docker/Prometheus Watchdogs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Garantir utilitários básicos da cloud se necessário 
RUN apt-get update && apt-get install -y gcc

# Instalação do gerenciador corporativo adotado
RUN pip install poetry

# Configura o poetry para não criar um virtualenv interno no container, já que o próprio Docker *é* o absoluto encapsulador MLOps
RUN poetry config virtualenvs.create false

# Otimização de Layer Docker: Copiamos apenas os manifestos base para abusar do Layer Caching
COPY pyproject.toml poetry.lock* ./

# Instala a malha inteira validada no lock sem travar em modo interativo
RUN poetry install --no-root --no-interaction --no-ansi

# Sobe o esqueleto algoritmico (src)
COPY . .

# Expondo o portão oficial
EXPOSE 8000

# Disparo asíncrono em alta escalabilidade do script do APP
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
