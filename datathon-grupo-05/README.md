# Datathon Grupo 05

Este repositório implementa um projeto completo para a fase 05 do Datathon, incluindo:

- pipeline de treinamento reprodutível com MLflow
- modelo baseline de classificação binária
- API FastAPI para inferência e agente de demonstração
- monitoramento de drift e métricas
- guardrails de segurança e detecção de PII
- documentação de governança e entrega
- testes automatizados com `pytest`

## Começando

1. Copie o template de ambiente:
   ```bash
   cp .env.example .env
   ```
2. Instale dependências:
   ```bash
   make install
   ```
3. Execute treinamento:
   ```bash
   make train
   ```
4. Inicie a API:
   ```bash
   make serve
   ```

## Estrutura do projeto

- `src/` - código de features, modelos, agente, monitoramento, segurança e serving
- `configs/` - hiperparâmetros de modelo e thresholds de drift
- `tests/` - cobertura de testes unitários e de API
- `docker-compose.yml` - ambiente local com MLflow e API
- `dvc.yaml` - pipeline de treino reprodutível
- `.env.example` - template de variáveis sensíveis
