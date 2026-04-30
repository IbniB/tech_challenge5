# ADR 003: Governança de Modelos e Métrica de Negócio (σ-tolerance)

**Data:** 28/04/2026
**Status:** Aceito
**Autores:** Grupo 05

## Contexto

Em arquiteturas MLOps corporativas, não basta acompanhar métricas matemáticas genéricas como RMSE (Root Mean Squared Error) ou Loss. Modelos financeiros exigem rastreabilidade regulatória e critérios de aceite vinculados diretamente às metas de negócio da tesouraria.

## Decisão

A governança do projeto foi inteiramente acoplada ao **MLflow** com requisitos estritos:

1. **Rastreamento via Metadata Obrigatório (Tags):** Todo treinamento de rede neural LSTM obrigatoriamente loga a versão do código (`git_sha`), dono, tipo de modelo e criticidade de risco. Modelos sem essas tags são bloqueados de promoção.
2. **Implementação da Métrica de Negócio (σ-tolerance):** Criamos a métrica de tolerância Sigma, que exige que no mínimo **70% das predições do modelo estejam num raio de 0.5 desvios-padrões** (0.5σ) do valor real. 
3. **Decisão Automática (Go/No-Go):** O script `train.py` foi atualizado para marcar a flag boolean `meets_70pct_criterion`. Somente modelos aprovados nesta regra são classificados como adequados pela API de Serving.

## Consequências

- O gap qualitativo (GAP 05) de "modelos órfãos" foi eliminado.
- Qualquer desenvolvedor no time pode auditar a rastreabilidade do experimento pelo SQLite do MLflow, garantindo segurança na validação algorítmica da CVM.
