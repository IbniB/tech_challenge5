# ADR 004: Métrica de Negócio (Sigma-Tolerance) como Critério de Aceite

**Data:** 28/04/2026
**Status:** Aceito
**Autores:** Grupo 05

## Contexto

Avaliar redes neurais preditivas de séries temporais usando apenas MSE (Mean Squared Error) ou MAE (Mean Absolute Error) gera desconexão com o negócio (Tesouraria). Um MSE baixo não garante que o modelo esteja acertando a direção do mercado de forma útil para alocação de risco. O GAP 05 detectado no projeto original destacava a ausência de uma métrica traduzível para o board de diretoria.

## Decisão

Instituiu-se a métrica de negócio **$\sigma$-tolerance (Sigma-Tolerance)**.

- **Definição:** A predição do modelo é considerada "aceitável" para aquele dia se a diferença entre a predição e o valor real for menor que 0.5 vezes o desvio-padrão (Sigma) histórico da janela do ativo.
- **Critério Go/No-Go:** O pipeline de CI/CD e MLflow exigem que **≥ 70%** das predições do dataset de teste estejam dentro dessa tolerância Sigma.

## Consequências

- **Alinhamento:** A área de risco agora consegue entender a eficácia do modelo ("75% das vezes o modelo erra menos do que a oscilação normal diária").
- **Governança:** Modelos que caem abaixo de 70% (`meets_70pct_criterion=0`) são bloqueados automaticamente pelo MLflow de entrarem em produção.
