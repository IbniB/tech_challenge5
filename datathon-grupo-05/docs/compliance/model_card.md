# Model Card: LSTM para Previsão de Ativos

## 1. Detalhes do Modelo
- **Arquitetura:** Long Short-Term Memory (LSTM) Multivariável.
- **Entradas:** Janela temporal de 60 dias com 12 features técnicas (Preço, Volume, Retornos, SMA_20, EMA_20, Bollinger Bands, RSI_14, MACD).
- **Saídas:** Preço de fechamento previsto para o próximo dia útil (t+1), em escala normalizada ou absoluta.
- **Framework:** PyTorch.
- **Treinamento:** Otimizador Adam, MSE Loss, split cronológico 80/20.

## 2. Intenção de Uso
Prever a direção direcional e o preço estimado de fechamento (t+1) de ativos listados na B3 (PETR4) e BDRs (NVDC34) para auxiliar nas decisões da mesa de operações e tesouraria.

## 3. Limitações Críticas
- **Falta de Contexto Macroeconômico:** O modelo não consome notícias, indicadores macro (Selic, Inflação) ou eventos sistêmicos. Ele baseia-se puramente em Análise Técnica e momentum.
- **Volatilidade Extrema (Cisnes Negros):** O modelo sofre de degradação rápida em cenários de alta incerteza (ex: greves, quebras financeiras), onde o passado imediato não reflete o futuro.

## 4. Métricas de Avaliação
- **MSE/MAE:** Utilizado para calcular o erro médio nas janelas de teste.
- **σ-tolerance:** Métrica primária de negócio. Exige que $\geq$ 70% das previsões caiam em um raio de 0.5 Desvios Padrões ($\sigma$) do valor real do ativo.
