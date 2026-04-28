# Tech Challenge (Fase 05) - MLOps Analytics para Tesouraria e Mercado Financeiro

Este repositório concentra a entrega final do Tech Challenge, unindo o desenvolvimento de algoritmos de Deep Learning para previsão de Séries Temporais (Fase 04) com as práticas de Engenharia e Arquitetura MLOps exigidas na Fase 05.

---

## O Problema de Negócio

Em mesas de operações e tesourarias, modelos de análise técnica e predição de preços de ações são muito comuns. Porém, o maior desafio prático para escalar isso não é a precisão matemática do modelo, mas sim o seu ciclo de vida. Modelos não monitorados sofrem degradação rápida devido a mudanças econômicas (Concept Drift) ou falhas na extração de dados (Data Drift).

Se um sistema tomar decisões automatizadas baseado em uma rede neural descalibrada, os prejuízos para instituições financeiras sobem exponencialmente.

### Nossa Solução Aplicada
Em vez de focar apenas no desenvolvimento isolado da Rede Neural, construímos e refatoramos uma infraestrutura conectada, que conta com:
- Predição técnica de séries temporais usando redes de memória de longo prazo (LSTM) sobre a fundação do PyTorch.
- API construída em FastAPI encapsulando os pesos resultantes do modelo.
- Acoplamento do modelo a um agente generativo (ReAct / LangChain) que responde perguntas financeiras validando os dados na web e cruzando com as políticas de risco da corretora.
- Monitoramento contínuo usando MLflow e Prometheus para auditoria.

---

## Decisões de Arquitetura MLOps

Para suprir as exigências da banca da Fase 05, tomamos as seguintes ações para sanar gaps de mercado:

### 1. Versionamento Base com DVC
Não subimos as extrações locais e massivas diretamente no Git. Dados brutos e tratamentos intermediários foram abstraídos via DVC, melhorando a organização do repositório e permitindo que qualquer membro novo clone o projeto e retome as modelagens a partir de "Data states" rastreáveis.

### 2. CI/CD Automatizado e Isolamento de Codificação
Mudamos a cultura do exploratório padrão. Processamentos de dados rodando em Notebooks descartáveis dão lugar para rotinas de extração corporativas na raiz da pasta `src/`. Esses scripts possuem testes de integração via Pytest e auditoria de nulos usando Data Contracts (Pandera). Isso ameniza o risco de códigos mortos.

### 3. Rastreamento (MLflow) e Observabilidade (Prometheus)
A etapa de Treinamento da LSTM ocorre injetando o MLflow no código. Ele salvará cada hiperparâmetro (como tamanho de janelas) e o artefato de modelo gerado no final do processo, mantendo registro claro de autoria de cada versão. Na ponta da inferência, integramos de cara um endpoint Promotheus consumido assincronamente via Docker para entender picos de uso e flutuações anormais e bruscas no preço das ações requisitadas.

### 4. Inteligência RAG e Agentes Generativos
Aplicamos bibliotecas GenAI para dar utilidade para as inferências da predição. O Agente cruza três "Tools" do próprio sistema: Busca a última cotação legítima na nuvem via yfinance, solicita predição interna do nosso modelo neural, e cruza o output sob checagens com uma base RAG onde subimos as diretrizes corporativas restritas da CVM. Tudo isso avaliado com Ragas para comprovar falta de alucinação sintética.

---

## Por que LSTM? Decisão de Modelagem e o que Esperar dos Resultados

A escolha do algoritmo não foi arbitrária. Avaliamos as alternativas antes de fixar a arquitetura.

Modelos como Prophet são eficientes para capturar sazonalidade explícita (feriados, ciclos mensais), mas têm limitações sérias com a volatilidade não-linear do mercado financeiro, especialmente em ativos de alta liquidez como PETR4.SA e NVDC34.SA. Redes convolucionais (CNNs para séries temporais) funcionam bem em padrões locais de curto prazo, mas perdem contexto de janelas mais longas.

A LSTM (Long Short-Term Memory) foi projetada especificamente para memória sequencial de longo prazo, resolvendo o problema de gradientes que desaparecem em RNNs simples. Para previsão de preços de fechamento usando uma janela de 60 dias, isso se traduz em: o modelo consegue "lembrar" que uma tendência de queda de três semanas atrás ainda é relevante para a predição de amanhã. Esse é exatamente o comportamento esperado em análise técnica de mercado.

### O que Defender sobre Desvios Padrão e Variância

Séries de preços financeiros são intrinsecamente ruidosas. Qualquer modelo que apresente erros próximos a zero em dados de teste deve ser questionado — provavelmente está sobreajustado. O que esperamos e defendemos:

- **MAPE entre 2% e 8%** é considerado competitivo para previsão de curto prazo em ações voláteis. Acima de 15% indica que o modelo perdeu a tendência principal.
- **RMSE** deve ser avaliado na escala real do ativo (desnormalizando as predições). Um RMSE de R$ 1,50 em uma ação de R$ 35,00 representa um erro de ~4%, que é defensável.
- **Desvio padrão dos resíduos**: esperamos que os erros se distribuam de forma razoavelmente simétrica em torno de zero. Viés sistemático (onde o modelo erra sempre na mesma direção) indica necessidade de retreinamento ou ajuste de features.

O MLflow registra as curvas de `val_rmse`, `val_mae` e `val_mape` por epoch justamente para que essa análise seja auditável e rastreável. Não basta apresentar o número final — a trajetória de aprendizagem do modelo já é evidência de qualidade de engenharia.

### Champion-Challenger como Próximo Passo

A LSTM é o nosso Champion. O challenger natural a avaliar em iteração futura é a **GRU (Gated Recurrent Unit)**, que tem arquitetura mais simples (menos parâmetros), tende a treinar mais rápido e em muitos benchmarks financeiros empata ou supera a LSTM em séries de médio prazo. Essa comparação está planejada como melhoria incremental no roadmap do projeto.

---

## Detecção de Drift e Estratégia de Retreino

Esse é um dos pontos mais críticos em qualquer sistema de predição financeira em produção. O maior risco não é o modelo errar — é ele errar silenciosamente por semanas sem que ninguém perceba.

### Por que a Degradação é Invisível

O modelo continua respondendo requisições normalmente, sem lançar exceções ou travar. Mas internamente, a distribuição dos dados de entrada mudou (por exemplo: uma crise setorial, mudança na política de juros, evento macroeconômico) e a relação que o modelo aprendeu entre os últimos 60 dias e o preço de amanhã já não vale mais. Isso é Concept Drift. Sem monitoramento ativo, o time só percebe o problema quando já acumulou semanas de decisões baseadas em predições degradadas.

### As Três Camadas de Alerta

**Data Drift — a entrada mudou:**
Monitoramos se a distribuição estatística da janela de preços de entrada divergiu em relação ao que o modelo viu durante o treino. A métrica usada é o PSI (Population Stability Index):
- PSI abaixo de 0.1: distribuição estável, sem ação.
- PSI entre 0.1 e 0.2: warning — investigar se houve evento de mercado relevante.
- PSI acima de 0.2: trigger automático de retreino.

**Performance Drift — as métricas em produção caíram:**
Diariamente, o sistema busca via `yfinance` o preço real de fechamento do ativo e compara com a predição feita no dia anterior. A partir disso calculamos um MAPE rolling de 7 dias. Se esse MAPE cruzar 12%, o pipeline de retreino é acionado independentemente do PSI.

**Concept Drift — a relação feature→target mudou estruturalmente:**
Detectado via testes estatísticos (ADWIN ou Page-Hinkley) aplicados na janela deslizante dos resíduos. Quando o padrão de erro muda de comportamento aleatório para sistemático (o modelo começa a errar sempre na mesma direção), isso indica que o mercado mudou de regime.

### Fluxo de Retreino com Champion-Challenger

Quando qualquer um dos alertas acima é ativado, o processo não é simplesmente "retreinar e substituir". Seguimos um protocolo de validação:

1. O modelo atual (Champion) permanece em produção respondendo requisições.
2. Um novo modelo (Challenger) é treinado com os dados mais recentes.
3. Ambos são avaliados no mesmo conjunto de holdout temporal.
4. O Challenger só substitui o Champion se apresentar melhoria de pelo menos 0.5% no MAPE — abaixo disso, a diferença pode ser ruído estatístico.
5. A promoção é registrada no MLflow Model Registry com as métricas comparativas e o `git_sha` do código utilizado.

Esse processo elimina o risco de promover um modelo pior por conta de overfitting no conjunto de validação. E mais importante: cria um rastro auditável de cada transição de versão em produção.

### O que já está implementado e o que é próximo passo

Atualmente o Prometheus monitora latência e volume de requisições na API de serving. A próxima camada — o cálculo de PSI e MAPE rolling em produção — está mapeada como entrega incremental usando Evidently, que já consta no `pyproject.toml` como dependência instalada.

---


## Como Executar o Projeto

O projeto tem dois modos de operação com propósitos distintos.

---

### Modo 1: Desenvolvimento Local (sem Docker)

**Para testes do dia a dia, validação de código e exploração dos dados, você não precisa de Docker.** Tudo roda via Poetry diretamente no terminal.

#### Passo 1 — Instalar e ativar o ambiente
```bash
export POETRY_CACHE_DIR=".poetry_cache"
poetry install
poetry shell
```

#### Passo 2 — Coletar e preparar os dados (Fase A)
```bash
# Baixar histórico do ativo
poetry run python src/features/data_collection.py --ticker PETR4.SA

# Versionar os dados brutos (não sobem para o Git)
poetry run dvc add data/raw/petr4_sa_raw.csv

# Gerar tensores com split 80/20 e validação de schema
poetry run python src/features/feature_engineering.py --ticker_id petr4_sa

# Rodar os testes de qualidade
poetry run pytest tests/ -v
```

#### Passo 3 — Explorar os dados (EDA)
```bash
poetry run jupyter notebook notebooks/01_eda.ipynb
```
O notebook contém análise de autocorrelação, estacionariedade, volatilidade e distribuição dos ativos. É exploratório — nenhum código aqui é trigger de produção.

#### Passo 4 — Treinar o modelo LSTM (Fase B)
```bash
# PETR4.SA
poetry run python src/models/train.py --ticker_id petr4_sa

# NVDC34.SA (NVIDIA BDR)
poetry run python src/models/train.py --ticker_id nvdc34_sa
```

#### Passo 5 — Inspecionar métricas no MLflow
```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Acesse `http://localhost:5000`. Você verá as curvas de RMSE, MAE e MAPE por epoch, as tags de governança e os artefatos do modelo.

#### Passo 6 — Detectar Drift (Fase C — Monitoramento)
```bash
# Roda as 3 camadas de detecção (PSI, MAPE rolling, viés de resíduos)
poetry run python src/monitoring/drift.py --ticker_id petr4_sa --ticker PETR4.SA
```
Os resultados aparecem no experimento `drift_monitoring` do MLflow. Exit code 1 indica alerta de retreino.

#### Passo 7 — Consultar o Agente Generativo (Fase D)
```bash
export OPENAI_API_KEY="sua_chave_aqui"
poetry run python src/agent/react_agent.py
```

---

### Modo 2: Produção com Docker (ecossistema completo)

**Para simular o ambiente de produção ou preparar a demonstração para a banca**, suba o ecossistema completo em containers isolados.

```bash
docker compose up --build
```

Isso inicializa 4 serviços simultaneamente, cada um em seu próprio container com compute isolado:

| Serviço | Porta | Propósito |
|---|---|---|
| `serving_api` | 8000 | API FastAPI de predições |
| `mlflow_server` | 5000 | Registro central de modelos |
| `prometheus` | 9090 | Métricas operacionais |
| `airflow` | 8080 | Orquestrador de pipelines (DAG) |

**Quando usar Docker vs. Poetry:**

- Use **Poetry** quando está desenvolvendo, ajustando código ou fazendo testes rápidos. É mais ágil e não exige Docker instalado ou rodando.
- Use **Docker** quando quer validar o sistema como um todo, quando vai apresentar para a banca ou quando precisa ver o Airflow orquestrando a DAG completa.

---

### Orquestração com Apache Airflow (Anti-SPOF — GAP 02)

Com o Docker rodando, acesse `http://localhost:8080` (usuário: `admin`, senha: `admin`).

O Airflow gerencia duas DAGs independentes — uma para cada ativo financeiro. Cada DAG executa o pipeline completo de forma declarativa:

```
coletar_dados → processar_features → validar_artefatos → treinar_modelo → detectar_drift
```

A separação por ativo garante que uma falha no pipeline da NVIDIA não interrompa o da Petrobras. Cada task roda em processo próprio, sem memória compartilhada — isso elimina o risco de Notebook como ponto único de falha (GAP 02).

