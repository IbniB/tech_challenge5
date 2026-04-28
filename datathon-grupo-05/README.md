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

## Estrutura Inicial e Como Executar Deste Ambiente

O projeto utiliza o Poetry como core de ambiente.

### Primeira Etapa: Instalação e Ativação do Ambiente
Na raiz deste diretório (`datathon-grupo-05/`), isolamos os pacotes criando a bolha nativa:
```bash
poetry install
```

Após a instalação terminar, **ative o ambiente virtual** na sua máquina para que o terminal passe a reconhecer as bibliotecas isoladas (o prefixo do seu terminal vai mudar indicando que você está seguro):
```bash
poetry shell
```

### Segunda Etapa: Pre-processamento e Baseline (Fase A)
Nós automatizamos as varreduras financeiras e os janelamentos tensores de Deep Learning. Realize o donwload inicial e a higienização unitária usando:
```bash
poetry run python src/features/data_collection.py --ticker PETR4.SA
poetry run dvc add data/raw/petr4_sa_raw.csv

# Cria os recortes matemáticos sequenciais validando integridade no background
poetry run python src/features/feature_engineering.py --ticker_id petr4_sa

# Provando os testes técnicos
poetry run pytest tests/ -v
```

### Terceira Etapa: Base Neutra Temporal (Fase B)
Com todos os recortes `.npy` criados pelo código acima, basta empurrar os treinos da nossa classe PyTorch para rodarem em background salvando estatísticas automáticas. Você deve passar qual ação deseja que ele puxe da pasta processed usando o parâmetro `--ticker_id`:
```bash
poetry run python src/models/train.py --ticker_id petr4_sa
```

Para provar a eficiência da governança de modelos, **teste o MLOps na prática**. Abra o dashboard interativo para inspecionar gráficos de erro, rastreio temporal e versões de modelos gravados rodando no bash:
```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
*(Após rodar, basta acessar `http://localhost:5000` ou `http://127.0.0.1:5000` em qualquer navegador web).*

### Quarta Etapa: Inicializando Servidores e Microserviços (Fase C)
O projeto agora é operado em múltiplos microsserviços blindados. A API do FastAPI roda em volta do Prometheus com o painel do MLflow operando assincronamente.
```bash
docker compose up --build
```

### Última Etapa: Consultando Agentes Humanos ou Autônomos (Fase D)
Para conversar de forma nativa com a infraestrutura ReAct embutida, você pode girar os scripts agentes desenvolvidos acoplados sobre a OpenAI:
```bash
poetry run python src/agent/react_agent.py
```
Atenção: Você vai precisar definir a variável do bash chamada `OPENAI_API_KEY` para as inferências Generativas. Os testes de avaliação estatística não dependem deste comando para aprovação local de dependências LLM-as-a-judge.
