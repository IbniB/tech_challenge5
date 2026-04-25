# Tech Challenge (Fases 04 e 05) - MLOps Analytics para Tesouraria e Mercado Financeiro

> Este repositório concentra a submissão final do Tech Challenge englobando as entregas de Deep Learning para Séries Temporais (Fase 04) e as exigências maduras de Engenharia e Arquitetura MLOps de um Datathon (Fase 05).

---

## 🎯 Por Que Esse Projeto Existe? (Problema de Negócio Real)

Em mesas de operações e tesourarias financeiras, modelos de predição de preços de ações são vastamente utilizados para detecção de tendências de mercado, otimização de portfólios, e balanceamento de risco corporativo. No entanto, o **maior gargalo de negócio hoje não é a acurácia do modelo**, mas sim a **degradação silenciosa em produção**.

Tentar basear decisões em um modelo de Deep Learning obsoleto, rodando sobre Notebooks que quebram sem alertas, ou recebendo features inválidas, leva instituições reguladas à quebra de confiança e pesados prejuízos, penalizados sob a rubrica da Governança MLOps apontada na Fase 05.

### ✔ Nossa Proposta de Solução:
Ao invés de criarmos apenas a Rede Neural, estamos empacotando-a em uma plataforma ponta a ponta resistente a falhas, que conta com:
- Predição de séries temporais financeiras de alta eficácia utilizando redes **LSTM** com arquitetura corporativa.
- Integração natural Language (LLM) que consome nossos próprios modelos via um **Agente ReAct**, oferecendo contexto e respostas financeiras.
- Visibilidade contra descalibramento através de *Drift Detection*.

---

## 🏗 Explicativo de Escolhas e Motivações Arquiteturais

Para atender perfeitamente às rigidades das instituições bancárias e não cometer os gaps críticos identificados na indústria, adotamos as seguintes trilhas arquiteturais. Todas essas ações devem ser validadas perante código de banca:

### 1. DVC (Data Version Control) ao Invés de Storage Manual do Git
- **Ação**: Implementamos uma `data/` branch mantida estritamente via `dvc.yaml` versionado no projeto. 
- **Justificativa de Negócio**: Não podemos armazenar bases e dumps de *yfinance* massivos do mercado no repositório de códigos. O DVC abstrai essas massas, impedindo poluição do repositório, garantindo reprodutibilidade em 100% para novos desenvolvedores, e solucionando o **Gap 08** de "Ambientes de Desenvolvimento Sem Dados".

### 2. Isolar Notebooks e Adotar CI/CD Automatizado  
- **Ação**: Todo o preprocessamento dos dados financeiros e loop de treino que originalmente reside em EDA (`notebooks/01_eda.ipynb`) é transposto integralmente para `src/features/` e testado via *PyTest*. Protegemos tudo contra _pushs_ destrutivos usando *GitHub Actions*.
- **Justificativa de Negócio**: Eliminar o famigerado **Gap 02 (SPOF em Notebooks)**. Arquiteturas que confiam em cadernos interativos são caóticas e não rastreáveis. Usamos CLI e `Makefiles` para rodar os mesmos steps declarativamente. Adicionalmente, resolve o **Gap 04 (Código sem testes)** provendo *Quality Gates* confiáveis.

### 3. Evitando a "Caixa Preta": MLflow, Prometheus e Evidently
- **Ação**: Adotar MLFlow rigoroso com métricas financeiras (RMSE, MAE e MAPE) e rastreamento constante. Toda a inferência em produção passará por loggers via Prometheus que analisam a degradação temporal *Drift*.
- **Justificativa de Negócio**: Elimina os **Gaps 01 e 06 (Monitoramento Zero e Concept Drift)**. Nosso modelo rastreará a variável (PSI > 0.1 como warning) das variações históricas da ação, impedindo a degradação silenciosa na previsão do ativo pela rede LSTM.

### 4. Inteligência Generativa Acoplada (RAG e ReAct Agent)
- **Ação**: Utilizar Langchain e FastApi para integrar agentes que usem o conhecimento corporativo (Tools) acoplados ao projeto.
- **Justificativa de Negócio**: Atende às diretrizes da Fase 05 fornecendo uma interface de Agente Inteligente capaz de justificar tendências frente as políticas macroeconômicas. Adiciona segurança em cima do dado (Guardrails estritos) lidando com RAGs corporativo limitados a informações provadas sem alucinações.

---

## Estrutura Inicial 

O projeto está estruturado nos moldes preconizados seguindo as melhores práticas do Datathon de Governança. Para rodar qualquer esteira, consulte a pasta correspondente e interaja através dos comandos contidos no `Makefile`.

---

## 🚀 Como Iniciar Seu Ambiente MLOps e Testar o Sistema

Nós migramos da abordagem `pip install global` indiscriminada para uma gestão profissional via ambientes virtuais. A plataforma orquestradora escolhida para a Fase A é o **Poetry**. 

### 1. Formando a Bolha de Segurança
Na raiz do diretório `datathon-grupo-05/`, abra seu terminal local e rode:
```bash
poetry install
```
*Isto irá isolar todas as dependências lendo formalmente o `pyproject.toml` sem interferir da sua máquina local.*

### 2. Extraindo o Ativo Preditivo (Petrobras - PETR4)
Busque da nuvem com nossa malha validada:
```bash
poetry run python src/features/data_collection.py --ticker PETR4.SA
```
Em seguida, lacre no cofre para evitar versionamento de base de dados gigante no git corporativo *(Governança Gap 08)*:
```bash
poetry run dvc add data/raw/petr4_sa_raw.csv
```

### 3. Engine de Transformação Numérica (Janelas LSTM)
Para converter o simples histórico tabular para a geometria tenso-tridimensional que a sua Rede Neural precisará para ler as memórias de transação:
```bash
poetry run python src/features/feature_engineering.py --ticker_id petr4_sa
```

### 4. Rodando os Testes do Motor (Quality Gate)
Como provamos de fato que o framework impede falhas silenciosas na hora da leitura do CSV baixado da rede? Rodando o `pytest` blindado pelas regras do MLOps *Pandera*:
```bash
poetry run pytest tests/ -v
```
