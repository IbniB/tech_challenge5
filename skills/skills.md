# Skills de Construção e Requisitos (Datathon MLOps)

## 1. Mapeamento do Problema de Negócio (Tech-Driven)
O problema real escolhido foi a **degradação silenciosa e ausência de observabilidade** de modelos de previsão financeira na bolsa de valores (Tesourarias). 
Nosso objetivo técnico é evitar que um modelo preditivo obsoleto de Deep Learning (LSTM) tome decisões ou direcione a equipe sem ter seus limites rastreáveis.

## 2. Gaps e Desafios Oficiais (Diretrizes a Serem Resolvidas)
As fases e decisões arquiteturais do repositório (`datathon-grupo-05`) devem obrigatoriamente matar os Gaps do Datathon:
- **Gap 01: Monitoramento Inexistente** -> Resolvido acoplando `Prometheus` e bibliotecas de Data Drift (Evidently) na ponta da API.
- **Gap 02: SPOF em Cadernos Jupyter** -> Resolvido extraindo lógicas brutas para scripts nativos (`.py`) dentro de `src/features/`.
- **Gap 04: Falta de Testes de Qualidade** -> Resolvido escrevendo garantias unitárias com `pytest` e Data Contracts estritos (`pandera`).
- **Gap 05: Código sem Rastreabilidade de Treino** -> Resolvido usando a plataforma `MLflow` bloqueando a existência de "Pesos sem Pai".
- **Gap 06: Degradação Silenciosa** -> Resolvido instruindo watchdogs locais na nossa Web API que alertam o MLflow em picos absurdos do ativo (CVM).
- **Gap 08: Repositórios Sem Dados (Gigantes)** -> Resolvido isolando toda sujeira raw de `.csv` via `DVC` versionado.

## 3. Justificativas e Tom de Voz
Cada ferramenta inserida (`Docker`, `Poetry`, `Faiss`, `Ragas`) não entra por "hype", entra para solucionar falhas na escalabilidade de projetos bancários. O README do projeto precisa ser claro e atestar isso para a bancada governamental.

## 4. Ambiente Operacional Nativo
- **Terminal Mandatório**: O desenvolvedor principal atua utilizando **Git Bash (MINGW64)** em ambiente Windows. 
- **Compatibilidade**: Toda instrução repassada de exportação de variáveis (ex: `export VAR=x`), paths ou execução de scripts deve ser aderente ao Bash (POSIX), evitando estritamente soluções exclusivas de PowerShell (`$env:`) ou CMD (`set`).
