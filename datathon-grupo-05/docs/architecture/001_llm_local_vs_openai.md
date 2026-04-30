# ADR 001: Adoção de LLM Local vs APIs Externas (OpenAI)

**Data:** 28/04/2026
**Status:** Aceito
**Autores:** Grupo 05

## Contexto

O desenvolvimento de um Agente de Inteligência Artificial para operar no contexto da tesouraria e mercado financeiro corporativo exige fortes controles de governança de dados. Na versão original do projeto, planejava-se usar a API da OpenAI (GPT-3.5/4) combinada com LangChain. No entanto, enviar contexto estratégico (como limites de tolerância de risco e teses internas de compliance) para servidores de terceiros fere diretrizes primárias de segurança da informação do banco. Além disso, a cobrança por tokens cria um gargalo financeiro de escala.

## Decisão

Optou-se por construir o Agente ReAct utilizando exclusivamente Modelos de Linguagem Locais (LLMs Open-Weight) através da biblioteca `transformers` e `langchain-huggingface`.

O modelo selecionado foi o **Qwen2.5-0.5B-Instruct**.

## Consequências

**Positivas:**
- **Zero Custo Operacional (OPEX):** Não há cobrança por token na geração de inferências, RAG ou avaliações.
- **Privacidade e Segurança:** Todo o processamento (e vetores FAISS) ocorre localmente ou na VPC da organização, impedindo o vazamento de segredos de compliance (CVM/Risco).
- **Maturidade Técnica (GAP 03 Resolvido):** Demonstra alta proficiência arquitetural ao encapsular o modelo na camada de serving da própria equipe.

**Negativas / Mitigações:**
- **Capacidade Cognitiva Menor:** Modelos de 0.5B de parâmetros podem falhar em parseamentos JSON estritos. *Mitigação:* Implementou-se fallback via Regex no script `llm_judge.py` e refinamentos fortes de system prompt no ReAct Agent.
- **Sobrecarga de Infraestrutura (CAPEX):** Exige alocação de GPUs/CPUs na infraestrutura interna.
