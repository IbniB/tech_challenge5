# OWASP Mapping: Mitigação das Top 10 Ameaças para LLM

Mapeamento das 7 principais ameaças da OWASP aplicáveis a este projeto.

| Ameaça OWASP | Risco no Projeto | Estratégia de Mitigação Implementada |
| :--- | :--- | :--- |
| **LLM01: Prompt Injections** | Usuário instruir o agente a revelar system prompts ou burlar as regras da CVM. | System prompt reforçado via LangChain. A arquitetura de ferramentas (tools) é restrita apenas a leitura de RAG e chamadas de Previsão de Preço (Read-Only). |
| **LLM02: Insecure Output Handling** | Agente retornar payloads XSS em respostas. | Retornos do LLM são tratados como texto puro. A interface (se houver no futuro) fará sanitização de Markdown. |
| **LLM03: Training Data Poisoning** | Fontes do Yahoo Finance serem manipuladas. | Utilização de normalização (MinMax Scaler) e detecção de Data Drift (PSI/Evidently) para bloquear inferência em caso de desvios anormais na entrada. |
| **LLM04: Model Denial of Service** | LLM sobrecarregar a CPU/VRAM do servidor com queries custosas. | LLM quantizado em 4-bit (bitsandbytes) para economizar VRAM. FastAPI suporta backpressure; limitação do tamanho do prompt via chunking estrito no RAG. |
| **LLM05: Supply Chain Vulnerabilities** | Pacotes PyPI maliciosos comprometerem o ambiente. | `pyproject.toml` fixado com hashes (Poetry lockfile), Dockerização sem root, dependência evitada em containers desconhecidos. |
| **LLM06: Sensitive Information Disclosure** | Agente revelar as teses de tesouraria do banco para pessoas não autorizadas. | O modelo RAG é preenchido com documentos de compliance, mas não com posições reais de clientes. RAG local isolado. |
| **LLM09: Overreliance** | Operadores confiarem cegamente nas predições LSTM/Agente. | Adicionado *Disclaimer* obrigatório na formatação das predições, métrica severa de σ-tolerance (> 70%) para validar viabilidade. |
