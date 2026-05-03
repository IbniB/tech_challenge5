# System Card: Agente ReAct Financeiro com MLOps

## 1. Visão Geral
O sistema provê uma interface de IA interativa e segura (Agente ReAct) suportado por um modelo preditivo profundo (LSTM) e uma base vetorial de compliance (RAG), desenhado exclusivamente para a Tesouraria.

## 2. Componentes Principais
- **API (FastAPI):** Expõe inferência em tempo real, monitoramento e acionamento do agente.
- **Agente (LLM Local):** Utiliza o Qwen2.5-0.5B-Instruct quantizado em INT4. Pensa autonomamente (Reasoning) e decide quais *tools* usar.
- **Modelo LSTM:** Rede neural Pytorch hospedada no endpoint `/infer`.
- **RAG (ChromaDB):** Banco vetorial que embute diretrizes da CVM e da Tesouraria usando `sentence-transformers/all-MiniLM-L6-v2`.

## 3. Segurança e Auditoria
- O Agente roda de forma **100% Offline/Local**, protegendo dados da tesouraria contra vazamento.
- **MLflow** orquestra o versionamento do modelo e métricas.
- Monitoramento de tráfego, drift e latência gerido via Prometheus + Grafana.

## 4. Limites de Responsabilidade
O sistema é classificado como "Auxiliar à Tomada de Decisão". Ordens de compra e venda não são executadas automaticamente pela plataforma. Todo conselho fornecido pelo Agente possui o *disclaimer* exigido pela CVM alertando o usuário final.
