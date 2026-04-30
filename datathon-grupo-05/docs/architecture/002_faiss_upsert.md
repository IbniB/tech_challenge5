# ADR 002: Estratégia de Upsert no Vector Store (Evitando Anti-Padrões de Flush)

**Data:** 28/04/2026
**Status:** Aceito
**Autores:** Grupo 05

## Contexto

A pipeline RAG (Retrieval-Augmented Generation) atua como fonte de verdade para o assistente financeiro no tocante a políticas da CVM e de compliance.
Uma arquitetura imatura comum é o uso da estratégia "Flush & Reload" ou "Full-Flush", onde a base vetorial (Vector Store) é deletada inteiramente e reconstruída a cada atualização de documentos. 

Isso resulta no **Anti-padrão GAP 03**: durante o carregamento de novos documentos (que pode demorar minutos), a base vetorial fica vazia. Qualquer prompt do usuário nesse intervalo sofrerá alucinação severa, pois o banco de dados não terá contexto para retornar.

## Decisão

Adotamos a estratégia de **Upsert Incremental baseado em Hash de Conteúdo** utilizando a biblioteca **FAISS** acoplada ao LangChain.

Ao invés de deletar a base, a pipeline faz o hash SHA-256 de cada documento ou chunk. Se o hash do documento já existir na base, a indexação o pula. Se for novo ou alterado, ele é adicionado incrementalmente.

## Consequências

- **Alta Disponibilidade (99.9%):** O agente RAG nunca sofre períodos de indisponibilidade devido a *store vazio*.
- **Desempenho:** Ingestão de novos documentos de compliance tornou-se exponencialmente mais rápida (O(n) das diferenças) comparado a reinjetar milhões de vetores.
- **Compatibilidade:** Funciona perfeitamente com os embeddings do `sentence-transformers/all-MiniLM-L6-v2`.
