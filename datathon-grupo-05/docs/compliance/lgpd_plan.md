# LGPD Plan: Conformidade em Plataforma Preditiva

Este documento detalha o tratamento e mitigação de dados sensíveis e pessoais (PII).

## 1. Mapeamento de PII
- O **Dataset de Treinamento** (yfinance) contém exclusivamente dados públicos agregados de bolsa de valores. Nenhum dado PII ou SPII existe na modelagem.
- O **Agente LLM** interage com usuários via prompt. Existe risco de inserção de PII nos logs.

## 2. Minimização de Dados
- **Zero Retenção Pessoal:** A API FastAPI foi projetada para não exigir ou gravar atributos identificadores de usuários. O rastreamento de sessão deve utilizar identificadores opacos (ex: UUID4).

## 3. Prevenção de Vazamentos (LLM)
- Por ser um LLM **local (Qwen2.5)**, não existe trânsito transfronteiriço de dados ou exposição para a nuvem de terceiros (ex: OpenAI, Anthropic). O dado fornecido pelo usuário nunca sai da infraestrutura.

## 4. Direito ao Esquecimento
Conforme previsto na LGPD (Art. 18), usuários da tesouraria podem solicitar remoção do histórico do Agente. Foi estabelecido nos guidelines internos um SLA de 15 dias úteis para purga de logs da base vetorial/sqlite.
