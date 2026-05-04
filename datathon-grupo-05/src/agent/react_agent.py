"""Agente ReAct — Datathon Fase 05 (Fase D).

Agente conversacional com raciocínio ReAct (Reasoning + Acting) usando
modelo local Qwen2.5-0.5B-Instruct via HuggingFace Transformers.

Sem dependência de API key externa — roda 100% local, zero custo.

Ferramentas (≥ 3 exigidas pela banca):
    1. Previsao_LSTM_Interno   — API FastAPI interna (modelo LSTM da tesouraria)
    2. Cotacao_Real_Mercado    — Cotação real via yfinance (âncora anti-alucinação)
    3. Compliance_CVM_Risco   — RAG semântico sobre diretrizes CVM/Risco (FAISS)

Uso:
    poetry run python src/agent/react_agent.py
    poetry run python src/agent/react_agent.py "PETR4 vai subir amanhã?"
"""
import logging
import sys
from pathlib import Path

from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.agent.tools import (
    fetch_real_stock_price,
    predict_lstm_price,
    query_compliance_rag,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Configurações do Modelo Local ──────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_CACHE   = Path("data/.model_cache")     # cache local para não re-baixar
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.1                           # baixo para respostas determinísticas


def _carregar_llm_local() -> HuggingFacePipeline:
    """Carrega o Qwen2.5-0.5B-Instruct localmente via HuggingFace Transformers.

    O modelo é baixado automaticamente na primeira execução (~1 GB) e
    armazenado em MODEL_CACHE para reutilização.

    Returns:
        LangChain HuggingFacePipeline compatível com LangChain agents.
    """
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    logger.info("Carregando modelo local: %s", MODEL_ID)
    logger.info("Cache em: %s (download automático na primeira vez ~1 GB)", MODEL_CACHE)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=str(MODEL_CACHE),
        trust_remote_code=True,
    )

    # Tentativa de carregamento otimizado com Quantização INT4 (BitsAndBytes)
    try:
        from transformers import BitsAndBytesConfig
        import torch
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=str(MODEL_CACHE),
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config,
        )
        logger.info("Modelo carregado com Quantização INT4 (VRAM ~300MB).")
    except Exception as e:
        logger.warning(f"Quantização falhou ou GPU não detectada ({e}). Fazendo fallback para CPU bruta.")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=str(MODEL_CACHE),
            trust_remote_code=True,
            device_map="cpu",     # CPU — fallback compatível com qualquer máquina
            low_cpu_mem_usage=True,
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=None,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    logger.info("Modelo %s carregado com sucesso.", MODEL_ID)
    return HuggingFacePipeline(pipeline=pipe)


def criar_agente():
    """Instancia o agente ReAct com LLM local e 3 tools.

    Returns:
        Agente LangChain configurado.
    """
    llm = _carregar_llm_local()

    tools = [
        Tool(
            name="Previsao_LSTM_Interno",
            func=predict_lstm_price,
            description=(
                "Use para obter a previsão de preço de fechamento do próximo pregão "
                "gerada pelo modelo LSTM interno da tesouraria. "
                "Input: ticker do ativo (ex: PETR4.SA, NVDC34.SA). "
                "Retorna o preço previsto em R$ desnormalizado."
            ),
        ),
        Tool(
            name="Cotacao_Real_Mercado",
            func=fetch_real_stock_price,
            description=(
                "Use para buscar o preço de fechamento real e mais recente de um ativo "
                "diretamente na B3 via yfinance. "
                "Input: ticker do ativo (ex: PETR4.SA, NVDC34.SA). "
                "SEMPRE use esta tool antes de qualquer análise comparativa."
            ),
        ),
        Tool(
            name="Compliance_CVM_Risco",
            func=query_compliance_rag,
            description=(
                "Use OBRIGATORIAMENTE antes de emitir qualquer recomendação envolvendo "
                "compra, venda ou posicionamento em ativos. "
                "Consulta a base de conformidade CVM e políticas de risco da corretora. "
                "Input: descrição do contexto ou ação que deseja validar."
            ),
        ),
    ]

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=3,
        return_messages=True,
    )

    agente = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors="Por favor, responda estritamente no formato exigido: Thought, Action, Action Input, ou Final Answer.",
        memory=memory,
        max_iterations=3,
        early_stopping_method="generate",
    )

    logger.info("Agente ReAct inicializado com %d tools.", len(tools))
    return agente


def executar_consulta(agente, pergunta: str) -> str:
    """Executa uma consulta no agente.

    Args:
        agente: Instância do agente ReAct.
        pergunta: Pergunta em linguagem natural.

    Returns:
        Resposta do agente.
    """
    logger.info(">> Consulta: %s", pergunta)
    try:
        resposta = agente.run(pergunta)
        logger.info("<< Resposta: %s", resposta)
        return resposta
    except Exception as exc:
        logger.error("Erro na execução do agente: %s", exc)
        return f"Erro ao processar a consulta: {exc}"


def run_agent():
    """Ponto de entrada — executa o agente com perguntas de demonstração."""
    agente = criar_agente()

    perguntas_demo = [
        "Quanto a PETR4.SA fechou hoje?",
        "O que o modelo LSTM prevê para PETR4.SA amanhã?",
        "Dado esse cenário, o que o compliance do banco recomenda?",
    ]

    for i, pergunta in enumerate(perguntas_demo, 1):
        print(f"\n{'='*60}")
        print(f"CONSULTA {i}: {pergunta}")
        print("="*60)
        resposta = executar_consulta(agente, pergunta)
        print(f"\nRESPOSTA:\n{resposta}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pergunta_custom = " ".join(sys.argv[1:])
        agente = criar_agente()
        print(executar_consulta(agente, pergunta_custom))
    else:
        run_agent()
