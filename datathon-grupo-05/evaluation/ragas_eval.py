"""Script de Avaliação RAGAS — Datathon Fase 05.

Avalia o pipeline RAG usando o Golden Set contra 4 métricas principais (Exigência da Banca):
- Context Precision: O RAG trouxe o contexto certo no topo?
- Context Recall: O RAG trouxe toda a informação necessária para a resposta?
- Faithfulness: A resposta é fiel ao contexto ou o modelo alucinou?
- Answer Relevance: A resposta responde diretamente à pergunta do usuário?

Fluxo:
1. Carrega as perguntas e respostas esperadas do Golden Set.
2. Usa o RAG local (FAISS + all-MiniLM) para buscar contextos.
3. Usa o LLM local (Qwen2.5-0.5B) para gerar a resposta.
4. Usa a OpenAI (LLM-as-a-judge) via RAGAS para avaliar rigorosamente a qualidade.

Uso:
    export OPENAI_API_KEY="sua_chave"
    poetry run python evaluation/ragas_eval.py
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Garante que módulos de src possam ser importados
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.rag_pipeline import _get_vectorstore
from src.agent.react_agent import _carregar_llm_local

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path("data/golden_set/golden_set.json")
RESULTS_DIR = Path("evaluation/results")


def carregar_golden_set(limite: int | None = None) -> list[dict]:
    """Carrega os pares de Q&A do JSON de avaliação."""
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(f"Golden Set não encontrado em {GOLDEN_SET_PATH}")
    
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pares = data.get("pairs", [])
    if limite:
        pares = pares[:limite]
        
    logger.info("Golden Set carregado: %d pares selecionados.", len(pares))
    return pares


def gerar_respostas_e_contextos(pares: list[dict]) -> dict:
    """Gera respostas usando RAG + Qwen2.5 local para serem julgadas."""
    logger.info("Iniciando geração de respostas com o LLM Local (Qwen2.5)...")
    
    llm_local = _carregar_llm_local()
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = PromptTemplate.from_template(
        "Você é um assistente de compliance financeiro corporativo.\n"
        "Use APENAS o contexto fornecido para responder de forma clara e direta.\n\n"
        "Contexto:\n{contexto}\n\n"
        "Pergunta: {pergunta}\n\n"
        "Resposta:"
    )
    
    # Formato exigido pelo Dataset do HuggingFace/RAGAS
    dados_avaliacao = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []  # Algumas versões do ragas usam ground_truths (list), manteremos string por padrão
    }
    
    for i, par in enumerate(pares, 1):
        pergunta = par["query"]
        expected = par["expected_answer"]
        
        # 1. Recuperação (Retrieval)
        docs = retriever.invoke(pergunta)
        textos_contexto = [doc.page_content for doc in docs]
        
        # 2. Geração (Generation)
        contexto_str = "\n".join(textos_contexto)
        prompt_formatado = prompt_template.format(contexto=contexto_str, pergunta=pergunta)
        
        try:
            resposta = llm_local.invoke(prompt_formatado)
        except Exception as e:
            logger.error("Falha ao gerar resposta para pergunta %d: %s", i, e)
            resposta = "Erro de geração."
            
        dados_avaliacao["question"].append(pergunta)
        dados_avaliacao["answer"].append(resposta.strip())
        dados_avaliacao["contexts"].append(textos_contexto)
        # Ragas 0.1.x geralmente aceita ground_truth como string simples no dict
        dados_avaliacao["ground_truth"].append(expected)
        
        logger.info("Processado [%d/%d]: %s...", i, len(pares), pergunta[:40])
        
    return dados_avaliacao


def rodar_avaliacao_ragas(dados_dict: dict) -> pd.DataFrame:
    """Invoca o RAGAS usando GPT-3.5 como juiz imparcial das respostas locais."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "========================================================================\n"
            "OPENAI_API_KEY ausente! O RAGAS usa prompts complexos que modelos de \n"
            "0.5B não conseguem processar adequadamente.\n\n"
            "Como você está fazendo tudo free, NÃO se preocupe com o RAGAS agora.\n"
            "Use o script 'evaluation/llm_judge.py' — ele foi construído para rodar\n"
            "100% local e free, julgando as respostas nas 3 métricas da banca.\n"
            "========================================================================"
        )
        return pd.DataFrame()
        
    logger.info("Convertendo para Dataset do HuggingFace...")
    # Ajuste de compatibilidade para diferentes versões do RAGAS:
    # Se ragas pedir ground_truths em formato de lista de lista, adaptamos.
    try:
        dataset = Dataset.from_dict(dados_dict)
    except Exception:
        dados_dict["ground_truths"] = [[gt] for gt in dados_dict.pop("ground_truth")]
        dataset = Dataset.from_dict(dados_dict)
        
    logger.info("Iniciando juízes do RAGAS (GPT-3.5-turbo)...")
    judge_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    judge_embeddings = OpenAIEmbeddings()
    
    metricas = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]
    
    resultado = evaluate(
        dataset,
        metrics=metricas,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    
    return resultado.to_pandas()


def main():
    parser = argparse.ArgumentParser(description="Avaliação RAGAS do Agente")
    parser.add_argument("--limite", type=int, default=None, 
                        help="Número máximo de perguntas (útil para testes rápidos)")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Carrega QA
    pares = carregar_golden_set(args.limite)
    
    # 2. Gera dados
    dados = gerar_respostas_e_contextos(pares)
    
    # 3. Avalia
    df_resultado = rodar_avaliacao_ragas(dados)
    
    if df_resultado.empty:
        return
        
    # 4. Salva e Sumariza
    arquivo_saida = RESULTS_DIR / "ragas_metrics_report.csv"
    df_resultado.to_csv(arquivo_saida, index=False)
    
    print("\n" + "="*50)
    print("🏆 RESULTADOS RAGAS (LLM-as-a-Judge) 🏆")
    print("="*50)
    
    metricas_chave = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    for metrica in metricas_chave:
        if metrica in df_resultado.columns:
            media = df_resultado[metrica].mean()
            print(f"{metrica.replace('_', ' ').title():<20}: {media:.4f}")
            
    print("="*50)
    print(f"Relatório detalhado por pergunta salvo em: {arquivo_saida}")
    print("="*50)


if __name__ == "__main__":
    main()
