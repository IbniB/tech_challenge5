"""Script de Avaliação LLM-as-a-Judge — Datathon Fase 05.

Avalia as respostas do agente segundo 3 critérios (Exigência da Banca):
1. Precisão técnica (Informações financeiras corretas?)
2. Relevância (Aborda diretamente a pergunta?)
3. Adequação ao negócio (É útil para decisões e respeita compliance?)

Usa o modelo local Qwen2.5-0.5B-Instruct para julgar, 100% free e sem API Key.
"""
import json
import logging
import re
from pathlib import Path

# Adiciona src ao path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.react_agent import _carregar_llm_local
from evaluation.ragas_eval import carregar_golden_set, gerar_respostas_e_contextos

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("evaluation/results")

JUDGE_PROMPT_TEMPLATE = """Você é um juiz avaliador especializado em compliance financeiro corporativo.
Avalie a resposta gerada para a pergunta abaixo em 3 critérios, dando uma nota de 1 a 5 para cada.

1. Precisão técnica: A resposta está correta segundo o contexto esperado?
2. Relevância: A resposta aborda diretamente a pergunta, sem tangenciar?
3. Adequação ao negócio: A resposta respeita os limites e diretrizes de risco e compliance?

Pergunta: {question}
Contexto Esperado (Gabarito): {ground_truth}

Resposta Gerada a ser avaliada: {answer}

Você DEVE responder EXATAMENTE neste formato JSON e nada mais:
{{"precisao_tecnica": 5, "relevancia": 5, "adequacao_negocio": 5, "justificativa": "Sua análise curta aqui"}}

Sua Avaliação JSON:"""


def evaluate_with_local_judge(pares: list[dict], respostas_geradas: dict) -> list[dict]:
    """Avalia respostas usando Qwen2.5 local como juiz."""
    llm_local = _carregar_llm_local()
    evaluations = []
    
    logger.info("Iniciando LLM-as-a-judge local (Qwen2.5-0.5B)...")
    
    for i, (par, ans) in enumerate(zip(pares, respostas_geradas["answer"]), 1):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=par["query"],
            ground_truth=par["expected_answer"],
            answer=ans,
        )
        
        try:
            # Invoca o LLM local
            raw_response = llm_local.invoke(prompt)
            # Tenta extrair apenas o JSON da resposta (modelos pequenos às vezes colocam texto a mais)
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = json.loads(raw_response)
                
            result = {
                "precisao_tecnica": int(result.get("precisao_tecnica", 0)),
                "relevancia": int(result.get("relevancia", 0)),
                "adequacao_negocio": int(result.get("adequacao_negocio", 0)),
                "justificativa": str(result.get("justificativa", "Parse OK"))
            }
                
        except Exception as e:
            logger.warning("Falha ao parsear JSON na pergunta %d. Usando Regex Fallback. Erro original: %s", i, e)
            # Regex Fallback: extrai os números diretamente do texto truncado
            try:
                precisao = int(re.search(r'"precisao_tecnica"\s*:\s*(\d)', raw_response).group(1))
            except: precisao = 0
            
            try:
                relevancia = int(re.search(r'"relevancia"\s*:\s*(\d)', raw_response).group(1))
            except: relevancia = 0
                
            try:
                adequacao = int(re.search(r'"adequacao_negocio"\s*:\s*(\d)', raw_response).group(1))
            except: adequacao = 0
                
            result = {
                "precisao_tecnica": precisao,
                "relevancia": relevancia,
                "adequacao_negocio": adequacao,
                "justificativa": "Recuperado via Regex."
            }
            
        evaluations.append(result)
        logger.info("Julgamento %d concluído: %s", i, result)
        
    return evaluations


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Vamos usar apenas 5 para teste rápido, o modelo de 0.5B demora um pouco
    pares = carregar_golden_set(limite=5)
    dados = gerar_respostas_e_contextos(pares)
    
    avaliações = evaluate_with_local_judge(pares, dados)
    
    # Salva e sumariza
    with open(RESULTS_DIR / "llm_judge_results.json", "w", encoding="utf-8") as f:
        json.dump(avaliações, f, indent=2, ensure_ascii=False)
        
    avg_precisao = sum(e["precisao_tecnica"] for e in avaliações) / len(avaliações)
    avg_relevancia = sum(e["relevancia"] for e in avaliações) / len(avaliações)
    avg_adequacao = sum(e["adequacao_negocio"] for e in avaliações) / len(avaliações)
    
    print("\n" + "="*50)
    print("⚖️ RESULTADOS LLM-AS-A-JUDGE (100% LOCAL) ⚖️")
    print("="*50)
    print(f"Critérios (Escala 1 a 5):")
    print(f"Precisão Técnica   : {avg_precisao:.1f}/5.0")
    print(f"Relevância         : {avg_relevancia:.1f}/5.0")
    print(f"Adequação Negócio  : {avg_adequacao:.1f}/5.0")
    print("="*50)
    print(f"Salvo em: evaluation/results/llm_judge_results.json")

if __name__ == "__main__":
    main()
