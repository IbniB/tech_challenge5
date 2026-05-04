"""Pipeline RAG — Construção e Consulta Incremental do Vector Store (GAP 03).

Constrói um índice ChromaDB a partir de documentos de conformidade corporativa
e expõe uma função de consulta semântica para o Agente ReAct.

Princípios de design (GAP 03 — Anti-Full-Flush):
    - O índice é persistido em disco via Chroma SQLite nativo.
    - Atualizações usam upsert por hash de conteúdo: apenas documentos
      novos ou modificados são re-embeddados. O store nunca fica vazio.
    - Se o índice já existir, é carregado sem re-embedding.

Uso:
    # Construir ou atualizar o índice:
    poetry run python src/agent/rag_pipeline.py --rebuild

    # Consultar:
    from src.agent.rag_pipeline import consultar_rag
    resultado = consultar_rag("limite de recomendação sem aprovação humana")
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Caminhos ───────────────────────────────────────────────────────────────────
COMPLIANCE_DIR  = Path("data/compliance")
VECTORSTORE_DIR = Path("data/vectorstore/chroma_index")
HASH_REGISTRY   = Path("data/vectorstore/doc_hashes.json")

# ─── Configurações de Chunking ──────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


def _hash_conteudo(texto: str) -> str:
    """Gera hash SHA-256 do conteúdo para detecção de mudanças."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _carregar_hashes_salvos() -> dict:
    """Carrega o registro de hashes dos documentos já indexados."""
    if HASH_REGISTRY.exists():
        with open(HASH_REGISTRY, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _salvar_hashes(hashes: dict) -> None:
    """Persiste o registro de hashes atualizado."""
    HASH_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_REGISTRY, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)


def _carregar_documentos_compliance() -> list[dict]:
    """Carrega todos os documentos .txt da pasta de conformidade."""
    documentos = []
    if not COMPLIANCE_DIR.exists():
        logger.warning("Pasta de compliance não encontrada: %s", COMPLIANCE_DIR)
        return documentos

    for arquivo in sorted(COMPLIANCE_DIR.glob("*.txt")):
        conteudo = arquivo.read_text(encoding="utf-8")
        documentos.append({
            "nome": arquivo.name,
            "conteudo": conteudo,
            "hash": _hash_conteudo(conteudo),
        })
        logger.info("Documento carregado: %s (%d chars)", arquivo.name, len(conteudo))

    return documentos


# Modelo de embedding local — sem custo, sem API key.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _embeddings() -> HuggingFaceEmbeddings:
    """Instancia o modelo de embeddings local via sentence-transformers."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def construir_ou_atualizar_indice(forcar_rebuild: bool = False) -> Chroma:
    """Constrói o índice Chroma ou atualiza incrementalmente (upsert por hash)."""
    embedder = _embeddings()
    documentos = _carregar_documentos_compliance()

    if not documentos:
        raise ValueError("Nenhum documento de compliance encontrado em data/compliance/.")

    hashes_salvos = {} if forcar_rebuild else _carregar_hashes_salvos()

    # Identifica apenas os documentos que mudaram ou são novos
    novos_docs = [
        doc for doc in documentos
        if doc["nome"] not in hashes_salvos or hashes_salvos[doc["nome"]] != doc["hash"]
    ]

    if forcar_rebuild and VECTORSTORE_DIR.exists():
        logger.info("Apagando índice existente para rebuild...")
        shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)

    if not novos_docs and VECTORSTORE_DIR.exists() and not forcar_rebuild:
        logger.info("Todos os documentos já estão indexados. Carregando índice existente ChromaDB.")
        return Chroma(persist_directory=str(VECTORSTORE_DIR), embedding_function=embedder)

    logger.info("%d documento(s) novo(s) ou modificado(s) para indexar.", len(novos_docs))

    # Chunking dos documentos novos
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n===", "\n---", "\n\n", "\n", " "],
    )

    chunks_texto = []
    chunks_metadata = []
    chunk_ids = []
    for doc in novos_docs:
        pedacos = splitter.split_text(doc["conteudo"])
        for i, pedaco in enumerate(pedacos):
            chunks_texto.append(pedaco)
            chunks_metadata.append({"source": doc["nome"], "chunk": i})
            chunk_ids.append(f"{doc['nome']}_chunk_{i}")

    logger.info("Total de chunks para embedding: %d", len(chunks_texto))

    if chunks_texto:
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embedder
        )
        vectorstore.add_texts(chunks_texto, metadatas=chunks_metadata, ids=chunk_ids)
        logger.info("Índice atualizado no ChromaDB.")
    else:
        vectorstore = Chroma(persist_directory=str(VECTORSTORE_DIR), embedding_function=embedder)

    # Atualiza o registro de hashes
    hashes_atualizados = dict(hashes_salvos)
    for doc in novos_docs:
        hashes_atualizados[doc["nome"]] = doc["hash"]
    _salvar_hashes(hashes_atualizados)

    logger.info("Índice persistido em %s.", VECTORSTORE_DIR)
    return vectorstore


# ─── Singleton do vectorstore (evita re-carregar a cada chamada do agente) ──────
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    """Retorna o vectorstore carregado, inicializando se necessário."""
    global _vectorstore
    if _vectorstore is None:
        embedder = _embeddings()
        if VECTORSTORE_DIR.exists():
            _vectorstore = Chroma(persist_directory=str(VECTORSTORE_DIR), embedding_function=embedder)
            logger.info("Vectorstore Chroma carregado do disco.")
        else:
            logger.info("Índice Chroma não encontrado. Construindo pela primeira vez...")
            _vectorstore = construir_ou_atualizar_indice()
    return _vectorstore


def consultar_rag(query: str, k: int = 3) -> str:
    """Consulta semântica no vectorstore de compliance."""
    try:
        vs = _get_vectorstore()
        resultados = vs.similarity_search(query, k=k)

        if not resultados:
            return "Nenhuma diretriz encontrada para esta consulta na base corporativa."

        trechos = []
        for r in resultados:
            fonte = r.metadata.get("source", "desconhecido")
            trechos.append(f"[{fonte}] {r.page_content.strip()}")

        return "\n\n".join(trechos)

    except EnvironmentError as exc:
        return f"RAG indisponível: {exc}"
    except Exception as exc:
        logger.error("Erro na consulta RAG: %s", exc)
        return f"Erro ao consultar base de compliance: {exc}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RAG — Índice de Compliance (ChromaDB)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Força reconstrução completa do índice")
    args = parser.parse_args()

    construir_ou_atualizar_indice(forcar_rebuild=args.rebuild)
    logger.info("Pipeline RAG finalizado.")
