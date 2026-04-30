"""Pipeline RAG — Construção e Consulta Incremental do Vector Store (GAP 03).

Constrói um índice FAISS a partir de documentos de conformidade corporativa
e expõe uma função de consulta semântica para o Agente ReAct.

Princípios de design (GAP 03 — Anti-Full-Flush):
    - O índice é persistido em disco após a primeira construção.
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
import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── Caminhos ───────────────────────────────────────────────────────────────────
COMPLIANCE_DIR  = Path("data/compliance")
VECTORSTORE_DIR = Path("data/vectorstore/faiss_index")
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
# all-MiniLM-L6-v2: 22M params, bom para buscas semânticas em textos técnicos.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _embeddings() -> HuggingFaceEmbeddings:
    """Instancia o modelo de embeddings local via sentence-transformers.
    
    Sem necessidade de API key. Modelo baixado automaticamente do HuggingFace Hub
    na primeira execução e armazenado em cache local.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def construir_ou_atualizar_indice(forcar_rebuild: bool = False) -> FAISS:
    """Constrói o índice FAISS ou atualiza incrementalmente (upsert por hash).

    GAP 03: documentos não modificados não são re-embeddados.
    O índice existente nunca é deletado antes do novo estar pronto.

    Args:
        forcar_rebuild: Se True, reconstrói o índice do zero.

    Returns:
        Instância do FAISS vectorstore carregada.
    """
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

    if not novos_docs and VECTORSTORE_DIR.exists() and not forcar_rebuild:
        logger.info("Todos os documentos já estão indexados. Carregando índice existente.")
        return FAISS.load_local(
            str(VECTORSTORE_DIR),
            embedder,
            allow_dangerous_deserialization=True,
        )

    logger.info("%d documento(s) novo(s) ou modificado(s) para indexar.", len(novos_docs))

    # Chunking dos documentos novos
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n===", "\n---", "\n\n", "\n", " "],
    )

    chunks_texto = []
    chunks_metadata = []
    for doc in novos_docs:
        pedacos = splitter.split_text(doc["conteudo"])
        for i, pedaco in enumerate(pedacos):
            chunks_texto.append(pedaco)
            chunks_metadata.append({"source": doc["nome"], "chunk": i})

    logger.info("Total de chunks para embedding: %d", len(chunks_texto))

    if VECTORSTORE_DIR.exists() and not forcar_rebuild:
        # Upsert incremental: carrega índice existente e adiciona novos chunks
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            embedder,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_texts(chunks_texto, metadatas=chunks_metadata)
        logger.info("Índice atualizado incrementalmente (upsert).")
    else:
        # Primeira construção ou rebuild forçado
        from langchain.schema import Document
        docs_langchain = [
            Document(page_content=t, metadata=m)
            for t, m in zip(chunks_texto, chunks_metadata)
        ]
        vectorstore = FAISS.from_documents(docs_langchain, embedder)
        logger.info("Índice FAISS construído do zero.")

    # Persiste o índice (atômico: salva em tmp e renomeia — GAP 03)
    tmp_dir = Path("data/vectorstore/.tmp_faiss_index")
    vectorstore.save_local(str(tmp_dir))
    if VECTORSTORE_DIR.exists():
        import shutil
        shutil.rmtree(VECTORSTORE_DIR)
    tmp_dir.rename(VECTORSTORE_DIR)

    # Atualiza o registro de hashes
    hashes_atualizados = dict(hashes_salvos)
    for doc in novos_docs:
        hashes_atualizados[doc["nome"]] = doc["hash"]
    _salvar_hashes(hashes_atualizados)

    logger.info("Índice persistido em %s.", VECTORSTORE_DIR)
    return vectorstore


# ─── Singleton do vectorstore (evita re-carregar a cada chamada do agente) ──────
_vectorstore: FAISS | None = None


def _get_vectorstore() -> FAISS:
    """Retorna o vectorstore carregado, inicializando se necessário."""
    global _vectorstore
    if _vectorstore is None:
        embedder = _embeddings()
        if VECTORSTORE_DIR.exists():
            _vectorstore = FAISS.load_local(
                str(VECTORSTORE_DIR),
                embedder,
                allow_dangerous_deserialization=True,
            )
            logger.info("Vectorstore carregado do disco.")
        else:
            logger.info("Índice não encontrado. Construindo pela primeira vez...")
            _vectorstore = construir_ou_atualizar_indice()
    return _vectorstore


def consultar_rag(query: str, k: int = 3) -> str:
    """Consulta semântica no vectorstore de compliance.

    Args:
        query: Pergunta ou termo a buscar nos documentos corporativos.
        k: Número de chunks mais relevantes a retornar.

    Returns:
        String com os trechos mais relevantes encontrados.
    """
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
    parser = argparse.ArgumentParser(description="Pipeline RAG — Índice de Compliance")
    parser.add_argument("--rebuild", action="store_true",
                        help="Força reconstrução completa do índice")
    args = parser.parse_args()

    construir_ou_atualizar_indice(forcar_rebuild=args.rebuild)
    logger.info("Pipeline RAG finalizado.")
