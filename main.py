import io
import os

import pandas as pd
from minio import Minio
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_core.documents import Document

# ==========================================
# CONFIGURAÇÕES
# ==========================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = "csgodatalake"

# Modelos disponíveis no Ollama (aparecerão no dropdown do Gradio)
SUPPORTED_MODELS = [
    "llama3.2",
    "llama3",
    "mistral",
    "gemma2",
    "phi3",
]

# Modelo de embeddings (especializado em busca semântica)
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "csgo_rag"

# ==========================================
# CLIENTE MINIO
# ==========================================

def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


# ==========================================
# CARREGAMENTO DOS DADOS DA CAMADA GOLD
# ==========================================

def load_gold_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê as tabelas Gold do MinIO e retorna dois DataFrames."""
    client = get_minio_client()

    def read_parquet(key: str) -> pd.DataFrame:
        response = client.get_object(BUCKET, key)
        return pd.read_parquet(io.BytesIO(response.read()))

    df_combat = read_parquet("gold/combat_context_stats.parquet")
    df_bomb = read_parquet("gold/bomb_context_stats.parquet")
    return df_combat, df_bomb


# ==========================================
# CONVERSÃO DOS DADOS EM DOCUMENTOS DE TEXTO
# ==========================================

def dataframes_to_documents(
    df_combat: pd.DataFrame,
    df_bomb: pd.DataFrame,
) -> list[Document]:
    """
    Transforma cada linha das tabelas Gold em um texto legível para o LLM,
    seguindo o padrão LangChain Document.
    """
    docs: list[Document] = []

    # --- Tabela de Combate ---
    for _, row in df_combat.iterrows():
        text = (
            f"Arma: {row['wp']} | Rank do atacante: {row['att_rank']} | "
            f"Hitbox: {row['hitbox']} | Lado: {row['att_side']} | "
            f"Dano médio HP: {row['avg_hp_damage']:.2f} | "
            f"Dano total HP: {row['total_hp_damage']} | "
            f"Dano total armadura: {row['total_armor_damage']} | "
            f"Total de acertos: {row['total_hits']}"
        )
        docs.append(Document(page_content=text, metadata={"source": "combat"}))

    # --- Tabela de Bomba Plantada ---
    for _, row in df_bomb.iterrows():
        text = (
            f"Site da bomba: {row['bomb_site']} | Lado: {row['att_side']} | "
            f"Arma: {row['wp']} | "
            f"Dano médio pós-plant: {row['avg_damage_post_plant']:.2f} | "
            f"Dano total pós-plant: {row['damage_post_plant']} | "
            f"Acertos pós-plant: {row['hits_post_plant']}"
        )
        docs.append(Document(page_content=text, metadata={"source": "bomb"}))

    return docs


# ==========================================
# SPRINT 5 — PIPELINE DE EMBEDDINGS
# ==========================================

def build_vector_store(force_rebuild: bool = False) -> Milvus:
    """
    Requisito 1 — Integração com Ollama:
        Conecta ao Ollama local usando o modelo nomic-embed-text para gerar embeddings.

    Requisito 2 — Geração de Embeddings:
        Converte cada chunk de texto num vetor numérico de alta dimensão.

    Requisito 3 — Indexação Vetorial:
        Salva e indexa os vetores no Milvus para busca semântica eficiente.

    Retorna a instância do Milvus pronta para similarity_search.
    """

    # --- Requisito 1: Integração com Ollama ---
    print("[Sprint 5] Conectando ao Ollama para embeddings...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # Verifica se a coleção já existe no Milvus para evitar re-indexação desnecessária
    if not force_rebuild:
        try:
            vector_store = Milvus(
                embedding_function=embeddings,
                connection_args={"uri": MILVUS_URI},
                collection_name=COLLECTION_NAME,
            )
            print("[Sprint 5] Coleção existente carregada do Milvus.")
            return vector_store
        except Exception:
            print("[Sprint 5] Coleção não encontrada. Criando do zero...")

    # Carrega dados do MinIO
    print("[Sprint 5] Carregando dados da camada Gold (MinIO)...")
    df_combat, df_bomb = load_gold_data()
    print(f"  -> {len(df_combat)} registros de combate | {len(df_bomb)} registros de bomba")

    # Converte linhas em documentos de texto
    documents = dataframes_to_documents(df_combat, df_bomb)
    print(f"  -> {len(documents)} documentos gerados")

    # Divide documentos muito grandes em chunks menores
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"  -> {len(chunks)} chunks após text splitting")

    # --- Requisito 2 e 3: Geração de Embeddings + Indexação Vetorial ---
    print("[Sprint 5] Gerando embeddings com Ollama e indexando no Milvus...")
    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={"uri": MILVUS_URI},
        collection_name=COLLECTION_NAME,
        drop_old=True,  # Recria a coleção do zero
    )
    print("[Sprint 5] Index funcional -- embeddings gerados e indexados com sucesso!")
    print("[Sprint 5] Processo automatizado -- pipeline completo executado!")

    return vector_store


# ==========================================
# PIPELINE RAG: RECUPERAÇÃO + GERAÇÃO
# ==========================================

# Cache global do vector store para não re-indexar a cada query
_vector_store: Milvus | None = None


def get_vector_store() -> Milvus:
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store()
    return _vector_store


def query(model_name: str, question: str) -> tuple[str, str]:
    """
    Executa a pipeline RAG completa:
      1. Busca semântica no Milvus pelos chunks mais relevantes
      2. Monta o prompt com o contexto recuperado
      3. Envia ao modelo Ollama selecionado e retorna a resposta

    Retorna: (resposta_do_llm, logs_do_pipeline)
    """
    logs: list[str] = []

    try:
        # --- Recuperação (R do RAG) ---
        logs.append("Iniciando busca vetorial no Milvus...")
        vector_store = get_vector_store()
        retrieved_docs = vector_store.similarity_search(question, k=5)

        if not retrieved_docs:
            return "Nenhum contexto relevante encontrado para responder à pergunta.", "\n".join(logs)

        logs.append(f"{len(retrieved_docs)} documentos recuperados:\n")
        for i, doc in enumerate(retrieved_docs, 1):
            fonte = doc.metadata.get("source", "desconhecida")
            logs.append(f"  [{i}] (fonte: {fonte}) {doc.page_content[:120]}...")

        # --- Montagem do Contexto ---
        context = "\n".join(doc.page_content for doc in retrieved_docs)

        prompt = f"""Você é um assistente especialista em CS:GO. Use APENAS o contexto abaixo para responder.
Se a resposta não estiver no contexto, diga: "Não encontrei essa informação nos dados disponíveis."
Responda em português, de forma clara e objetiva.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""

        # --- Geração (G do RAG) ---
        logs.append(f"\nGerando resposta com o modelo '{model_name}' via Ollama...")
        llm = Ollama(model=model_name, base_url=OLLAMA_BASE_URL)
        answer = llm.invoke(prompt)

        logs.append("Resposta gerada com sucesso.")
        return answer.strip(), "\n".join(logs)

    except Exception as e:
        error_msg = f"Erro na pipeline RAG: {e}"
        logs.append(f"{error_msg}")
        return error_msg, "\n".join(logs)


# ==========================================
# EXECUÇÃO DIRETA (TESTE / SETUP INICIAL)
# ==========================================

if __name__ == "__main__":
    print("=== Sprint 5 -- Pipeline de Embeddings ===")
    print("Forçando rebuild do índice vetorial...")
    vs = build_vector_store(force_rebuild=True)

    print("\nTeste de busca vetorial:")
    pergunta = "Qual arma causa mais dano médio na cabeça?"
    resultados = vs.similarity_search(pergunta, k=3)
    for i, r in enumerate(resultados, 1):
        print(f"  Resultado {i}: {r.page_content[:100]}...")

    print("\nPipeline do Sprint 5 validada com sucesso!")
