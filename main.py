import os
import requests
from pymilvus import MilvusClient

# ==========================================
# CONFIGURAÇÕES
# ==========================================

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MILVUS_URI       = os.getenv("MILVUS_URI",       "http://localhost:19530")
COLLECTION_NAME  = "csgo_rag"
METADATA_COLLECTION = "csgo_metadata"
EMBEDDING_MODEL  = "nomic-embed-text"

SUPPORTED_MODELS = [
    "llama3.2",
    "llama3",
    "mistral",
    "gemma2",
    "phi3",
]

# Palavras que indicam pergunta sobre o sistema
METADATA_KEYWORDS = [
    "modelo", "modelos", "arquitetura", "sistema", "api", "endpoint",
    "pipeline", "tecnologia", "governança", "mlflow", "milvus", "ollama",
    "gradio", "fastapi", "docker", "bronze", "silver", "gold", "sprint",
    "projeto", "facens", "dataset", "kaggle", "embedding", "treinamento",
    "xgboost", "randomforest", "r2", "rmse", "versão", "infraestrutura",
    "quais modelos", "como funciona", "o que é", "qual tecnologia",
]

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def get_vector_store():
    return MilvusClient(uri=MILVUS_URI)

def build_vector_store(force_rebuild: bool = False):
    return get_vector_store()

def _is_metadata_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in METADATA_KEYWORDS)

def _embed(text: str) -> list:
    translations = {
        "cabeça": "Head", "peito": "Chest", "perna": "Leg",
        "braço": "Arm", "pescoço": "Neck", "estômago": "Stomach",
        "terrorista": "Terrorist", "contra-terrorista": "CounterTerrorist",
        "CT": "CounterTerrorist", "TR": "Terrorist",
    }
    for pt, en in translations.items():
        text = text.replace(pt, en)

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

# ==========================================
# PIPELINE RAG
# ==========================================

def query(model_name: str, question: str) -> tuple[str, str]:
    logs: list[str] = []

    try:
        import mlflow
        import time as _time

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("rag-queries")

        client = MilvusClient(uri=MILVUS_URI)
        embedding = _embed(question)

        # Decide qual coleção usar
        if _is_metadata_question(question):
            collection = METADATA_COLLECTION
            logs.append("Buscando em metadados do sistema...")
        else:
            collection = COLLECTION_NAME
            logs.append("Iniciando busca vetorial no Milvus...")

        if not client.has_collection(collection):
            collection = COLLECTION_NAME
            logs.append("Coleção de metadados não encontrada, usando dados de CS:GO...")

        start = _time.time()

        results = client.search(
            collection_name=collection,
            data=[embedding],
            limit=10,
            output_fields=["text"],
        )

        chunks = [hit["entity"].get("text", "") for hit in results[0]]
        chunks = [c for c in chunks if c]

        if not chunks:
            return "Nenhum contexto relevante encontrado.", "\n".join(logs)

        logs.append(f"{len(chunks)} documentos recuperados:\n")
        for i, chunk in enumerate(chunks[:5], 1):
            logs.append(f"  [{i}] {chunk[:120]}...")

        context = "\n".join(chunks)
        prompt = f"""Você é um assistente especialista em CS:GO e no sistema StrikeMetrics Solutions.
Use o contexto abaixo para responder. Responda em português, de forma clara e objetiva.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""

        logs.append(f"\nGerando resposta com o modelo '{model_name}' via Ollama...")
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        answer = resp.json()["response"].strip()
        latency = int((_time.time() - start) * 1000)

        # Relevância simples: proporção de chunks com palavras da pergunta
        question_words = set(question.lower().split())
        relevant = sum(
            1 for c in chunks
            if any(w in c.lower() for w in question_words if len(w) > 3)
        )
        relevance_score = round(relevant / len(chunks), 2) if chunks else 0.0

        # Loga no MLflow
        try:
            with mlflow.start_run():
                mlflow.log_param("model", model_name)
                mlflow.log_param("collection", collection)
                mlflow.log_param("question_length", len(question))
                mlflow.log_metric("latency_ms", latency)
                mlflow.log_metric("chunks_retrieved", len(chunks))
                mlflow.log_metric("relevance_score", relevance_score)
                mlflow.log_param("question", question[:250])
        except Exception as mlflow_err:
            logs.append(f"[MLflow] aviso: {mlflow_err}")

        logs.append("Resposta gerada com sucesso.")
        return answer, "\n".join(logs)

    except Exception as e:
        error_msg = f"Erro na pipeline RAG: {e}"
        logs.append(error_msg)
        return error_msg, "\n".join(logs)
    