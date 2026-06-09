import os
import requests

from pymilvus import MilvusClient

# ==========================================
# CONFIGURAÇÕES
# ==========================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "csgo_rag"
EMBEDDING_MODEL = "nomic-embed-text"

SUPPORTED_MODELS = [
    "llama3.2",
    "llama3",
    "mistral",
    "gemma2",
    "phi3",
]

# ==========================================
# VECTOR STORE (compatível com pymilvus 2.4.x)
# ==========================================

def get_vector_store():
    return MilvusClient(uri=MILVUS_URI)

def build_vector_store(force_rebuild: bool = False):
    print("[main] Vector store via pymilvus direto.")
    return get_vector_store()

# ==========================================
# PIPELINE RAG
# ==========================================

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


def query(model_name: str, question: str) -> tuple[str, str]:
    logs: list[str] = []

    try:
        logs.append("Iniciando busca vetorial no Milvus...")
        client = MilvusClient(uri=MILVUS_URI)

        embedding = _embed(question)
        results = client.search(
            collection_name=COLLECTION_NAME,
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
        prompt = f"""Você é um assistente especialista em CS:GO.
Use o contexto abaixo para responder. O contexto contém estatísticas reais de partidas.
Analise os dados disponíveis e responda com base neles.
Responda em português, de forma clara e objetiva.

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

        logs.append("Resposta gerada com sucesso.")
        return answer, "\n".join(logs)

    except Exception as e:
        error_msg = f"Erro na pipeline RAG: {e}"
        logs.append(error_msg)
        return error_msg, "\n".join(logs)
    