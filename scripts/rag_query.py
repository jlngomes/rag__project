import os
import time
import requests
from pymilvus import MilvusClient

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
COLLECTION_NAME = "csgo_rag"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
TOP_K = 12


def normalize_question(question: str) -> str:
    translations = {
        "cabeça": "Head",
        "peito": "Chest",
        "perna": "Leg",
        "braço": "Arm",
        "pescoço": "Neck",
        "estômago": "Stomach",
        "terrorista": "Terrorist",
        "contra-terrorista": "CounterTerrorist",
        "CT": "CounterTerrorist",
        "TR": "Terrorist",
    }
    for pt, en in translations.items():
        question = question.replace(pt, en)
    return question


def embed_question(question: str) -> list:
    question = normalize_question(question)
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": question},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["embedding"]


def search_milvus(embedding: list) -> list:
    client = MilvusClient(uri=MILVUS_URI)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        limit=TOP_K,
        output_fields=["text"]
    )
    return [hit["entity"]["text"] for hit in results[0]]


def generate_answer(question: str, chunks: list) -> str:
    context = "\n".join(chunks)
    prompt = f"""Você é um assistente especialista em CS:GO.
Use o contexto abaixo para responder. O contexto contém estatísticas reais de partidas.
Analise os dados disponíveis e responda com base neles, mesmo que não seja uma resposta perfeita.
Responda em português, de forma clara e objetiva.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=300
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def check_services():
    print("Verificando serviços...")
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        print("  Ollama OK")
    except Exception:
        print("  ERRO: Ollama não está respondendo")
        return False

    try:
        client = MilvusClient(uri=MILVUS_URI)
        if not client.has_collection(COLLECTION_NAME):
            print(f"  ERRO: Coleção '{COLLECTION_NAME}' não existe no Milvus")
            return False
        print("  Milvus OK")
    except Exception as e:
        print(f"  ERRO: Milvus — {e}")
        return False

    return True


def check_llm_model():
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    models = [m["name"] for m in response.json().get("models", [])]
    for m in models:
        if LLM_MODEL in m:
            return True
    return False


def pull_llm_model():
    print(f"  Baixando modelo '{LLM_MODEL}' (pode demorar alguns minutos)...")
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/pull",
        json={"name": LLM_MODEL},
        stream=True,
        timeout=600
    )
    for line in response.iter_lines():
        if line and b"success" in line:
            print(f"  Modelo '{LLM_MODEL}' pronto.")
            return


def run():
    print("=" * 55)
    print("  StrikeMetrics Solutions — RAG CS:GO")
    print("  Sprint 6 — Consulta via Terminal")
    print("=" * 55)

    if not check_services():
        print("\nSuba os serviços com: make up")
        return

    if not check_llm_model():
        print(f"  Modelo '{LLM_MODEL}' não encontrado.")
        pull_llm_model()

    print("\nServiços prontos. Digite 'sair' para encerrar.\n")

    while True:
        try:
            question = input("Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if question.lower() in ("sair", "exit", "quit"):
            print("Encerrando.")
            break

        if not question:
            continue

        print("\nBuscando contexto no Milvus...")
        start = time.time()

        try:
            embedding = embed_question(question)
            chunks = search_milvus(embedding)

            print(f"  {len(chunks)} chunks recuperados")

            print("\nGerando resposta com Ollama (pode demorar em CPU)...\n")

            answer = generate_answer(question, chunks)
            elapsed = time.time() - start

            print(f"Resposta:\n{answer}")
            print(f"\n[{elapsed:.1f}s] Fontes: {len(chunks)} chunks do Milvus")
            print("-" * 55)

        except Exception as e:
            print(f"Erro: {e}")
            print("-" * 55)


if __name__ == "__main__":
    run()