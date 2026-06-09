import os
import time
import requests
from pymilvus import MilvusClient

MILVUS_URI      = os.getenv("MILVUS_URI",       "http://localhost:19530")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
COLLECTION_NAME = "csgo_metadata"
EMBED_MODEL     = "nomic-embed-text"

METADATA_DOCS = [
    "Os modelos de linguagem disponíveis no sistema são: llama3.2, llama3, mistral, gemma2 e phi3. O modelo padrão recomendado é o llama3.2.",
    "O modelo de embedding utilizado para vetorização dos dados é o nomic-embed-text, rodando localmente via Ollama.",
    "O modelo XGBoost foi treinado com dados de combate do CS:GO e atingiu R² de 0.97 e RMSE de 2.92.",
    "O modelo RandomForest foi treinado com dados de bomba plantada e atingiu R² de 0.69 e RMSE de 10.15.",
    "A arquitetura do sistema utiliza: MinIO para armazenamento, PostgreSQL para metadados, Milvus para vetores, Ollama para LLMs e MLflow para rastreamento de experimentos.",
    "O sistema utiliza arquitetura RAG que combina busca semântica vetorial com geração de texto por LLM.",
    "A infraestrutura roda em containers Docker orquestrados via Docker Compose.",
    "O sistema é 100% local, sem dependência de APIs externas. Todos os modelos rodam via Ollama.",
    "O pipeline de dados segue arquitetura Medallion: Bronze (CSVs brutos), Silver (Parquet limpo) e Gold (dados agregados para RAG).",
    "Os dados são do dataset CS:GO Matchmaking Damage do Kaggle com registros de dano de partidas reais.",
    "A camada Gold contém duas tabelas: combat_context_stats com estatísticas de combate e bomb_context_stats com análise de bomba plantada.",
    "O pipeline processa aproximadamente 1165 chunks de texto vetorizados no Milvus.",
    "O MLflow registra todos os experimentos com parâmetros, métricas e artefatos salvos no PostgreSQL.",
    "Os metadados de auditoria são armazenados no PostgreSQL para rastreabilidade completa.",
    "A API expõe os endpoints: GET / health check, GET /models lista modelos, GET /metadata informações do sistema, POST /query consulta RAG, POST /rebuild-index reconstrói índice.",
    "A documentação da API está disponível em http://localhost:8000/docs via Swagger UI.",
    "A interface de usuário é construída com Gradio e acessível em http://localhost:7860.",
    "O projeto StrikeMetrics Solutions foi desenvolvido na Facens seguindo metodologia Scrum com 12 sprints.",
    "O sistema foi desenvolvido para rodar com mínimo 16GB RAM sem GPU dedicada.",
    "O domínio do sistema é análise de performance de eSports com foco em CS:GO.",
]

def embed_text(text: str) -> list:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

def main():
    print("=" * 55)
    print("  Vetorização de Metadados — Governança")
    print("=" * 55)

    print("Aguardando Ollama...")
    for i in range(12):
        try:
            requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            print("Ollama disponível.")
            break
        except Exception:
            print(f"  Tentativa {i+1}/12...")
            time.sleep(10)

    client = MilvusClient(uri=MILVUS_URI)

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=768,
        metric_type="COSINE",
        auto_id=True,
    )
    print(f"Coleção '{COLLECTION_NAME}' criada.")

    data = []
    for i, doc in enumerate(METADATA_DOCS):
        embedding = embed_text(doc)
        data.append({"text": doc, "vector": embedding})
        print(f"  [{i+1}/{len(METADATA_DOCS)}] OK")

    client.insert(COLLECTION_NAME, data)
    time.sleep(5)
    client.load_collection(COLLECTION_NAME)

    print(f"\n✓ {len(data)} documentos indexados em '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()
