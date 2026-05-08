import os
import time
import pandas as pd
from minio import Minio
from io import BytesIO
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pymilvus import MilvusClient

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY",  "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY",  "minioadmin")
MILVUS_URI       = os.getenv("MILVUS_URI",        "http://localhost:19530")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
BUCKET           = "csgodatalake"
COLLECTION_NAME  = "csgo_rag"
EMBED_MODEL      = "nomic-embed-text"


def row_to_text_combat(row: pd.Series) -> str:
    return (
        f"Arma: {row['wp']} | Rank do atacante: {row['att_rank']} | "
        f"Hitbox: {row['hitbox']} | Lado: {row['att_side']} | "
        f"Dano médio (HP): {row['avg_hp_damage']:.2f} | "
        f"Dano total (HP): {row['total_hp_damage']} | "
        f"Dano total (Armadura): {row['total_armor_damage']} | "
        f"Total de acertos: {row['total_hits']}"
    )


def row_to_text_bomb(row: pd.Series) -> str:
    return (
        f"Site da bomba: {row['bomb_site']} | Lado: {row['att_side']} | "
        f"Arma: {row['wp']} | "
        f"Dano pós-plantação: {row['damage_post_plant']} | "
        f"Dano médio pós-plantação: {row['avg_damage_post_plant']:.2f} | "
        f"Acertos pós-plantação: {row['hits_post_plant']}"
    )


def wait_for_ollama(max_retries: int = 12, delay: int = 15):
    import urllib.request
    url = f"{OLLAMA_BASE_URL}/api/tags"
    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlopen(url, timeout=5)
            print("   Ollama disponível.")
            return
        except Exception:
            print(f"   Aguardando Ollama... ({attempt}/{max_retries})")
            time.sleep(delay)
    raise RuntimeError("Ollama não ficou disponível a tempo.")


print("=" * 55)
print("  Pipeline de Embeddings — Sprint 5")
print("=" * 55)

print("\n[1/5] Verificando disponibilidade do Ollama...")
wait_for_ollama()

print("\n[2/5] Lendo dados Gold do MinIO...")
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

response = minio_client.get_object(BUCKET, "gold/combat_context_stats.parquet")
df_combat = pd.read_parquet(BytesIO(response.read()))

response = minio_client.get_object(BUCKET, "gold/bomb_context_stats.parquet")
df_bomb = pd.read_parquet(BytesIO(response.read()))

print(f"   Combat stats: {len(df_combat)} linhas")
print(f"   Bomb stats:   {len(df_bomb)} linhas")

print("\n[3/5] Convertendo linhas em documentos de texto...")
combat_docs = [
    Document(
        page_content=row_to_text_combat(row),
        metadata={"source": "combat_stats", "weapon": row["wp"], "hitbox": row["hitbox"]},
    )
    for _, row in df_combat.iterrows()
]

bomb_docs = [
    Document(
        page_content=row_to_text_bomb(row),
        metadata={"source": "bomb_stats", "weapon": row["wp"], "site": row["bomb_site"]},
    )
    for _, row in df_bomb.iterrows()
]

all_docs = combat_docs + bomb_docs
print(f"   Total de documentos: {len(all_docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)
print(f"   Total após chunking: {len(chunks)} chunks")

print(f"\n[4/5] Gerando embeddings com '{EMBED_MODEL}' e indexando no Milvus...")
print("   (Isso pode levar alguns minutos dependendo do volume de dados...)")

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

client = MilvusClient(uri=MILVUS_URI)

# Remove coleção se existir
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)
    print(f"   Coleção '{COLLECTION_NAME}' anterior removida.")

# Cria coleção
client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=768,
    metric_type="COSINE",
    auto_id=True
)
print(f"   Coleção '{COLLECTION_NAME}' criada com dimensão 768")

# Gera textos e embeddings
texts = [doc.page_content for doc in chunks]
print(f"   Gerando {len(texts)} embeddings...")

embeddings_list = embeddings.embed_documents(texts)
print(f"   Embeddings gerados: {len(embeddings_list)}")

# Prepara dados para inserção
data = []
for text, embedding in zip(texts, embeddings_list):
    data.append({
        "text": text,
        "vector": embedding
    })

# Insere em lote
print(f"   Inserindo {len(data)} chunks no Milvus...")
insert_result = client.insert(COLLECTION_NAME, data)
print(f"   ✓ {len(data)} chunks indexados na coleção '{COLLECTION_NAME}'!")

# Aguarda persistência (pymilvus 2.4.x faz flush automático, só precisa de tempo)
print("   Aguardando persistência...")
time.sleep(10)

# Carrega coleção em memória
client.load_collection(COLLECTION_NAME)
print("   ✓ Coleção carregada em memória — pronta para queries.")

print("\n   Verificando dados inseridos...")
try:
    # Método 1: Tentar buscar registros
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="id >= 0",
        output_fields=["id", "text"],
        limit=5
    )
    
    if results:
        print(f"   ✓ CONFIRMADO: {len(results)} registros encontrados (amostra)")
        print(f"   ✓ IDs dos primeiros registros: {[r['id'] for r in results]}")
    else:
        print(f"   ⚠️ Query retornou 0 registros")
        
    # Método 2: Estatísticas da coleção
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"   Estatísticas da coleção: {stats}")
    
    # Método 3: Contagem via query (mais confiável)
    count_result = client.query(
        collection_name=COLLECTION_NAME,
        filter="id >= 0",
        output_fields=["id"],
        limit=10000
    )
    print(f"   ✓ Contagem real via query: {len(count_result)} entidades")
    
except Exception as e:
    print(f"   ⚠️ Erro na verificação: {e}")
    print("   Os dados podem ter sido inseridos mesmo assim.")

print("\n[5/5] Concluido! Index vetorial pronto para consultas RAG.")