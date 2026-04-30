import os
import logging
from io import BytesIO, StringIO
from typing import Tuple

import pandas as pd
from minio import Minio
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# ── Modelos disponíveis no dropdown ──────────────────────────────────────────
SUPPORTED_MODELS = [
    "llama3.2",
    "llama3.2:1b",
    "mistral",
]

# ── Configuração do MinIO (sobrescreva por variável de ambiente) ──────────────
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET           = "csgodatalake"
GOLD_FILES       = ["combat_context_stats.parquet", "bomb_context_stats.parquet"]

# ── Dados de exemplo embutidos (usados quando MinIO está offline) ─────────────
_SAMPLE_DATA = [
    # Estatísticas de combate por arma / hitbox / lado
    "arma: ak-47 | patente: 3 | hitbox: cabeca | lado: T | dano_hp_total: 485200 | dano_hp_medio: 97.2 | total_acertos: 4992",
    "arma: ak-47 | patente: 3 | hitbox: torso | lado: T | dano_hp_total: 1247800 | dano_hp_medio: 28.4 | total_acertos: 43930",
    "arma: ak-47 | patente: 3 | hitbox: estomago | lado: T | dano_hp_total: 623400 | dano_hp_medio: 21.8 | total_acertos: 28600",
    "arma: ak-47 | patente: 3 | hitbox: pernas | lado: T | dano_hp_total: 312000 | dano_hp_medio: 15.6 | total_acertos: 20000",
    "arma: m4a1-s | patente: 4 | hitbox: cabeca | lado: CT | dano_hp_total: 357900 | dano_hp_medio: 85.2 | total_acertos: 4200",
    "arma: m4a1-s | patente: 4 | hitbox: torso | lado: CT | dano_hp_total: 980000 | dano_hp_medio: 25.1 | total_acertos: 39044",
    "arma: m4a4 | patente: 4 | hitbox: cabeca | lado: CT | dano_hp_total: 340000 | dano_hp_medio: 82.0 | total_acertos: 4146",
    "arma: m4a4 | patente: 4 | hitbox: torso | lado: CT | dano_hp_total: 1020000 | dano_hp_medio: 27.5 | total_acertos: 37090",
    "arma: awp | patente: 5 | hitbox: cabeca | lado: T | dano_hp_total: 312500 | dano_hp_medio: 125.0 | total_acertos: 2500",
    "arma: awp | patente: 5 | hitbox: torso | lado: T | dano_hp_total: 275000 | dano_hp_medio: 110.0 | total_acertos: 2500",
    "arma: awp | patente: 5 | hitbox: estomago | lado: T | dano_hp_total: 247500 | dano_hp_medio: 99.0 | total_acertos: 2500",
    "arma: awp | patente: 5 | hitbox: pernas | lado: T | dano_hp_total: 185000 | dano_hp_medio: 74.0 | total_acertos: 2500",
    "arma: deagle | patente: 3 | hitbox: cabeca | lado: T | dano_hp_total: 220000 | dano_hp_medio: 110.0 | total_acertos: 2000",
    "arma: deagle | patente: 3 | hitbox: torso | lado: T | dano_hp_total: 350000 | dano_hp_medio: 53.8 | total_acertos: 6506",
    "arma: usp-s | patente: 4 | hitbox: cabeca | lado: CT | dano_hp_total: 168000 | dano_hp_medio: 84.0 | total_acertos: 2000",
    "arma: usp-s | patente: 4 | hitbox: torso | lado: CT | dano_hp_total: 294000 | dano_hp_medio: 42.0 | total_acertos: 7000",
    "arma: glock | patente: 2 | hitbox: cabeca | lado: T | dano_hp_total: 120000 | dano_hp_medio: 60.0 | total_acertos: 2000",
    "arma: glock | patente: 2 | hitbox: torso | lado: T | dano_hp_total: 231000 | dano_hp_medio: 33.0 | total_acertos: 7000",
    "arma: mp9 | patente: 2 | hitbox: cabeca | lado: CT | dano_hp_total: 195000 | dano_hp_medio: 78.0 | total_acertos: 2500",
    "arma: mp9 | patente: 2 | hitbox: torso | lado: CT | dano_hp_total: 520000 | dano_hp_medio: 26.0 | total_acertos: 20000",
    "arma: mac-10 | patente: 2 | hitbox: cabeca | lado: T | dano_hp_total: 190000 | dano_hp_medio: 76.0 | total_acertos: 2500",
    "arma: sg553 | patente: 4 | hitbox: cabeca | lado: T | dano_hp_total: 280000 | dano_hp_medio: 112.0 | total_acertos: 2500",
    "arma: sg553 | patente: 4 | hitbox: torso | lado: T | dano_hp_total: 980000 | dano_hp_medio: 36.3 | total_acertos: 27000",
    "arma: famas | patente: 3 | hitbox: cabeca | lado: CT | dano_hp_total: 195000 | dano_hp_medio: 78.0 | total_acertos: 2500",
    "arma: galil | patente: 3 | hitbox: cabeca | lado: T | dano_hp_total: 188000 | dano_hp_medio: 75.2 | total_acertos: 2500",
    "arma: p250 | patente: 2 | hitbox: cabeca | lado: T | dano_hp_total: 174000 | dano_hp_medio: 87.0 | total_acertos: 2000",
    # Dados pós-plant (bomba plantada)
    "site: A | lado: T | arma: ak-47 | dano_pos_plant: 198000 | dano_medio_pos_plant: 31.2 | acertos_pos_plant: 6346",
    "site: B | lado: T | arma: ak-47 | dano_pos_plant: 175000 | dano_medio_pos_plant: 30.8 | acertos_pos_plant: 5682",
    "site: A | lado: CT | arma: m4a1-s | dano_pos_plant: 145000 | dano_medio_pos_plant: 28.4 | acertos_pos_plant: 5105",
    "site: B | lado: CT | arma: m4a4 | dano_pos_plant: 138000 | dano_medio_pos_plant: 27.9 | acertos_pos_plant: 4946",
    "site: A | lado: T | arma: awp | dano_pos_plant: 88000 | dano_medio_pos_plant: 105.0 | acertos_pos_plant: 838",
    "site: B | lado: T | arma: awp | dano_pos_plant: 76000 | dano_medio_pos_plant: 104.0 | acertos_pos_plant: 731",
    "site: A | lado: CT | arma: awp | dano_pos_plant: 62000 | dano_medio_pos_plant: 109.0 | acertos_pos_plant: 569",
]

# ── Prompt ────────────────────────────────────────────────────────────────────
_PROMPT = PromptTemplate.from_template(
    """Você é um analista especializado em CS:GO para a StrikeMetrics Solutions.
Use APENAS as estatísticas de combate abaixo para responder com precisão e em português.
Se os dados não forem suficientes, diga claramente que não há informação disponível.

Contexto recuperado:
{context}

Pergunta: {question}
Resposta:"""
)


# ── Captura de logs ───────────────────────────────────────────────────────────

class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self._buf = StringIO()
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        self._buf.write(self.format(record) + "\n")

    def get_logs(self) -> str:
        return self._buf.getvalue()


# ── Carregamento de dados ─────────────────────────────────────────────────────

def _load_docs(logger: logging.Logger) -> list[Document]:
    """Tenta carregar dados do MinIO. Usa dados de exemplo se MinIO estiver offline."""
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False,
        )
        docs: list[Document] = []
        for filename in GOLD_FILES:
            path = f"gold/{filename}"
            resp = client.get_object(BUCKET, path)
            df = pd.read_parquet(BytesIO(resp.read()))
            for _, row in df.iterrows():
                content = " | ".join(f"{col}: {val}" for col, val in row.items())
                docs.append(Document(page_content=content, metadata={"source": filename}))
            logger.info(f"MinIO: {len(df)} registros carregados de {path}")
        return docs

    except Exception as exc:
        logger.warning(f"MinIO indisponível ({exc}) — usando dados de exemplo embutidos.")
        docs = [Document(page_content=d, metadata={"source": "sample"}) for d in _SAMPLE_DATA]
        logger.info(f"Dados de exemplo: {len(docs)} documentos carregados.")
        return docs


# ── Função pública ────────────────────────────────────────────────────────────

def query(model_name: str, question: str) -> Tuple[str, str]:
    """Executa o pipeline RAG e retorna (resposta, logs)."""
    handler = _LogCapture()
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    try:
        log = logging.getLogger(__name__)
        log.info(f"Pipeline iniciado | modelo={model_name}")

        docs = _load_docs(log)

        retriever = BM25Retriever.from_documents(docs, k=6)
        log.info(f"BM25 pronto | buscando documentos relevantes para: '{question}'")

        retrieved = retriever.invoke(question)
        log.info(f"Documentos recuperados: {len(retrieved)}")
        for i, doc in enumerate(retrieved, 1):
            preview = doc.page_content[:100]
            log.info(f"  [Doc {i}] {preview}…")

        context = "\n".join(d.page_content for d in retrieved)

        llm = ChatOllama(model=model_name, temperature=0.2)
        log.info("Enviando contexto + pergunta ao LLM (Ollama)...")

        chain = _PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        log.info("Resposta gerada com sucesso.")
        return answer, handler.get_logs()

    except Exception as exc:
        logging.getLogger(__name__).error(f"Erro: {exc}", exc_info=True)
        return f"Erro: {exc}", handler.get_logs()

    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)
