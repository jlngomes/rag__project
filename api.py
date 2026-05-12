"""
API FastAPI — StrikeMetrics Solutions RAG CS:GO
Sprint 7 (AC2): /query, /metadata, validação de input/output.
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from main import SUPPORTED_MODELS, build_vector_store, get_vector_store, query


# ==========================================
# SCHEMAS — Input
# ==========================================

class QueryRequest(BaseModel):
    model_name: str = Field(
        default="llama3.2",
        description="Nome do modelo Ollama a ser usado na geração.",
        examples=["llama3.2", "mistral"],
    )
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Pergunta em linguagem natural sobre estatísticas de CS:GO (5–500 caracteres).",
        examples=["Qual arma causa mais dano médio na cabeça?"],
    )

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in SUPPORTED_MODELS:
            raise ValueError(
                f"Modelo '{v}' não suportado. Modelos disponíveis: {SUPPORTED_MODELS}"
            )
        return v

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("A pergunta não pode ser composta apenas de espaços.")
        if not any(c.isalpha() for c in v):
            raise ValueError("A pergunta deve conter ao menos uma letra.")
        return v


# ==========================================
# SCHEMAS — Output
# ==========================================

class QueryResponse(BaseModel):
    answer: str = Field(description="Resposta gerada pelo LLM.")
    logs: str = Field(description="Logs internos do pipeline RAG.")

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("O modelo retornou uma resposta vazia.")
        return v.strip()


class MetadataResponse(BaseModel):
    api_version: str = Field(description="Versão da API.")
    supported_models: list[str] = Field(description="Modelos Ollama disponíveis.")
    vector_store: str = Field(description="Tecnologia de vector store utilizada.")
    embedding_model: str = Field(description="Modelo de embedding utilizado.")
    data_sources: list[str] = Field(description="Fontes de dados indexadas.")
    endpoints: list[str] = Field(description="Endpoints disponíveis nesta API.")
    timestamp: str = Field(description="Data/hora da consulta (ISO 8601).")


class RebuildResponse(BaseModel):
    message: str


class ModelsResponse(BaseModel):
    models: list[str]


# ==========================================
# LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega (ou cria) o índice vetorial uma única vez na inicialização."""
    print("[API] Inicializando vector store...")
    get_vector_store()
    print("[API] Vector store pronto.")
    yield
    print("[API] Encerrando API.")


# ==========================================
# APLICAÇÃO
# ==========================================

app = FastAPI(
    title="StrikeMetrics RAG API",
    description=(
        "API para consultas RAG sobre estatísticas de combate e bomba do CS:GO. "
        "Utiliza Milvus como vector store e modelos Ollama locais para geração. "
        "Sprint 7 — AC2."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/", tags=["health"])
def root():
    """Health check básico."""
    return {"status": "ok", "service": "StrikeMetrics RAG API"}


@app.get("/models", response_model=ModelsResponse, tags=["config"])
def list_models():
    """Retorna a lista de modelos Ollama disponíveis."""
    return ModelsResponse(models=SUPPORTED_MODELS)


@app.get("/metadata", response_model=MetadataResponse, tags=["config"])
def get_metadata():
    """
    Retorna metadados da API: versão, modelos suportados, vector store,
    modelo de embedding, fontes de dados e endpoints disponíveis.
    """
    return MetadataResponse(
        api_version="1.0.0",
        supported_models=SUPPORTED_MODELS,
        vector_store="Milvus",
        embedding_model="nomic-embed-text (Ollama)",
        data_sources=[
            "csgo_combat_stats (Kaggle → MinIO Gold)",
            "csgo_bomb_stats (Kaggle → MinIO Gold)",
        ],
        endpoints=["/", "/models", "/metadata", "/query", "/rebuild-index"],
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def run_query(body: QueryRequest):
    """
    Executa o pipeline RAG completo:
    1. Busca semântica no Milvus pelos chunks mais relevantes.
    2. Monta o prompt com o contexto recuperado.
    3. Gera a resposta com o modelo Ollama escolhido.

    Validações de input: model_name em /models; question entre 5–500 chars.
    Validações de output: resposta não pode ser vazia.
    """
    answer, logs = query(body.model_name, body.question)
    return QueryResponse(answer=answer, logs=logs)


@app.post("/rebuild-index", response_model=RebuildResponse, tags=["admin"])
def rebuild_index():
    """
    Força a reconstrução completa do índice vetorial no Milvus.
    Use com cuidado: apaga e recria toda a coleção.
    """
    try:
        build_vector_store(force_rebuild=True)
        return RebuildResponse(message="Índice vetorial reconstruído com sucesso.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# EXECUÇÃO DIRETA
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
